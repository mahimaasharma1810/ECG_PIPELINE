#!/usr/bin/env python3
"""generate_validation_pdf.py -- one fillable clinician validation PDF per ECG
segment, ready to hand to a reviewer and scan back.

    python3 -m ecg_pipeline.generate_validation_pdf \
        --report data/reports/vitalpatch/184B27/<segment>.json \
        --ecg-csv data/raw/vitalpatch/Patch_184B27/<file>.csv \
        --output data/reports/validation_pdfs/ \
        [--vitals-root data/vitals_downloads] \
        [--ground-truth labels.csv]

WHAT THIS DOES NOT DO: it does not decide, recompute, or alter any
classification, risk level, or rule-trace value. Every number rendered comes
either from the supplied report JSON or from a deterministic re-run of the
frozen pipeline on the same input file.

WHY IT RE-RUNS THE PIPELINE: the saved report JSON stores AGGREGATE beat counts
only. It does not store the filtered waveform, R-peak sample positions, or the
per-beat N/S/V/F/Q sequence -- none of which exist anywhere on disk in reloadable
form, and all of which are needed to draw an ECG strip with beat markers. The
re-run uses the same frozen classifier and the same code path
(agent_bridge.run_full_report), so it is reproduction, not a new judgement.

SAFETY NET: the fresh result is compared against the supplied report JSON. If the
final risk level or deciding rule differ, a prominent banner says so on page 1
rather than silently rendering one version's waveform beside the other's verdict.
This is expected for reports saved before a later pipeline fix (e.g. the
2026-08-02 local_hrv gap guard) and must never be hidden.

AMPLITUDE IS UNCALIBRATED. VitalPatch/ProRhythm expose no published mV-per-count
constant, so the vertical axis is raw device units and standard 0.1/0.5 mV
amplitude boxes are deliberately NOT drawn. The TIME axis is standard ECG paper
(0.04 s small box, 0.2 s large box) and is drawn as such. Same decision as
annotation/generate_annotation_pdf.py -- do not "fix" one without the other.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas

# Marker-anchored, not hop-counted -- see scripts/verify_sqi_gate.py.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ecg_pipeline").is_dir())
sys.path.insert(0, str(REPO_ROOT))

from ecg_pipeline.ecg_pipeline_core import (  # noqa: E402
    MODELS_DIR, RISK, TARGET_FS, BEATS, apply_filter_chain, detect_and_segment,
    parse_prorhythm_ecg, parse_vitalpatch_ecg, run_sqi_gate, to_target_rate,
)
from ecg_pipeline.agent_bridge import (  # noqa: E402
    beat_summary_pct, load_classifier, run_full_report,
)

PAGE_W, PAGE_H = A4
MARGIN = 38
FONT = "Helvetica"
FONT_B = "Helvetica-Bold"

# Verdict banner colours are specified by the review protocol, not chosen here.
VERDICT_COLORS = {
    "CRITICAL": "#CC0000",
    "HIGH": "#FF8C00",
    "MEDIUM": "#B8860B",
    "LOW": "#006400",
    "NOT_ASSESSABLE": "#57606a",
}
# Beat-class palette is shared with annotation/generate_annotation_pdf.py so a
# clinician reviewing both documents never sees the same class in two colours.
CLASS_COLORS = {"N": "#1a7f37", "S": "#9a6700", "V": "#cf222e",
                "F": "#8250df", "Q": "#57606a"}
CLASS_NAMES = {
    "N": "Normal sinus beat",
    "S": "Supraventricular (PAC, junctional)",
    "V": "Ventricular (PVC, ectopic)",
    "F": "Fusion beat",
    "Q": "Artifact / rejected by quality gate",
}
# Per-class reliability from the frozen model's DS2 held-out evaluation
# (recomputed 2026-08-03 under current code -- see
# docs/CLASSIFIER_EVAL_MITDB_SVDB_2026-08-03.md). Shown so a reviewer is never
# asked to trust a count the model cannot support.
CLASS_F1 = {"N": None, "S": 0.152, "V": 0.830, "F": 0.005, "Q": None}

# Threshold provenance. Checked against the code and docs, NOT copied from any
# summary table -- two entries commonly get stated backwards:
#   * the AFib DETECTOR threshold (RR-CV > 0.10) IS externally validated
#     (LTAFDB, 84 records, 449,749 annotated windows); the AFib BURDEN cutoff
#     (30% of windows) is not.
#   * the SDNN < 20 ms cutoff is NOT from clinical literature -- it is an
#     in-code "conservative low-HRV cutoff".
PROVENANCE = {
    "PVC burden > CRITICAL threshold": (
        "PROJECT-DEFINED", "Not externally validated."),
    "PVC burden > HIGH threshold": (
        "PROJECT-DEFINED", "Not externally validated."),
    "VT run count > 0 (a run = >=3 consecutive V beats)": (
        "MIXED", "The >=3-beat definition of a run is standard clinical usage; "
                 "escalating ANY run straight to CRITICAL is project-defined."),
    "PAC burden > HIGH threshold OR AFib burden > HIGH threshold": (
        "MIXED", "Burden cutoffs (PAC 15%, AFib 30%) are project-defined. The "
                 "underlying AFib detector threshold (RR-CV > 0.10) IS validated "
                 "against LTAFDB (84 records, 449,749 annotated windows)."),
    "Sustained HRV suppression: SDNN < threshold": (
        "PROJECT-DEFINED", "In-code description is 'conservative low-HRV cutoff'. "
                           "Not from clinical literature."),
    "NEWS2 safety override (>= critical threshold)": (
        "CLINICAL LITERATURE", "Royal College of Physicians NEWS2 (2017)."),
    "qSOFA safety override (>= high threshold)": (
        "CLINICAL LITERATURE (threshold only)",
        "Seymour et al. JAMA 2016 sets the >=2 threshold. This system's score is "
        "a 1-of-3-criteria proxy, not a true qSOFA -- see page 6."),
}


# ==========================================================================
# Data assembly
# ==========================================================================

def _parse_ecg(path: Path, segment_id: str | None):
    """Parses an ECG CSV and returns the Recording matching segment_id."""
    name = path.name.lower()
    if "_ecg" in name or "vitalpatch" in str(path).lower():
        recs = parse_vitalpatch_ecg(path)
    else:
        try:
            recs = [parse_prorhythm_ecg(path)]
        except Exception:
            recs = parse_vitalpatch_ecg(path)
    if segment_id:
        for r in recs:
            if r.segment_id == segment_id:
                return r
    if len(recs) == 1:
        return recs[0]
    raise SystemExit(
        f"--ecg-csv holds {len(recs)} segments; none match segment_id "
        f"'{segment_id}'. Available: {[r.segment_id for r in recs]}")


def rebuild_waveform(recording):
    """Re-derives the filtered waveform and per-beat data for drawing.

    Returns (filtered, valid_mask, beats, fs). `valid_mask` is False wherever the
    SQI gate rejected the signal; those samples are interpolated filler in the
    pipeline and are drawn as NO DATA rather than as a trace -- deleted signal
    must never be indistinguishable from asystole on a clinician-facing form.
    """
    keep, _ = run_sqi_gate(recording.signal_mv, recording.timestamps_ms,
                           recording.fs_nominal, clip_value=None)
    sig = recording.signal_mv.copy()
    sig[~keep] = np.nan
    resampled, t_res = to_target_rate(sig, recording.timestamps_ms,
                                      recording.fs_nominal, TARGET_FS)
    valid = ~np.isnan(resampled)
    if len(resampled) == 0 or not valid.any():
        return np.array([]), np.array([], dtype=bool), [], TARGET_FS
    filled = np.interp(t_res, t_res[valid], resampled[valid])
    filtered = apply_filter_chain(
        filled, TARGET_FS, already_bandpass_filtered=recording.already_bandpass_filtered)
    snap = 8 if recording.source == "wfdb" else 15
    beats = detect_and_segment(filtered, TARGET_FS, BEATS,
                               detection_signal=filtered, snap_radius=snap)
    return filtered, valid, beats, TARGET_FS


def pick_window(beats, labels, fs, n_samples, window_s=30.0):
    """Start/end sample index of the `window_s` stretch containing the most
    ectopic (V) beats -- reviewing 30 s of uneventful sinus teaches nothing.
    Falls back to the densest beat region, then to the start."""
    span = int(window_s * fs)
    if n_samples <= span:
        return 0, n_samples
    v_pos = [b.r_peak_idx for b, l in zip(beats, labels) if l == "V"]
    pool = v_pos if v_pos else [b.r_peak_idx for b in beats]
    if not pool:
        return 0, span
    pool = np.asarray(pool)
    best_start, best_n = 0, -1
    for start in range(0, n_samples - span + 1, max(1, int(fs))):  # 1 s stride
        n = int(((pool >= start) & (pool < start + span)).sum())
        if n > best_n:
            best_start, best_n = start, n
    return best_start, best_start + span


# ==========================================================================
# Plot rendering
# ==========================================================================

def _shade_gaps(ax, t, valid_slice, ylo, yhi, label=True):
    """Hatched grey over SQI-rejected stretches, captioned NO DATA."""
    gap = ~valid_slice
    if not gap.any():
        return
    edges = np.diff(gap.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if gap[0]:
        starts = [0] + starts
    if gap[-1]:
        ends = ends + [len(gap) - 1]
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], facecolor="#8c959f", alpha=0.30, zorder=5,
                   hatch="///", edgecolor="#57606a", linewidth=0.0)
        if label and (t[e] - t[s]) > 1.0:
            ax.text(0.5 * (t[s] + t[e]), 0.5 * (ylo + yhi), "NO DATA",
                    ha="center", va="center", fontsize=8, rotation=90,
                    color="#24292f", fontweight="bold", zorder=8)


def render_main_strip(filtered, valid, beats, labels, fs, i0, i1,
                      row_seconds=10.0, width_in=10.2, row_height_in=3.4) -> ImageReader:
    """The review strip, drawn as stacked `row_seconds` panels like a printed
    rhythm strip rather than one compressed line.

    30 s squeezed onto a single axis leaves each QRS a few pixels wide, which is
    useless for judging morphology -- the whole point of showing a clinician the
    waveform. Three 10 s rows triple the horizontal resolution at the same page
    width and match how rhythm strips are actually read.

    Amplitude is shared across rows so beat-to-beat height stays comparable.
    """
    y_all = filtered[i0:i1].astype(float).copy()
    v_all = valid[i0:i1]
    y_all[~v_all] = np.nan
    total_s = len(y_all) / float(fs)
    n_rows = max(1, int(np.ceil(total_s / row_seconds)))

    fin = y_all[np.isfinite(y_all)]
    if fin.size:
        lo, hi = float(fin.min()), float(fin.max())
        pad = max((hi - lo) * 0.10, 0.5)
        ylo, yhi = lo - pad, hi + pad
    else:
        lo = hi = float("nan")
        ylo, yhi = -1.0, 1.0

    fig, axes = plt.subplots(n_rows, 1, figsize=(width_in, row_height_in * n_rows), dpi=150)
    if n_rows == 1:
        axes = [axes]
    per = int(row_seconds * fs)

    for r, ax in enumerate(axes):
        s, e = r * per, min(len(y_all), (r + 1) * per)
        y = y_all[s:e]
        vs = v_all[s:e]
        t = np.arange(len(y)) / float(fs)
        t0_abs = (i0 + s) / float(fs)

        ax.set_xticks(np.arange(0, row_seconds + 1e-9, 1.0))
        ax.set_xticks(np.arange(0, row_seconds + 1e-9, 0.2), minor=True)
        ax.set_yticks(np.linspace(ylo, yhi, 5))
        ax.set_yticks(np.linspace(ylo, yhi, 21), minor=True)
        ax.grid(which="major", color="#f2a0a8", linewidth=0.7, zorder=0)
        ax.grid(which="minor", color="#f8d3d7", linewidth=0.4, zorder=0)
        ax.plot(t, y, color="#111111", linewidth=1.0, zorder=3, solid_joinstyle="round")
        if len(t):
            _shade_gaps(ax, t, vs, ylo, yhi)

        for b, lab in zip(beats, labels):
            bi_abs = b.r_peak_idx - i0
            if not (s <= bi_abs < e):
                continue
            bt = (bi_abs - s) / float(fs)
            col = CLASS_COLORS.get(lab, "#57606a")
            yv = y[bi_abs - s] if np.isfinite(y[bi_abs - s]) else yhi - 0.12 * (yhi - ylo)
            if lab == "Q":
                ax.plot([bt], [yv], marker="x", markersize=7, color=col,
                        markeredgewidth=1.8, zorder=6)
            else:
                ax.plot([bt], [yv], marker="o", markersize=6, markerfacecolor=col,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=6)
            ax.annotate(f"{_beat_index(beats, b)}", xy=(bt, yv), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=6.0,
                        color=col, fontweight="bold", zorder=7)
            ax.annotate(lab, xy=(bt, ylo), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=col,
                        fontweight="bold", zorder=7)

        ax.set_xlim(0, row_seconds)
        ax.set_ylim(ylo, yhi)
        ax.tick_params(labelsize=7)
        ax.set_ylabel(f"t = {t0_abs:.0f}s", fontsize=7.5)
        if r == n_rows - 1:
            ax.set_xlabel("Seconds within row  —  small box 0.2 s, large box 1.0 s", fontsize=8)
        if r == 0 and np.isfinite(lo):
            ax.set_title(f"Filtered signal (post filter chain)  |  amplitude in RAW DEVICE UNITS, "
                         f"NOT mV  |  range {lo:.0f} to {hi:.0f} ({hi - lo:.0f} peak-to-peak)",
                         fontsize=8, color="#57606a", loc="left", pad=5)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf), n_rows


_BEAT_IDX_CACHE: dict[int, int] = {}


def _beat_index(beats, b) -> int:
    key = id(b)
    if key not in _BEAT_IDX_CACHE:
        for i, x in enumerate(beats):
            _BEAT_IDX_CACHE[id(x)] = i
    return _BEAT_IDX_CACHE.get(key, -1)


def render_thumbnail(filtered, valid, beats, labels, fs, bi,
                     width_in=2.25, height_in=1.15) -> ImageReader:
    """1-second window centred on one beat, same style as the main strip."""
    half = int(0.5 * fs)
    c = beats[bi].r_peak_idx
    i0, i1 = max(0, c - half), min(len(filtered), c + half)
    y = filtered[i0:i1].astype(float).copy()
    vs = valid[i0:i1]
    y[~vs] = np.nan
    t = np.arange(len(y)) / float(fs)

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=150)
    fin = y[np.isfinite(y)]
    if fin.size:
        lo, hi = float(fin.min()), float(fin.max())
        pad = max((hi - lo) * 0.12, 0.5)
        ylo, yhi = lo - pad, hi + pad
    else:
        ylo, yhi = -1.0, 1.0
    ax.set_xticks(np.arange(0, t[-1] + 1e-9, 0.2))
    ax.set_xticks(np.arange(0, t[-1] + 1e-9, 0.04), minor=True)
    ax.set_yticks(np.linspace(ylo, yhi, 4))
    ax.grid(which="major", color="#f2a0a8", linewidth=0.5, zorder=0)
    ax.grid(which="minor", color="#f8d3d7", linewidth=0.3, zorder=0)
    ax.plot(t, y, color="#111111", linewidth=0.9, zorder=3)
    _shade_gaps(ax, t, vs, ylo, yhi, label=False)
    lab = labels[bi]
    col = CLASS_COLORS.get(lab, "#57606a")
    ct = (c - i0) / float(fs)
    ax.axvline(ct, color=col, linewidth=1.1, alpha=0.85, zorder=4)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(ylo, yhi)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


# ==========================================================================
# PDF primitives
# ==========================================================================

def _txt(c, x, y, s, size=9, font=FONT, color=black):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, str(s))


def _wrap(c, x, y, s, width, size=8.5, leading=10.5, font=FONT, color=black):
    c.setFont(font, size)
    c.setFillColor(color)
    words, line = str(s).split(), ""
    for w in words:
        trial = (line + " " + w).strip()
        if c.stringWidth(trial, font, size) > width and line:
            c.drawString(x, y, line)
            y -= leading
            line = w
        else:
            line = trial
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def _page_header(c, title, patient, segment, page_no, total):
    c.setFillColor(HexColor("#24292f"))
    c.setFont(FONT_B, 11)
    c.drawString(MARGIN, PAGE_H - MARGIN, title)
    c.setFont(FONT, 7.5)
    c.setFillColor(HexColor("#57606a"))
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN,
                      f"{patient} / {segment}   page {page_no} of {total}")
    c.setStrokeColor(HexColor("#d0d7de"))
    c.setLineWidth(0.6)
    c.line(MARGIN, PAGE_H - MARGIN - 6, PAGE_W - MARGIN, PAGE_H - MARGIN - 6)
    return PAGE_H - MARGIN - 22


def _footer(c, text="RESEARCH USE ONLY — NOT APPROVED FOR CLINICAL USE — NOT A DIAGNOSIS"):
    c.setFont(FONT_B, 7)
    c.setFillColor(HexColor("#CC0000"))
    c.drawCentredString(PAGE_W / 2.0, 18, text)


def _table(c, x, y, rows, widths, size=8, header=True, leading=13, colors=None):
    for r, row in enumerate(rows):
        cx = x
        if header and r == 0:
            c.setFillColor(HexColor("#f6f8fa"))
            c.rect(x - 2, y - 3.5, sum(widths) + 4, leading, fill=1, stroke=0)
        for i, cell in enumerate(row):
            col = black
            if colors and r > 0 and colors.get((r, i)):
                col = HexColor(colors[(r, i)])
            c.setFont(FONT_B if (header and r == 0) else FONT, size)
            c.setFillColor(col)
            c.drawString(cx, y, str(cell))
            cx += widths[i]
        y -= leading
    return y


# ==========================================================================
# Pages
# ==========================================================================

def page1(c, ctx):
    rj, rec = ctx["report"], ctx["recording"]
    r = rj["recording"]
    y = _page_header(c, "ECG Validation Report", r["patient_id"], r["segment_id"], 1, ctx["n_pages"])

    dev = {"vitalpatch": "VitalPatch (single-lead)",
           "prorhythm": "ProRhythm / SeNSiO",
           "wfdb": "WFDB record"}.get(r["source"], r["source"])
    start_iso = "—"
    try:
        start_iso = datetime.utcfromtimestamp(int(rec.timestamps_ms[0]) / 1000).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        pass
    rows = [
        ["Patient", r["patient_id"], "Segment", r["segment_id"]],
        ["Recording start", start_iso, "Duration", f"{r['duration_s']} s"],
        ["Device", dev, "Sampling rate", f"{r['fs_nominal_hz']} Hz -> {r['fs_processed_hz']} Hz"],
        ["Pipeline", ctx["pipeline_version"], "Git commit", ctx["git_commit"]],
        ["Report generated", date.today().isoformat(), "Beats detected / analyzed",
         f"{r['n_beats_detected']} / {r['n_beats_analyzed']}"],
    ]
    y = _table(c, MARGIN, y, rows, [78, 165, 95, 150], size=8, header=False, leading=12.5)
    y -= 8

    # --- Signal quality box ---
    c.setFillColor(HexColor("#f6f8fa"))
    c.setStrokeColor(HexColor("#d0d7de"))
    c.rect(MARGIN, y - 74, PAGE_W - 2 * MARGIN, 74, fill=1, stroke=1)
    _txt(c, MARGIN + 8, y - 14, "SIGNAL QUALITY", 9, FONT_B)
    sq = ctx["sqi"]
    _txt(c, MARGIN + 8, y - 28,
         f"SQI score: {r['quality_score']:.2f}     "
         f"Rejection rate: {r['sqi_window_rejection_rate'] * 100:.1f}%     "
         f"Usable signal: {sq['usable_pct']:.1f}%", 8.5)
    def _yn(v):
        return "YES" if v else "NO"
    _txt(c, MARGIN + 8, y - 42,
         f"Baseline wander: {_yn(sq['baseline_wander'])}     "
         f"Flatline: {_yn(sq['flatline'])}     "
         f"Clipping: {_yn(sq['clipping'])}", 8.5)
    _txt(c, MARGIN + 8, y - 56,
         f"SQI windows: {sq['n_windows']} total, {sq['n_rejected']} rejected"
         + (f"   ({', '.join(f'{k} x{v}' for k, v in sq['codes'].items())})" if sq['codes'] else ""),
         7.5, color=HexColor("#57606a"))
    hr = r.get("heart_rate_bpm")
    _txt(c, MARGIN + 8, y - 68,
         f"Heart rate: {hr} bpm ({r.get('heart_rate_method', 'n/a')}, "
         f"{r.get('heart_rate_n_intervals', '?')} of {r.get('heart_rate_n_intervals_total', '?')} intervals)"
         if hr is not None else "Heart rate: not derivable from this segment", 7.5,
         color=HexColor("#57606a"))
    y -= 86

    # --- Verdict banner ---
    lvl = rj.get("final_risk_level", rj.get("risk_level", "UNKNOWN"))
    c.setFillColor(HexColor(VERDICT_COLORS.get(lvl, "#57606a")))
    c.rect(MARGIN, y - 40, PAGE_W - 2 * MARGIN, 40, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont(FONT_B, 20)
    c.drawCentredString(PAGE_W / 2.0, y - 27, lvl)
    y -= 52

    dr = rj.get("deciding_rule") or {}
    _txt(c, MARGIN, y, "Deciding rule", 9, FONT_B)
    y -= 12
    y = _wrap(c, MARGIN, y, _rule_sentence(dr), PAGE_W - 2 * MARGIN, size=9, leading=11)
    y -= 6

    if ctx["mismatch"]:
        c.setFillColor(HexColor("#fff8c5"))
        c.setStrokeColor(HexColor("#d4a72c"))
        c.rect(MARGIN, y - 52, PAGE_W - 2 * MARGIN, 52, fill=1, stroke=1)
        _txt(c, MARGIN + 8, y - 14, "NOTE — SAVED REPORT DIFFERS FROM CURRENT PIPELINE", 8.5,
             FONT_B, HexColor("#7d4e00"))
        _wrap(c, MARGIN + 8, y - 26, ctx["mismatch"], PAGE_W - 2 * MARGIN - 16,
              size=7.5, leading=9.5, color=HexColor("#7d4e00"))
        y -= 60

    _txt(c, MARGIN, y, "How to use this document", 9, FONT_B)
    y -= 12
    y = _wrap(c, MARGIN, y,
              "Page 2 shows the cleaned ECG with each detected beat marked and labelled. "
              "Page 3 lists what the classifier decided and how confident it was. Page 4 shows "
              "every rule the system evaluated. Page 5 is the form for your assessment — please "
              "complete it. Page 6 states what this system can and cannot do.",
              PAGE_W - 2 * MARGIN, size=8.5, leading=10.5)
    _footer(c)


def _rule_sentence(dr):
    if not dr:
        return "No deciding rule recorded."
    mv, th = dr.get("measured_value"), dr.get("threshold")
    if isinstance(mv, dict):
        mv = ", ".join(f"{k}={v}" for k, v in mv.items())
    if isinstance(th, dict):
        th = ", ".join(f"{k}={v}" for k, v in th.items())
    return f"{dr.get('condition', '?')}  —  measured {mv} vs threshold {th}  ->  {'FIRED' if dr.get('fired') else 'did not fire'}"


def page2(c, ctx):
    rj = ctx["report"]
    r = rj["recording"]
    y = _page_header(c, "Cleaned ECG strip", r["patient_id"], r["segment_id"], 2, ctx["n_pages"])
    i0, i1 = ctx["win"]
    fs = ctx["fs"]
    _txt(c, MARGIN, y,
         f"Showing {(i1 - i0) / fs:.0f} s from t = {i0 / fs:.1f}s to {i1 / fs:.1f}s "
         f"(the window with the most ectopic beats in this segment), in "
         f"{ctx['strip_rows']} rows of 10 s", 8.5)
    y -= 10
    img_w = PAGE_W - 2 * MARGIN
    img_h = img_w * (3.4 * ctx["strip_rows"]) / 10.2
    max_h = y - 132  # leave room for legend + beat sequence below
    if img_h > max_h:
        img_h = max_h
    c.drawImage(ctx["strip"], MARGIN, y - img_h, width=img_w, height=img_h)
    y -= img_h + 14

    # legend — two rows so the long class names never collide
    _txt(c, MARGIN, y, "Beat markers:", 8, FONT_B)
    y -= 12
    lx = MARGIN + 6
    for n, k in enumerate(["N", "V", "S", "F", "Q"]):
        if n == 3:
            y -= 11
            lx = MARGIN + 6
        c.setFillColor(HexColor(CLASS_COLORS[k]))
        if k == "Q":
            c.setFont(FONT_B, 9)
            c.drawString(lx, y, "x")
        else:
            c.circle(lx + 3, y + 2.6, 3.2, fill=1, stroke=0)
        c.setFillColor(black)
        c.setFont(FONT, 7.5)
        c.drawString(lx + 11, y, f"{k} — {CLASS_NAMES[k]}")
        lx += 172
    y -= 13
    c.setFillColor(HexColor("#57606a"))
    c.setFont(FONT, 7.5)
    c.drawString(MARGIN, y, "Grey hatched regions = signal rejected by the quality gate (NO DATA — "
                            "not recorded signal, and not asystole).")
    y -= 8
    c.drawString(MARGIN, y, "Small numbers above each marker are beat indices, matching the tables "
                            "on pages 3 and 5.")
    y -= 16

    _txt(c, MARGIN, y, "Beat sequence in this window", 9, FONT_B)
    y -= 12
    seq = [l for b, l in zip(ctx["beats"], ctx["labels"]) if i0 <= b.r_peak_idx < i1]
    y = _wrap(c, MARGIN, y, " ".join(seq) if seq else "(no beats detected in this window)",
              PAGE_W - 2 * MARGIN, size=8, leading=10, font="Courier")
    y -= 6
    c.setFillColor(HexColor("#57606a"))
    c.setFont(FONT, 7.5)
    c.drawString(MARGIN, y, f"Full segment: {len(ctx['labels'])} beats detected. "
                            f"Sequence above covers the displayed window only.")
    _footer(c)


def page3(c, ctx):
    rj = ctx["report"]
    r = rj["recording"]
    y = _page_header(c, "Beat classification", r["patient_id"], r["segment_id"], 3, ctx["n_pages"])

    bc = rj.get("beat_confidence") or {}
    per = bc.get("per_class") or {}
    bs = rj.get("beat_summary") or {}
    gt = ctx.get("ground_truth")

    hdr = ["Class", "Description", "Count", "% of detected", "Avg conf.", "Min conf."]
    widths = [34, 158, 42, 68, 52, 52]
    if gt:
        hdr += ["Correct", "Acc."]
        widths += [46, 40]
    rows = [hdr]
    colors = {}
    for i, k in enumerate(["N", "V", "S", "F", "Q"], start=1):
        info = bs.get(k, {})
        p = per.get(k, {})
        mean_c = p.get("mean_confidence")
        min_c = p.get("min_confidence")
        row = [k, CLASS_NAMES[k], info.get("count", 0),
               f"{beat_summary_pct(info):.1f}%" if info else "—",
               "N/A (rejected)" if k == "Q" else (f"{mean_c:.3f}" if mean_c is not None else "—"),
               "" if k == "Q" else (f"{min_c:.3f}" if min_c is not None else "—")]
        if gt:
            tot = sum(1 for j, l in enumerate(ctx["labels"]) if l == k and j in gt)
            ok = sum(1 for j, l in enumerate(ctx["labels"]) if l == k and gt.get(j) == l)
            row += [f"{ok}/{tot}" if tot else "—", f"{100*ok/tot:.0f}%" if tot else "—"]
        colors[(i, 0)] = CLASS_COLORS[k]
        if CLASS_F1.get(k) is not None and CLASS_F1[k] < 0.3:
            colors[(i, 4)] = "#CC0000"
        rows.append(row)
    y = _table(c, MARGIN, y, rows, widths, size=8)
    y -= 6
    c.setFillColor(HexColor("#CC0000"))
    c.setFont(FONT, 7.5)
    c.drawString(MARGIN, y, "S class F1 = 0.152 and F class F1 = 0.005 on the held-out validation set. "
                            "Treat S and F counts as a screening signal only — not reliable.")
    y -= 10
    c.setFillColor(HexColor("#57606a"))
    c.drawString(MARGIN, y, "Percentages are of ALL detected beats (including quality-rejected Q beats), "
                            "matching the burden figures used by the rule engine.")
    y -= 16

    lows = bc.get("lowest_confidence_beats") or []
    _txt(c, MARGIN, y, f"Lowest-confidence beats (n={len(lows)})", 9, FONT_B)
    y -= 14
    hdr2 = ["#", "Beat", "Time (s)", "Label", "Confidence", "Prev", "Next", "Runner-up"]
    w2 = [16, 34, 50, 34, 58, 32, 32, 100]
    if gt:
        hdr2 += ["Clinician"]
        w2 += [56]
    rows2 = [hdr2]
    colors2 = {}
    runner = ctx.get("runner_up") or {}
    for i, b in enumerate(lows, start=1):
        ru = runner.get(b["beat_index"], "—")
        row = [i, b["beat_index"], f"{b['timestamp_s']:.2f}", b["predicted_class"],
               f"{b['confidence']:.4f}",
               (b.get("neighbors") or {}).get("prev") or "—",
               (b.get("neighbors") or {}).get("next") or "—", ru]
        if gt:
            row += [gt.get(b["beat_index"], "—")]
        colors2[(i, 3)] = CLASS_COLORS.get(b["predicted_class"], "#57606a")
        rows2.append(row)
    y = _table(c, MARGIN, y, rows2, w2, size=7.5, leading=11, colors=colors2)
    y -= 10

    _txt(c, MARGIN, y, "Why the model chose these labels (SHAP attributions)", 9, FONT_B)
    y -= 12
    c.setFillColor(HexColor("#57606a"))
    c.setFont(FONT, 7.5)
    c.drawString(MARGIN, y, "Exact TreeSHAP from the frozen model, in margin (log-odds) space. "
                            "These explain the MODEL's decision — not that the decision is correct.")
    y -= 13
    for b in lows[:5]:
        sh = b.get("shap") or {}
        feats = sh.get("top_features") or []
        if not feats:
            continue
        _txt(c, MARGIN, y, f"Beat {b['beat_index']} at {b['timestamp_s']:.2f}s — "
                           f"predicted {b['predicted_class']} (conf {b['confidence']:.3f})",
             8, FONT_B, HexColor(CLASS_COLORS.get(b["predicted_class"], "#24292f")))
        y -= 10
        for f in feats[:3]:
            nm = f["feature"]
            mean = f.get("meaning") or "wavelet shape coefficient"
            _txt(c, MARGIN + 12, y,
                 f"{nm:<22} SHAP {f['shap']:+.3f}  ({f['direction']} {b['predicted_class']})   "
                 f"value {f['value']:.3f}   — {mean}", 7.2)
            y -= 9
        y -= 4
        if y < 70:
            break
    _footer(c)


def page4(c, ctx):
    rj = ctx["report"]
    r = rj["recording"]
    y = _page_header(c, "Rule trace — every rule the system evaluated",
                     r["patient_id"], r["segment_id"], 4, ctx["n_pages"])

    rows = [["Rule", "Threshold", "Provenance", "Measured", "Fired", "Sets level"]]
    colors = {}
    notes = []
    for i, tr in enumerate(rj.get("rule_trace", []), start=1):
        cond = tr.get("condition", "")
        prov, note = PROVENANCE.get(cond, ("—", ""))
        mv, th = tr.get("measured_value"), tr.get("threshold")
        if isinstance(mv, dict):
            mv = " / ".join(f"{v}" for v in mv.values())
        if isinstance(th, dict):
            th = " / ".join(f"{v}" for v in th.values())
        if not tr.get("evaluated", True):
            mv = "not evaluated"
        fired = "YES" if tr.get("fired") else "no"
        rows.append([_short(cond, 46), str(th), prov, str(mv), fired,
                     tr.get("would_set_level", "")])
        if tr.get("fired"):
            colors[(i, 4)] = "#CC0000"
        if tr.get("is_deciding_rule"):
            colors[(i, 0)] = "#CC0000"
        if note and note not in notes:
            notes.append(note)
    y = _table(c, MARGIN, y, rows, [186, 52, 104, 74, 30, 66], size=7.2, leading=11.5, colors=colors)
    y -= 8
    c.setFillColor(HexColor("#57606a"))
    c.setFont(FONT, 7)
    c.drawString(MARGIN, y, "The deciding rule is shown in red. Rules are evaluated in severity order, "
                            "first match wins.")
    y -= 12
    _txt(c, MARGIN, y, "Threshold provenance notes", 8.5, FONT_B)
    y -= 11
    for n in notes:
        y = _wrap(c, MARGIN + 8, y, "• " + n, PAGE_W - 2 * MARGIN - 16, size=7, leading=8.8)
        y -= 1
    y -= 8

    # --- vitals ---
    _txt(c, MARGIN, y, "Vitals used in this assessment", 9, FONT_B)
    y -= 13
    v = rj.get("vitals") or {}
    if v.get("status") != "matched":
        reason = v.get("note") or "No vitals were supplied for this run."
        y = _wrap(c, MARGIN, y, reason, PAGE_W - 2 * MARGIN, size=8, leading=10)
        y -= 4
    else:
        comp = (v.get("news2") or {}).get("components") or {}
        vr = [["Vital", "Value", "NEWS2 score", "Source"]]
        vcolors = {}
        order = [("heart_rate", "Heart rate (bpm)"), ("respiratory_rate", "Respiratory rate (/min)"),
                 ("temperature", "Temperature (C)"), ("spo2", "SpO2 (%)"),
                 ("systolic_bp", "Systolic BP (mmHg)"), ("consciousness", "Consciousness (AVPU)")]
        for j, (k, lbl) in enumerate(order, start=1):
            cc = comp.get(k) or {}
            avail = cc.get("available")
            vr.append([lbl,
                       cc.get("value") if avail else "NOT AVAILABLE",
                       cc.get("score") if avail else "—",
                       cc.get("source") if avail else (cc.get("reason") or "not measured")])
            if not avail:
                vcolors[(j, 1)] = "#CC0000"
        y = _table(c, MARGIN, y, vr, [126, 92, 66, 230], size=7.4, leading=11, colors=vcolors)
        y -= 6
        n2 = v.get("news2") or {}
        qs = v.get("qsofa") or {}
        _txt(c, MARGIN, y,
             f"NEWS2: {n2.get('total_score')}  "
             f"({n2.get('components_available')} of {n2.get('components_total')} components)      "
             f"qSOFA: {qs.get('total_score')}  "
             f"({qs.get('components_available')} of {qs.get('components_total')} components)",
             8.5, FONT_B)
        y -= 11
        y = _wrap(c, MARGIN, y,
                  "Unavailable components contribute 0 and are listed, never imputed — both scores are "
                  "LOWER BOUNDS, not complete scores. Time gap to nearest vitals: "
                  f"{v.get('time_offset_minutes')} min (match: {v.get('match_method')}).",
                  PAGE_W - 2 * MARGIN, size=7.2, leading=9, color=HexColor("#57606a"))
    _footer(c)


def _short(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


def page5(c, ctx):
    rj = ctx["report"]
    r = rj["recording"]
    y = _page_header(c, "Clinician review form", r["patient_id"], r["segment_id"], 5, ctx["n_pages"])
    form = c.acroForm
    lvl = rj.get("final_risk_level", "—")

    y = _wrap(c, MARGIN, y,
              "Please review the ECG strip on page 2 and answer the questions below. Your answers "
              "will be used to validate the automated analysis. This form is fillable digitally in "
              "Acrobat Reader, Firefox, Edge or Preview, or may be completed by hand.",
              PAGE_W - 2 * MARGIN, size=8.5, leading=10.5)
    y -= 8

    _txt(c, MARGIN, y, "SECTION A — Overall assessment", 9.5, FONT_B)
    y -= 16

    def q(qname, text, options, y):
        y = _wrap(c, MARGIN, y, text, PAGE_W - 2 * MARGIN, size=8.5, leading=10.5, font=FONT_B)
        y -= 2
        for val, lbl in options:
            form.checkbox(name=f"{qname}_{val}", x=MARGIN + 10, y=y - 9, size=10,
                          buttonStyle="check", borderWidth=0.7,
                          borderColor=HexColor("#57606a"), fillColor=white)
            _txt(c, MARGIN + 26, y - 6.5, lbl, 8.2)
            y -= 13.5
        return y - 4

    y = q("q1", "Q1. Does the ECG strip show the arrhythmia described?", [
        ("yes", "Yes, clearly visible"),
        ("partial", "Partially — some findings present"),
        ("no", "No — I do not see this arrhythmia"),
        ("cannot", "Cannot assess — signal quality too poor")], y)

    y = q("q2", f"Q2. Is the {lvl} verdict clinically justified?", [
        ("yes", "Yes — I would act on this alert"),
        ("probably", "Probably — warrants monitoring"),
        ("no", "No — this appears to be a false alarm"),
        ("uncertain", "Uncertain — need more information")], y)

    pvc_thr = RISK.pvc_burden_critical_pct
    y = _wrap(c, MARGIN, y,
              f"Q3. Is the {pvc_thr:.0f}% PVC burden threshold appropriate for triggering a "
              f"CRITICAL alert?", PAGE_W - 2 * MARGIN, size=8.5, leading=10.5, font=FONT_B)
    y -= 2
    for val, lbl in [("ok", "Yes — appropriate threshold"),
                     ("higher", "Too sensitive — should be higher, suggest:"),
                     ("lower", "Not sensitive enough — should be lower, suggest:")]:
        form.checkbox(name=f"q3_{val}", x=MARGIN + 10, y=y - 9, size=10, buttonStyle="check",
                      borderWidth=0.7, borderColor=HexColor("#57606a"), fillColor=white)
        _txt(c, MARGIN + 26, y - 6.5, lbl, 8.2)
        if val != "ok":
            form.textfield(name=f"q3_{val}_pct", x=MARGIN + 250, y=y - 10, width=46, height=12,
                           borderWidth=0.7, borderColor=HexColor("#57606a"), fontSize=8)
            _txt(c, MARGIN + 300, y - 6.5, "%", 8.2)
        y -= 14
    y -= 8

    _txt(c, MARGIN, y, "SECTION B — Beat-by-beat review", 9.5, FONT_B)
    y -= 11
    c.setFillColor(HexColor("#57606a"))
    c.setFont(FONT, 7.5)
    c.drawString(MARGIN, y, "These are the beats the model was least sure about — where your "
                            "judgement is worth the most. Each strip is 1 second centred on the beat.")
    y -= 12

    thumbs = ctx["thumbs"]
    tw, th = 150, 62
    col_x = [MARGIN, MARGIN + 178, MARGIN + 356]
    i = 0
    for entry in thumbs:
        cx = col_x[i % 3]
        if i % 3 == 0 and i > 0:
            y -= (th + 46)
        if y - th - 46 < 60:
            break
        c.drawImage(entry["img"], cx, y - th, width=tw, height=th, preserveAspectRatio=False)
        _txt(c, cx, y - th - 10,
             f"Beat {entry['beat_index']}  t={entry['t']:.2f}s", 7, FONT_B)
        _txt(c, cx, y - th - 19,
             f"Pipeline: {entry['label']} (conf {entry['conf']:.3f})", 7,
             color=HexColor(CLASS_COLORS.get(entry["label"], "#57606a")))
        bx = cx
        for code in ["N", "S", "V", "Q", "?"]:
            form.checkbox(name=f"beat{entry['beat_index']}_{code}", x=bx, y=y - th - 33,
                          size=8, buttonStyle="check", borderWidth=0.6,
                          borderColor=HexColor("#57606a"), fillColor=white)
            _txt(c, bx + 10, y - th - 31.5, code, 6.8)
            bx += 26
        _txt(c, cx, y - th - 42, "agree / correct label / unsure(?)", 6, color=HexColor("#8c959f"))
        i += 1
    y -= (th + 52)

    if y < 150:
        c.showPage()
        y = _page_header(c, "Clinician review form (continued)", r["patient_id"],
                         r["segment_id"], 5, ctx["n_pages"])

    _txt(c, MARGIN, y, "SECTION C — Free text", 9.5, FONT_B)
    y -= 14
    for nm, lbl in [("missing", "What is missing from this report that you would need to make a "
                                "clinical decision?"),
                    ("changes", "What would need to change before you would use this system "
                                "clinically?")]:
        y = _wrap(c, MARGIN, y, lbl, PAGE_W - 2 * MARGIN, size=8.2, leading=10)
        form.textfield(name=nm, x=MARGIN, y=y - 40, width=PAGE_W - 2 * MARGIN, height=38,
                       borderWidth=0.7, borderColor=HexColor("#57606a"), fontSize=8,
                       fieldFlags="multiline")
        y -= 50

    y -= 4
    for nm, lbl, w in [("reviewer_name", "Reviewer", 170), ("reviewer_specialty", "Specialty", 150),
                       ("review_date", "Date", 96)]:
        _txt(c, MARGIN if nm == "reviewer_name" else (MARGIN + 210 if nm == "reviewer_specialty"
                                                      else MARGIN + 400), y, lbl, 8, FONT_B)
    y -= 14
    form.textfield(name="reviewer_name", x=MARGIN, y=y, width=170, height=14,
                   borderWidth=0.7, borderColor=HexColor("#57606a"), fontSize=8)
    form.textfield(name="reviewer_specialty", x=MARGIN + 210, y=y, width=150, height=14,
                   borderWidth=0.7, borderColor=HexColor("#57606a"), fontSize=8)
    form.textfield(name="review_date", x=MARGIN + 400, y=y, width=96, height=14,
                   borderWidth=0.7, borderColor=HexColor("#57606a"), fontSize=8)
    y -= 24
    _txt(c, MARGIN, y, "Signature", 8, FONT_B)
    c.setStrokeColor(HexColor("#57606a"))
    c.line(MARGIN + 52, y - 1, MARGIN + 300, y - 1)
    _footer(c)


def page6(c, ctx):
    rj = ctx["report"]
    r = rj["recording"]
    y = _page_header(c, "Limitations and disclaimer", r["patient_id"], r["segment_id"],
                     6, ctx["n_pages"])

    _txt(c, MARGIN, y, "What this system CAN do", 10, FONT_B, HexColor("#006400"))
    y -= 14
    for s in [
        "Detect R-peaks. Validated against the device's own firmware-reported RR intervals: "
        "93.3% of segments agree within 20 ms; mean heart-rate difference 0.68 bpm; "
        "n = 2,651 segments against 4,364,472 reference intervals.",
        "Flag elevated PVC burden above a configurable threshold.",
        "Identify rhythm patterns — bigeminy, trigeminy, VT runs, RR-irregularity suggestive of AF.",
        "Provide a complete, explainable rule trace for every verdict it produces.",
        "Compute a partial early-warning score from the vitals the device does measure.",
    ]:
        y = _wrap(c, MARGIN + 10, y, "• " + s, PAGE_W - 2 * MARGIN - 20, size=8, leading=9.8)
        y -= 2
    y -= 8

    _txt(c, MARGIN, y, "What this system CANNOT do", 10, FONT_B, HexColor("#CC0000"))
    y -= 14
    for s in [
        "Diagnose arrhythmias. This is decision support only.",
        "Reliably classify supraventricular (S / PAC) beats — held-out F1 = 0.152.",
        "Classify fusion (F) beats — held-out F1 = 0.005, essentially unsolved.",
        "Compute a complete NEWS2. SpO2 and blood pressure are not measured by this device, "
        "so the score is a lower bound computed from a subset of components.",
        "Compute a true qSOFA. Two of its three criteria (systolic BP, altered mentation) have no "
        "input on this device; the reported figure is a 1-of-3 proxy that can never reach the "
        "published threshold of 2.",
        "Establish accuracy on this device. The classifier was trained and validated on MIT-BIH "
        "(2-lead, different population). No clinician-annotated VitalPatch labels exist — which is "
        "precisely what this review is for.",
        "Replace clinical judgement.",
    ]:
        y = _wrap(c, MARGIN + 10, y, "• " + s, PAGE_W - 2 * MARGIN - 20, size=8, leading=9.8)
        y -= 2
    y -= 8

    _txt(c, MARGIN, y, "Known limitations of THIS segment", 10, FONT_B)
    y -= 14
    lims = list(rj.get("known_limitations") or [])
    sq = ctx["sqi"]
    if sq["rejection_rate"] > 0.25:
        lims.append(f"{sq['rejection_rate'] * 100:.0f}% of this recording was discarded by the "
                    f"quality gate, leaving {sq['usable_pct']:.0f}% usable. Detected beat counts "
                    f"and any rate derived from them cover only the surviving signal.")
    bc = rj.get("beat_confidence") or {}
    if bc.get("n_below_0_70"):
        lims.append(f"{bc['n_below_0_70']} of {bc.get('n_classified')} classified beats were below "
                    f"0.70 confidence.")
    if not lims:
        lims = ["No segment-specific limitations were recorded."]
    for s in lims:
        y = _wrap(c, MARGIN + 10, y, "• " + s, PAGE_W - 2 * MARGIN - 20, size=8, leading=9.8)
        y -= 2
    y -= 12

    box_h = 62
    c.setFillColor(HexColor("#fff0f0"))
    c.setStrokeColor(HexColor("#CC0000"))
    c.setLineWidth(1.4)
    c.rect(MARGIN, y - box_h, PAGE_W - 2 * MARGIN, box_h, fill=1, stroke=1)
    _txt(c, MARGIN + 10, y - 16, "DISCLAIMER", 10, FONT_B, HexColor("#CC0000"))
    _wrap(c, MARGIN + 10, y - 30,
          "This output is generated by a research pipeline undergoing clinical validation. It is "
          "NOT approved for clinical use. It is NOT a diagnosis. Do not make clinical decisions "
          "based solely on this output.",
          PAGE_W - 2 * MARGIN - 20, size=8.5, leading=10.5, font=FONT_B, color=HexColor("#CC0000"))
    _footer(c)


def page_confusion(c, ctx):
    """Only rendered when --ground-truth is supplied."""
    rj = ctx["report"]
    r = rj["recording"]
    gt = ctx["ground_truth"]
    y = _page_header(c, "Ground-truth comparison", r["patient_id"], r["segment_id"],
                     ctx["n_pages"], ctx["n_pages"])
    labels = ctx["labels"]
    classes = ["N", "S", "V", "F", "Q"]
    pairs = [(labels[i], gt[i]) for i in sorted(gt) if i < len(labels)]
    if not pairs:
        _txt(c, MARGIN, y, "No ground-truth rows matched a detected beat index.", 9)
        _footer(c)
        return
    _txt(c, MARGIN, y, f"{len(pairs)} reviewed beats matched to pipeline predictions", 9, FONT_B)
    y -= 18

    mat = {(p, t): 0 for p in classes for t in classes}
    for p, t in pairs:
        if (p, t) in mat:
            mat[(p, t)] += 1
    rows = [["Predicted \\ True"] + classes + ["Total", "Precision"]]
    colors = {}
    for i, p in enumerate(classes, start=1):
        tot = sum(mat[(p, t)] for t in classes)
        corr = mat[(p, p)]
        rows.append([p] + [mat[(p, t)] for t in classes] + [tot,
                     f"{100 * corr / tot:.0f}%" if tot else "—"])
        colors[(i, 0)] = CLASS_COLORS[p]
    tots = ["Total"] + [sum(mat[(p, t)] for p in classes) for t in classes] + [len(pairs), ""]
    rows.append(tots)
    rec_row = ["Recall"]
    for t in classes:
        tt = sum(mat[(p, t)] for p in classes)
        rec_row.append(f"{100 * mat[(t, t)] / tt:.0f}%" if tt else "—")
    rec_row += ["", ""]
    rows.append(rec_row)
    y = _table(c, MARGIN, y, rows, [92, 46, 46, 46, 46, 46, 50, 60], size=8, leading=13, colors=colors)
    y -= 14

    acc = sum(1 for p, t in pairs if p == t) / len(pairs)
    _txt(c, MARGIN, y, f"Overall agreement: {acc * 100:.1f}%  ({sum(1 for p, t in pairs if p == t)} "
                       f"of {len(pairs)} beats)", 9.5, FONT_B)
    y -= 18
    _txt(c, MARGIN, y, "Disagreements", 9, FONT_B)
    y -= 13
    dis = [(i, labels[i], gt[i]) for i in sorted(gt) if i < len(labels) and labels[i] != gt[i]]
    if not dis:
        _txt(c, MARGIN + 8, y, "None — pipeline agreed with the clinician on every reviewed beat.", 8)
    else:
        rows3 = [["Beat", "Time (s)", "Pipeline", "Clinician"]]
        for i, p, t in dis[:34]:
            ts = ctx["beats"][i].r_peak_ms / 1000.0 if i < len(ctx["beats"]) else float("nan")
            rows3.append([i, f"{ts:.2f}", p, t])
        y = _table(c, MARGIN, y, rows3, [46, 60, 60, 60], size=8, leading=11.5)
    _footer(c)


# ==========================================================================
# Orchestration
# ==========================================================================

def _git_commit():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _sqi_facts(recording):
    keep, verdicts = run_sqi_gate(recording.signal_mv, recording.timestamps_ms,
                                  recording.fs_nominal, clip_value=None)
    codes = Counter(v.reject_code for v in verdicts if not v.passed and v.reject_code)
    return {
        "n_windows": len(verdicts),
        "n_rejected": sum(1 for v in verdicts if not v.passed),
        "rejection_rate": 1.0 - (sum(1 for v in verdicts if v.passed) / max(1, len(verdicts))),
        "usable_pct": 100.0 * float(keep.mean()) if len(keep) else 0.0,
        "codes": dict(codes),
        "baseline_wander": "BASELINE_WANDER" in codes,
        "flatline": "FLATLINE" in codes,
        "clipping": "CLIPPING" in codes,
    }


def load_ground_truth(path: Path, beats) -> dict[int, str]:
    """beat_index -> true_label. Falls back to nearest-timestamp matching (within
    100 ms) when beat_index is absent, so a reviewer who recorded only times is
    not silently dropped."""
    gt: dict[int, str] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            lbl = (row.get("true_label") or "").strip().upper()
            if not lbl:
                continue
            bi = row.get("beat_index")
            if bi not in (None, ""):
                try:
                    gt[int(bi)] = lbl
                    continue
                except ValueError:
                    pass
            ts = row.get("timestamp_ms")
            if ts:
                try:
                    tms = float(ts)
                except ValueError:
                    continue
                best, bd = None, 1e18
                for i, b in enumerate(beats):
                    d = abs(b.r_peak_ms - tms)
                    if d < bd:
                        best, bd = i, d
                if best is not None and bd <= 100.0:
                    gt[best] = lbl
    return gt


def build(report_path: Path, ecg_csv: Path, out_dir: Path,
          vitals_root: Path | None = None, ground_truth: Path | None = None,
          classifier_path: Path | None = None) -> Path:
    saved = json.loads(report_path.read_text()) if report_path else None
    segment_id = (saved or {}).get("recording", {}).get("segment_id")
    recording = _parse_ecg(ecg_csv, segment_id)

    clf = load_classifier(classifier_path or (MODELS_DIR / "five_class_xgb.json"))
    fresh, result = run_full_report(recording, clf,
                                    vitals_root=str(vitals_root) if vitals_root else None)

    # Which report do we render? The fresh one is internally consistent with the
    # waveform we are about to draw, so it is the source of truth; the saved one
    # is used only to detect and DISCLOSE drift.
    rj = fresh
    mismatch = ""
    if saved:
        s_lvl = saved.get("final_risk_level")
        f_lvl = fresh.get("final_risk_level")
        s_rule = (saved.get("deciding_rule") or {}).get("condition")
        f_rule = (fresh.get("deciding_rule") or {}).get("condition")
        if s_lvl != f_lvl or s_rule != f_rule:
            mismatch = (
                f"The saved report on disk says {s_lvl} (deciding rule: {s_rule}). Re-running the "
                f"current pipeline on the same file gives {f_lvl} (deciding rule: {f_rule}). This "
                f"document shows the CURRENT result, which is what the ECG strip below depicts. "
                f"A difference is expected for reports saved before a later pipeline fix.")

    filtered, valid, beats, fs = rebuild_waveform(recording)
    labels = list(result.beat_labels)
    confs = list(result.beat_confidences or [])
    if len(beats) != len(labels):
        # Detection is deterministic, so this should not happen; if it does, trust
        # the pipeline's own beats rather than silently mis-aligning markers.
        beats = result.beats
    _BEAT_IDX_CACHE.clear()

    sqi = _sqi_facts(recording)
    i0, i1 = pick_window(beats, labels, fs, len(filtered))
    strip, strip_rows = render_main_strip(filtered, valid, beats, labels, fs, i0, i1)

    lows = (rj.get("beat_confidence") or {}).get("lowest_confidence_beats") or []

    # Runner-up class per low-confidence beat. Recomputed here from the stored
    # feature vector rather than added to the report schema -- the report keeps
    # only the winning probability, and a near-tie is exactly what a reviewer
    # needs to see (beat 52 below picks V over N by 0.0023).
    runner_up: dict[int, str] = {}
    fvs = result.beat_feature_vectors or []
    mean_rr = float(np.mean([b.rr_post_ms for b in result.beats
                             if b.rr_post_ms is not None])) if result.beats else 0.0
    for b in lows:
        bi = b["beat_index"]
        if bi >= len(fvs) or fvs[bi] is None:
            continue
        try:
            pr = clf.predict_one(fvs[bi], result.beats[bi], mean_rr).probabilities or {}
            ordered = sorted(pr.items(), key=lambda kv: -kv[1])
            if len(ordered) > 1:
                runner_up[bi] = f"{ordered[1][0]} {ordered[1][1]:.3f}"
        except Exception:
            pass
    thumbs = []
    for b in lows[:10]:
        bi = b["beat_index"]
        if bi >= len(beats):
            continue
        thumbs.append({"beat_index": bi, "t": b["timestamp_s"], "label": b["predicted_class"],
                       "conf": b["confidence"],
                       "img": render_thumbnail(filtered, valid, beats, labels, fs, bi)})

    gt = load_ground_truth(ground_truth, beats) if ground_truth else None

    ctx = {
        "report": rj, "recording": recording, "beats": beats, "labels": labels,
        "confs": confs, "fs": fs, "win": (i0, i1), "strip": strip,
        "strip_rows": strip_rows, "thumbs": thumbs, "runner_up": runner_up,
        "sqi": sqi, "mismatch": mismatch, "ground_truth": gt,
        "pipeline_version": "ecg_pipeline (agent_bridge.run_full_report)",
        "git_commit": _git_commit(),
        "n_pages": 7 if gt else 6,
    }

    r = rj["recording"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{r['patient_id']}_{r['segment_id']}_validation_report.pdf"
    c = pdfcanvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle(f"ECG Validation Report — {r['patient_id']} / {r['segment_id']}")
    for fn in (page1, page2, page3, page4, page5, page6):
        fn(c, ctx)
        c.showPage()
    if gt:
        page_confusion(c, ctx)
        c.showPage()
    c.save()
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", type=Path, required=True,
                    help="Pipeline JSON report for the segment (format written by save_report()).")
    ap.add_argument("--ecg-csv", type=Path, required=True,
                    help="Raw ECG CSV the report was produced from.")
    ap.add_argument("--output", type=Path, required=True, help="Output directory.")
    ap.add_argument("--vitals-root", type=Path, default=None,
                    help="Vitals tree (e.g. data/vitals_downloads) to evaluate NEWS2/qSOFA.")
    ap.add_argument("--ground-truth", type=Path, default=None,
                    help="CSV: beat_index,timestamp_ms,true_label. Adds comparison + confusion matrix.")
    ap.add_argument("--classifier", type=Path, default=None)
    a = ap.parse_args(argv)

    for p in [a.report, a.ecg_csv] + ([a.ground_truth] if a.ground_truth else []):
        if not p.exists():
            raise SystemExit(f"not found: {p}")

    out = build(a.report, a.ecg_csv, a.output, a.vitals_root, a.ground_truth, a.classifier)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
