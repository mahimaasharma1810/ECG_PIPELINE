#!/usr/bin/env python3
"""Generate a self-contained, fillable clinician review PDF from pipeline
beat predictions.

    python annotation/generate_annotation_pdf.py \
        --input annotation/sample_beats.json \
        --output annotation/review_sample.pdf

One beat per page: a 3-second ECG strip with an ECG-style grid, the
pipeline's prediction and rhythm context, and four fillable multiple-choice
questions plus a free-text comment. A final summary page lists every beat
in the document and carries the reviewer name/date fields.

The clinician needs nothing but a PDF reader that supports forms (Acrobat
Reader, Firefox, Edge, Preview). Answers are stored as AcroForm fields and
are read back out by parse_annotation_pdf.py.

Amplitude is plotted and labelled in RAW DEVICE UNITS. VitalPatch exposes
no published mV-per-count constant, so the vertical axis is deliberately
NOT calibrated to millivolts and the standard 0.1 mV/0.5 mV ECG amplitude
boxes are deliberately NOT drawn -- doing either would assert a
calibration this device does not provide. The time axis IS standard
(0.04 s small box, 0.2 s large box) and is drawn as such.
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas

PAGE_W, PAGE_H = A4
MARGIN = 38

CLASS_NAMES = {
    "N": "Normal sinus beat",
    "S": "Supraventricular (PAC, junctional)",
    "V": "Ventricular (PVC, ectopic)",
    "F": "Fusion beat",
    "Q": "Artifact / uninterpretable",
}
CLASS_COLORS = {
    "N": "#1a7f37", "S": "#9a6700", "V": "#cf222e",
    "F": "#8250df", "Q": "#57606a",
}

Q1_OPTIONS = [("N", "N — Normal sinus beat"),
              ("S", "S — Supraventricular (PAC, junctional)"),
              ("V", "V — Ventricular (PVC, ectopic)"),
              ("F", "F — Fusion beat"),
              ("Q", "Q — Artifact / uninterpretable"),
              ("Unsure", "Unsure — needs more context")]
Q2_OPTIONS = [("clean", "Clean — usable for analysis"),
              ("noisy", "Noisy — interpret with caution"),
              ("artifact", "Artifact — discard this beat")]
Q3_OPTIONS = [("yes", "Yes — pipeline is correct"),
              ("no", "No — pipeline is wrong (my label above is correct)"),
              ("borderline", "Borderline — could be either")]
Q4_OPTIONS = [("isolated", "Isolated — not concerning"),
              ("run", "Part of a run — concerning"),
              ("unsure", "Unsure")]


# --------------------------------------------------------------------------
# ECG strip rendering
# --------------------------------------------------------------------------
def render_strip(entry: dict, width_in=7.4, height_in=2.7) -> ImageReader:
    fs = entry.get("fs", 125)
    raw = entry.get("ecg_samples") or []
    y = np.array([np.nan if v is None else float(v) for v in raw], dtype=float)
    if y.size == 0:
        y = np.full(int(3 * fs), np.nan)
    t = np.arange(y.size) / float(fs)

    r_idx = entry.get("r_peak_in_strip")
    if r_idx is None:
        r_idx = int(round(1.0 * fs))  # exporter centres on 1 s pre by default
    r_t = r_idx / float(fs)

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=150)

    # Axis limits track the real signal range with only a hairline margin.
    # An earlier 18% pad made the y-tick labels read far wider than the
    # actual data (e.g. a -39 unit trough labelled -51.73), which invited
    # reviewers to judge amplitude off the axis rather than the waveform.
    finite = y[np.isfinite(y)]
    if finite.size:
        lo, hi = float(np.min(finite)), float(np.max(finite))
        span = hi - lo
        pad = max(span * 0.04, 0.5)
        ylo, yhi = lo - pad, hi + pad
    else:
        lo = hi = float("nan")
        ylo, yhi = -1.0, 1.0

    # Time grid only: 0.04 s minor / 0.2 s major (true ECG paper timing).
    # Amplitude gridlines are drawn as neutral guides at even divisions --
    # NOT as calibrated 0.1/0.5 mV boxes (see module docstring).
    ax.set_xticks(np.arange(0, t[-1] + 1e-9, 0.2))
    ax.set_xticks(np.arange(0, t[-1] + 1e-9, 0.04), minor=True)
    yticks = np.linspace(ylo, yhi, 6)
    ax.set_yticks(yticks)
    ax.set_yticks(np.linspace(ylo, yhi, 26), minor=True)
    ax.grid(which="major", color="#f2a0a8", linewidth=0.7, zorder=0)
    ax.grid(which="minor", color="#f8d3d7", linewidth=0.4, zorder=0)

    ax.plot(t, y, color="#111111", linewidth=1.05, zorder=3, solid_joinstyle="round")

    label = entry.get("predicted_label", "?")
    colr = CLASS_COLORS.get(label, "#57606a")
    ax.axvline(r_t, color=colr, linewidth=1.3, alpha=0.85, zorder=4)
    if np.isfinite(y[min(r_idx, y.size - 1)]):
        ax.plot([r_t], [y[min(r_idx, y.size - 1)]], marker="o", markersize=8,
                markerfacecolor=colr, markeredgecolor="white", markeredgewidth=1.4, zorder=6)
    ax.annotate(f"reviewed beat ({label})", xy=(r_t, yhi), xytext=(0, -11),
                textcoords="offset points", ha="center", va="top",
                fontsize=8.5, color=colr, fontweight="bold", zorder=7)

    # Blanked/removed samples are shaded and explicitly captioned "NO DATA".
    # These stretches are NOT recorded signal: the SQI gate rejects whole
    # ~5 s windows (e.g. BASELINE_WANDER) and the pipeline then fills them
    # by linear interpolation. Drawing that filler as a trace makes deleted
    # data indistinguishable from true asystole on a clinician-facing form,
    # so it is never plotted as a line here.
    nan_mask = np.isnan(y)
    if nan_mask.any():
        edges = np.diff(nan_mask.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if nan_mask[0]:
            starts = [0] + starts
        if nan_mask[-1]:
            ends = ends + [y.size - 1]
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[e], facecolor="#8c959f", alpha=0.30, zorder=5,
                       hatch="///", edgecolor="#57606a", linewidth=0.0)
            if (t[e] - t[s]) > 0.12:  # only caption gaps wide enough to read
                ax.text(0.5 * (t[s] + t[e]), 0.5 * (ylo + yhi), "NO DATA",
                        ha="center", va="center", fontsize=8.5, rotation=90,
                        color="#24292f", fontweight="bold", zorder=8)

    ax.set_xlim(0, t[-1])
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Time (s)  —  small box 0.04 s, large box 0.2 s", fontsize=8.5)
    ax.set_ylabel("Amplitude (raw device units)", fontsize=8.5)
    if np.isfinite(lo) and np.isfinite(hi):
        ax.set_title(f"preprocessed signal  |  actual range {lo:.1f} to {hi:.1f} "
                     f"({hi - lo:.1f} units peak-to-peak)",
                     fontsize=8, color="#57606a", loc="left", pad=4)
    ax.tick_params(labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#8c8c8c")
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


# --------------------------------------------------------------------------
# PDF page assembly
# --------------------------------------------------------------------------
def _fmt_conf(entry) -> str:
    c = entry.get("confidence")
    if c is None:
        if entry.get("quality_rejected"):
            return "n/a (deterministic quality gate, not a model score)"
        return "n/a (not reported)"
    try:
        return f"{float(c) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a (unparseable)"


def _fmt_rr(v) -> str:
    return "n/a" if v is None else f"{float(v):.0f} ms"


def _context_str(entry) -> str:
    neigh = entry.get("neighboring_labels") or []
    if len(neigh) >= 3:
        prev, cur, nxt = neigh[0], neigh[1], neigh[2]
    else:
        prev, cur, nxt = None, entry.get("predicted_label"), None
    return f"{prev or '—'}  →  [{cur or '?'}]  →  {nxt or '—'}"


def _coupling_note(entry) -> tuple[str, bool] | None:
    """A short interval note, only when the data supports one. Returns
    (text, is_warning).

    SAFETY: a long RR is only ever described as a possible pause when the
    interval is genuinely measured beat-to-beat. If it spans an
    SQI-blanked window, the interval is an artifact of removed data, not
    an observed cardiac event -- calling that a "pause" on a clinician
    form makes deleted data look like asystole, which is the single most
    dangerous confusion this document could introduce.
    """
    before, after = entry.get("rr_before_ms"), entry.get("rr_after_ms")
    spans_blank = entry.get("rr_after_spans_blanked_data")
    try:
        before = float(before) if before is not None else None
        after = float(after) if after is not None else None
    except (TypeError, ValueError):
        return None

    if after is not None and spans_blank:
        return (f"RR-after ({after:.0f} ms) spans removed data — NOT a measured pause. "
                f"Do not interpret as asystole.", True)
    if before is None or after is None or before <= 0:
        return None
    if after < 0.85 * before:
        return (f"Coupling interval {after:.0f} ms is {100 * (1 - after / before):.0f}% shorter "
                f"than the preceding RR — early beat.", False)
    if after > 1.15 * before:
        return (f"Following interval {after:.0f} ms exceeds the preceding RR "
                f"— possible pause (interval is fully measured).", False)
    return None


def draw_beat_page(c, entry, page_no, total, pipeline_version):
    tag = f"b{entry.get('beat_index', page_no)}"
    y = PAGE_H - MARGIN

    # ---- Title -------------------------------------------------------
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MARGIN, y - 4,
                 f"Beat {page_no} of {total}   |   Segment {entry.get('segment_id', '—')}")
    y -= 18
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#57606a"))
    c.drawString(MARGIN, y - 2,
                 f"Beat index {entry.get('beat_index', '—')}   |   "
                 f"t = {entry.get('r_peak_time_s', '—')} s   |   "
                 f"Pipeline confidence: {_fmt_conf(entry)}")
    y -= 12

    # ---- Section 1: ECG strip ---------------------------------------
    img = render_strip(entry)
    img_w = PAGE_W - 2 * MARGIN
    img_h = img_w * (2.7 / 7.4)
    c.drawImage(img, MARGIN, y - img_h, width=img_w, height=img_h,
                preserveAspectRatio=True, anchor="n", mask="auto")
    y -= img_h + 12

    # ---- Section 2: pipeline prediction ------------------------------
    label = entry.get("predicted_label", "?")
    # Every provenance/warning line gets its own full-width row -- these
    # used to be squeezed onto the RR row and overprinted it.
    n_extra = (1 if _coupling_note(entry) else 0) \
        + (1 if entry.get("strip_interpolated_fraction") else 0) \
        + (1 if entry.get("quality_rejected") else 0)
    box_h = 48 + 12 * n_extra
    c.setFillColor(HexColor("#f6f8fa"))
    c.setStrokeColor(HexColor("#d0d7de"))
    c.rect(MARGIN, y - box_h, PAGE_W - 2 * MARGIN, box_h, stroke=1, fill=1)

    c.setFillColor(HexColor(CLASS_COLORS.get(label, "#57606a")))
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(MARGIN + 10, y - 16,
                 f"Pipeline predicted:  {label} ({CLASS_NAMES.get(label, 'unknown')})"
                 f"   —   Confidence: {_fmt_conf(entry)}")
    c.setFillColor(HexColor("#24292f"))
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN + 10, y - 30, f"Rhythm context:  {_context_str(entry)}")
    c.drawString(MARGIN + 10, y - 42,
                 f"RR before: {_fmt_rr(entry.get('rr_before_ms'))}     "
                 f"RR after: {_fmt_rr(entry.get('rr_after_ms'))}")
    row = y - 54
    note = _coupling_note(entry)
    if note:
        text, is_warning = note
        c.setFillColor(HexColor("#cf222e" if is_warning else "#9a6700"))
        c.setFont("Helvetica-Bold" if is_warning else "Helvetica-Oblique", 8)
        c.drawString(MARGIN + 10, row, text)
        row -= 12
    frac = entry.get("strip_interpolated_fraction")
    if frac:
        c.setFillColor(HexColor("#cf222e"))
        c.setFont("Helvetica-Bold", 8)
        c.drawString(MARGIN + 10, row,
                     f"{frac * 100:.0f}% of this 3 s strip is removed data (shaded NO DATA) — "
                     f"not recorded signal.")
        row -= 12
    if entry.get("quality_rejected"):
        c.setFillColor(HexColor("#cf222e"))
        c.setFont("Helvetica-Oblique", 8.5)
        c.drawString(MARGIN + 10, row,
                     f"Flagged by quality gate: {entry.get('quality_reject_reason') or 'unspecified'}")
    y -= box_h + 14

    # ---- Section 3: MCQ ----------------------------------------------
    def question(title, group, options, ycur, note=None):
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 9.8)
        c.drawString(MARGIN, ycur, title)
        ycur -= 3
        if note:
            c.setFont("Helvetica-Oblique", 8)
            c.setFillColor(HexColor("#57606a"))
            # Right-aligned so it can never collide with a long question title.
            c.drawRightString(PAGE_W - MARGIN, ycur + 3, note)
        ycur -= 12
        for val, text in options:
            c.acroForm.radio(
                name=f"{tag}_{group}", value=val, selected=False,
                x=MARGIN + 6, y=ycur - 2, size=10.5,
                buttonStyle="circle", shape="circle",
                borderColor=HexColor("#57606a"), fillColor=white,
                textColor=black, borderWidth=0.8, forceBorder=True,
            )
            c.setFillColor(HexColor("#24292f"))
            c.setFont("Helvetica", 9)
            c.drawString(MARGIN + 24, ycur + 0.5, text)
            ycur -= 14.5
        return ycur - 4

    y = question("Q1.  What is your label for this beat?", "q1", Q1_OPTIONS, y)
    y = question("Q2.  Signal quality for this beat?", "q2", Q2_OPTIONS, y)
    y = question("Q3.  Do you agree with the pipeline prediction?", "q3", Q3_OPTIONS, y)
    y = question("Q4.  If you answered V or S above — clinically significant?", "q4",
                 Q4_OPTIONS, y, note="(skip if not applicable)")

    # free-text comment
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 9.8)
    c.drawString(MARGIN, y, "Comment (optional):")
    c.acroForm.textfield(
        name=f"{tag}_comment", value="", x=MARGIN + 105, y=y - 5,
        width=PAGE_W - 2 * MARGIN - 105, height=17, fontSize=9,
        borderColor=HexColor("#8c959f"), fillColor=white,
        textColor=black, borderWidth=0.7, forceBorder=True,
    )
    y -= 30

    # ---- Section 4: footer -------------------------------------------
    c.setStrokeColor(HexColor("#d0d7de"))
    c.line(MARGIN, y + 6, PAGE_W - MARGIN, y + 6)
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(HexColor("#cf222e"))
    c.drawString(MARGIN, y - 6, "This is a pseudo-label review, not a diagnosis.")
    c.setFont("Helvetica", 7.5)
    c.setFillColor(HexColor("#57606a"))
    # Two short lines rather than one long one, which previously ran off
    # the right edge and truncated mid-word.
    c.drawString(MARGIN, y - 17,
                 f"Pipeline version: {pipeline_version}    |    Page {page_no}/{total}")
    c.drawString(MARGIN, y - 27,
                 f"Segment: {entry.get('segment_id', '—')}    |    "
                 f"Amplitude: raw device units (uncalibrated, not mV)")
    c.drawString(MARGIN, y - 37,
                 f"Signal shown: {entry.get('signal_stage', 'preprocessed')}")
    c.showPage()


def draw_summary_page(c, entries, pipeline_version):
    y = PAGE_H - MARGIN
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, y, "Review summary")
    y -= 16
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#57606a"))
    c.drawString(MARGIN, y, f"{len(entries)} beat(s) in this document. "
                            f"Answers are stored in the form fields on each page.")
    y -= 24

    c.setFillColor(HexColor("#f6f8fa"))
    c.rect(MARGIN, y - 4, PAGE_W - 2 * MARGIN, 18, stroke=0, fill=1)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 9)
    for label, dx in [("Page", 0), ("Beat idx", 42), ("Segment", 100),
                      ("Predicted", 330), ("Conf.", 400), ("Context", 450)]:
        c.drawString(MARGIN + 4 + dx, y + 2, label)
    y -= 16

    c.setFont("Helvetica", 8.3)
    for i, e in enumerate(entries, start=1):
        if y < MARGIN + 120:
            c.showPage()
            y = PAGE_H - MARGIN
            c.setFont("Helvetica", 8.3)
        seg = str(e.get("segment_id", "—"))
        seg = seg if len(seg) <= 40 else seg[:37] + "..."
        c.setFillColor(HexColor("#24292f"))
        c.drawString(MARGIN + 4, y, str(i))
        c.drawString(MARGIN + 46, y, str(e.get("beat_index", "—")))
        c.drawString(MARGIN + 104, y, seg)
        c.setFillColor(HexColor(CLASS_COLORS.get(e.get("predicted_label"), "#57606a")))
        c.drawString(MARGIN + 334, y, str(e.get("predicted_label", "—")))
        c.setFillColor(HexColor("#24292f"))
        c.drawString(MARGIN + 404, y, _fmt_conf(e).split(" ")[0])
        c.drawString(MARGIN + 454, y, _context_str(e).replace("  ", ""))
        y -= 13

    y -= 20
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN, y, "Reviewer")
    y -= 20
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, y + 4, "Name:")
    c.acroForm.textfield(name="reviewer_name", value="", x=MARGIN + 40, y=y - 1,
                         width=200, height=17, fontSize=9,
                         borderColor=HexColor("#8c959f"), fillColor=white,
                         textColor=black, borderWidth=0.7, forceBorder=True)
    c.drawString(MARGIN + 260, y + 4, "Date:")
    c.acroForm.textfield(name="review_date", value="", x=MARGIN + 295, y=y - 1,
                         width=140, height=17, fontSize=9,
                         borderColor=HexColor("#8c959f"), fillColor=white,
                         textColor=black, borderWidth=0.7, forceBorder=True)
    y -= 34
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(HexColor("#cf222e"))
    c.drawString(MARGIN, y, "This is a pseudo-label review, not a diagnosis.")
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#57606a"))
    c.drawString(MARGIN, y - 12,
                 f"Pipeline version: {pipeline_version}    |    Generated {date.today().isoformat()}")
    c.showPage()


def build_pdf(entries, out_path: Path, pipeline_version: str):
    c = pdfcanvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle(f"VitalPatch beat review — {len(entries)} beat(s)")
    total = len(entries)
    for i, entry in enumerate(entries, start=1):
        draw_beat_page(c, entry, i, total, pipeline_version)
    draw_summary_page(c, entries, pipeline_version)
    c.save()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="beat predictions JSON")
    p.add_argument("--output", type=Path, required=True, help="PDF to write")
    p.add_argument("--pipeline-version", default="ecg_inference/five_class_xgb.json")
    p.add_argument("--max-beats", type=int, default=None,
                   help="cap pages (default: all beats in the input)")
    args = p.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"--input not found: {args.input}")
    data = json.loads(args.input.read_text())
    entries = data.get("beats", data) if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"No beat entries found in {args.input}")
    if args.max_beats:
        entries = entries[:args.max_beats]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(entries, args.output, args.pipeline_version)
    print(f"Wrote {args.output}  ({len(entries)} beat page(s) + 1 summary page)")


if __name__ == "__main__":
    main()
