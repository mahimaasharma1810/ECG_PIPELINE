"""Clinician review package for one ProRhythm capture (brief 8, 10.6).

Every verdict carries:
  - the ECG strip for that window, with detected R-peaks overlaid
  - the RR series, with flagged outliers SHOWN, not hidden
  - the computed features and the threshold applied
  - the SQI and the reason for any UNABLE_TO_DETERMINE
  - a narrative, structurally validated (brief 7) or the deterministic template

Outputs: <out>/report.json, <out>/report.md, <out>/window_*.png
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.ingest import load_capture
from rhythm.resample import to_uniform_segments
from rhythm.pipeline import (detect_peaks, BSQI_MIN, LAG1_MIN, SDNN_GUARD_MS,
                             DETECTOR_A, DETECTOR_B)
from rhythm.features import window_features, BEATS_PER_WINDOW, RR_MIN_MS, RR_MAX_MS, screen_rr
from rhythm.sqi import assess
from rhythm.verdict import decide, THRESHOLD_PROVENANCE, UNABLE
from rhythm.axes import assess_rate, assess_events, combine_axes
from rhythm.axes.rate import rate_trustworthy
from rhythm.axes.rate import PROVENANCE as RATE_PROVENANCE
from rhythm.axes.events import PROVENANCE as EVENTS_PROVENANCE
from rhythm.smoothing import smooth, DEFAULT_N_CONSECUTIVE
from rhythm.narrative import generate
from rhythm.domain_gate import read_gate

warnings.filterwarnings("ignore")


def figure(path, seg, w, rr_ms, flagged, feat, q, v, title):
    i0, i1 = int(w[0]), int(w[-1])
    t = np.arange(i0, i1 + 1) / seg.fs
    fig, ax = plt.subplots(2, 1, figsize=(15, 7),
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(t, seg.x[i0:i1 + 1], lw=.7, color="#222")
    ax[0].plot(w / seg.fs, seg.x[w], "v", ms=7, color="#d62728",
               label=f"R-peak ({DETECTOR_A})")
    ax[0].set_title(title, fontsize=10)
    ax[0].set_ylabel("amplitude (units UNKNOWN)", fontsize=8)
    ax[0].legend(fontsize=8, loc="upper right"); ax[0].grid(alpha=.25)

    idx = np.arange(rr_ms.size)
    ax[1].plot(idx, rr_ms, "-o", ms=3, lw=.9, color="#1f77b4", label="RR interval")
    if flagged.any():
        ax[1].plot(idx[flagged], rr_ms[flagged], "x", ms=10, mew=2, color="#d62728",
                   label=f"flagged, outside {RR_MIN_MS:.0f}-{RR_MAX_MS:.0f} ms "
                         f"(n={int(flagged.sum())}, shown not dropped)")
    ax[1].axhline(np.nanmean(rr_ms[~flagged]) if (~flagged).any() else np.nan,
                  ls="--", lw=.8, color="#555", label="mean RR")
    ax[1].set_xlabel("interval index (64 beats = 63 intervals)")
    ax[1].set_ylabel("RR (ms)", fontsize=8)
    ax[1].legend(fontsize=7, loc="upper right"); ax[1].grid(alpha=.25)

    fig.suptitle(f"{v.verdict}   |   RR CV {feat.rr_cv:.4f} vs threshold "
                 f"{v.threshold:.4f} (refusal band {v.margin_low:.4f}-{v.margin_high:.4f})"
                 f"   |   bSQI {q.bsqi:.3f}\n{v.reason}",
                 fontsize=9, y=1.00)
    fig.text(.5, -.02, SCOPE_STATEMENT, ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--no-model", action="store_true",
                    help="skip MedGemma and use the deterministic template only")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # Device-domain gate. Fail closed: no clinical severity, no MedGemma call,
    # until R-peak timing is validated against a beat-level reference.
    gate = read_gate()
    print(f"[device-domain gate] {gate.state}: "
          f"rhythm_verdict={gate.may_emit_rhythm_verdict} "
          f"clinical_severity={gate.may_emit_clinical_severity} "
          f"medgemma={gate.may_call_medgemma}")
    if not gate.may_call_medgemma:
        print(f"  -> {gate.reason}")
    cap = load_capture(a.input)
    segs, n_replayed = to_uniform_segments(cap.t_ms, cap.amplitude)
    if not segs:
        raise SystemExit("no usable segment >= 40 s in this capture")
    seg = max(segs, key=lambda s: s.duration_s)
    pa, pb = detect_peaks(seg.x, seg.fs)

    windows, verdicts, t0s, axes = [], [], [], []
    for s in range(0, pa.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
        w = pa[s:s + BEATS_PER_WINDOW]
        i0, i1 = int(w[0]), int(w[-1])
        fs_loc = seg.local_fs(i0, i1)
        if not np.isfinite(fs_loc) or fs_loc <= 0:
            continue
        rr = np.diff(w).astype(float) * 1000.0 / fs_loc
        feat = window_features(rr)
        pbw = pb[(pb >= i0) & (pb <= i1)]
        q = assess(seg.x[i0:i1 + 1], fs_loc, w.astype(float) * 1000.0 / fs_loc,
                   pbw.astype(float) * 1000.0 / fs_loc, rr_ms=rr,
                   bsqi_min=BSQI_MIN, lag1_min=LAG1_MIN)
        passed, reason = q.passed, q.reason
        if not passed and np.isfinite(feat.sdnn_ms) and feat.sdnn_ms <= SDNN_GUARD_MS:
            rem = [r for r in reason.split("; ") if "lag-1" not in r]
            passed, reason = (not rem), ("; ".join(rem) or "ok")
        v = decide(feat, passed, reason)
        rate_ok, rate_why = rate_trustworthy(q, feat)
        ra = assess_rate(feat, rate_ok, rate_why)
        ev = assess_events(rr, feat, passed)
        comb = combine_axes(ra, v, ev, gate.may_emit_clinical_severity)
        axes.append((ra, ev, comb))
        windows.append((w, rr, screen_rr(rr), feat, q, v, passed, reason, fs_loc))
        verdicts.append(v); t0s.append(float(seg.t_raw_ms[i0]))

    states, transitions = smooth(verdicts, t0s)

    # narratives + figures for the first N windows
    win_json = []
    for k, (w, rr, flg, feat, q, v, passed, reason, fs_loc) in enumerate(windows):
        ra, ev, comb = axes[k]
        nar = None
        if k < a.max_windows:
            n = generate(v.verdict, v.reason, v.evidence,
                         use_model=(not a.no_model) and gate.may_call_medgemma)
            nar = dict(text=n.text, source=n.source, accepted=n.accepted,
                       validation_failures=list(n.failures))
            figure(out / f"window_{k:03d}.png", seg, w, rr, flg, feat, q, v,
                   f"{Path(a.input).parent.name}/{Path(a.input).name} - window {k} "
                   f"(64 beats, fs_local {fs_loc:.2f} Hz)")
        win_json.append(dict(
            index=k, t0_ms=t0s[k], reported_state=states[k],
            verdict=v.verdict, actionable=v.actionable, action_state=v.action_state,
            reason=v.reason,
            features=feat.as_dict(), sqi=q.as_dict(),
            axes=dict(rate=ra.as_dict(), events=ev.as_dict(),
                      combined=comb.as_dict()),
            sqi_passed=bool(passed), sqi_reason=reason,
            threshold=v.threshold, refusal_band=[v.margin_low, v.margin_high],
            narrative=nar,
            figure=f"window_{k:03d}.png" if k < a.max_windows else None))

    counts, act = {}, {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
        act[v.action_state] = act.get(v.action_state, 0) + 1

    report = dict(
        scope_statement=SCOPE_STATEMENT,
        device_domain_gate=dict(
            state=gate.state, reason=gate.reason,
            may_emit_rhythm_verdict=gate.may_emit_rhythm_verdict,
            may_emit_clinical_severity=gate.may_emit_clinical_severity,
            may_call_medgemma=gate.may_call_medgemma),
        output_class=("ENGINEERING - rhythm verdicts with evidence; NO clinical "
                      "severity, NO MedGemma report"
                      if not gate.may_emit_clinical_severity else "CLINICAL"),
        generated_from=str(a.input), sha256=cap.sha256, subject=cap.subject,
        ingest=dict(n_raw_samples=cap.n_samples, replayed_rows_removed=n_replayed,
                    span_s=cap.span_s, segments=len(segs),
                    analysed_segment_duration_s=seg.duration_s,
                    measured_fs_hz=seg.fs),
        detectors=dict(primary=DETECTOR_A, secondary_for_bsqi=DETECTOR_B),
        gate=dict(bsqi_min=BSQI_MIN, lag1_min=LAG1_MIN, sdnn_guard_ms=SDNN_GUARD_MS),
        threshold_provenance=THRESHOLD_PROVENANCE,
        axis_provenance=dict(rate=RATE_PROVENANCE, events=EVENTS_PROVENANCE),
        hysteresis=dict(n_consecutive=DEFAULT_N_CONSECUTIVE,
                        transitions=[t.__dict__ for t in transitions]),
        window_verdict_counts=counts, action_state_counts=act,
        operating_point=dict(
            policy="Youden threshold; ONLY the REGULAR verdict is actionable",
            rationale=("NPV falls monotonically as the threshold tightens while PPV "
                       "only exceeds 0.80 where sensitivity has collapsed to 0.13. "
                       "IRREGULAR (PPV ~0.40) prompts review and must never fire an "
                       "alarm or reach an escalation desk on its own."),
            actionable_verdicts=["REGULAR"]),
        n_windows=len(windows), windows=win_json,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    md = [f"# Rhythm regularity review - {cap.subject}/{Path(a.input).name}", "",
          f"> **{SCOPE_STATEMENT}**", "",
          f"## Device-domain gate: {gate.state}", "",
          f"- rhythm verdict: {'permitted' if gate.may_emit_rhythm_verdict else 'BLOCKED'}",
          f"- clinical severity: {'permitted' if gate.may_emit_clinical_severity else '**BLOCKED**'}",
          f"- MedGemma report: {'permitted' if gate.may_call_medgemma else '**BLOCKED**'}",
          "", f"{gate.reason}", "",
          "## Provenance", "",
          f"- SHA-256 `{cap.sha256[:32]}...`",
          f"- Raw samples {cap.n_samples:,}; **replayed rows removed {n_replayed:,}**",
          f"- Analysed segment {seg.duration_s:.0f} s at measured **{seg.fs:.3f} Hz**",
          f"- Detectors: `{DETECTOR_A}` (primary), `{DETECTOR_B}` (bSQI second opinion)",
          f"- Gate: bSQI >= {BSQI_MIN}, lag-1 >= {LAG1_MIN} guarded above SDNN {SDNN_GUARD_MS} ms",
          "", "## Threshold", "",
          f"RR CV >= **{THRESHOLD_PROVENANCE['threshold']}**, refusal band "
          f"+/-{THRESHOLD_PROVENANCE['refusal_half_width']}", "",
          f"| field | value |", "|---|---|"]
    for k, val in THRESHOLD_PROVENANCE.items():
        md.append(f"| {k} | {val} |")
    md += ["", "## Operating point", "",
           "**Youden threshold; only the REGULAR verdict is actionable.**",
           "NPV falls monotonically as the threshold tightens, while PPV only exceeds",
           "0.80 where sensitivity has collapsed to 0.13. A REGULAR call is trustworthy",
           "(NPV ~0.998); an IRREGULAR call (PPV ~0.40) prompts a look and must never",
           "fire an alarm on its own.", "",
           "## Verdicts", "",
           f"{len(windows)} windows of 64 beats:", ""]
    for k, val in sorted(counts.items()):
        md.append(f"- **{k}**: {val} ({val/len(windows):.1%})")
    md += [""] + [f"- action state **{k}**: {v} ({v/len(windows):.1%})"
                  for k, v in sorted(act.items())]
    md += ["", f"Hysteresis transitions ({DEFAULT_N_CONSECUTIVE} agreeing windows required): "
           f"{len(transitions)}", ""]
    for t in transitions:
        md.append(f"- window {t.at_index}: {t.from_state} -> {t.to_state} "
                  f"(n={t.n_agreeing}, RR CV {t.evidence.get('rr_cv')})")
    from collections import Counter
    md += ["", "## Axes", "",
           "Three independent axes, each refusing independently. "
           "A flat label cannot express partial confidence.", ""]
    for label, vals in (("Rate", [a[0].state for a in axes]),
                        ("Regularity", [v.verdict for v in verdicts]),
                        ("Events (ectopy burden)", [a[1].burden for a in axes]),
                        ("Combined severity", [a[2].severity for a in axes])):
        c = Counter(vals)
        md.append(f"- **{label}**: " +
                  ", ".join(f"{k} {n} ({n/len(axes):.0%})"
                            for k, n in c.most_common()))
    n_pause = sum(a[1].pause_count for a in axes)
    n_esc = sum(1 for a in axes if a[2].escalate)
    md += ["", f"- pauses over 2000 ms: **{n_pause}** across {len(axes)} windows",
           f"- windows escalating to CRITICAL: **{n_esc}**", "",
           "> Rate bands 40/150 bpm are ADOPTED convention, not derived or "
           "validated in this project. The ectopy axis reports BURDEN only - "
           "the TYPE of ectopic beat is not determinable from beat timing.", "",
           "## Windows", ""]
    for wj in win_json[:a.max_windows]:
        md += [f"### Window {wj['index']} - {wj['verdict']}", "",
               f"![window]({wj['figure']})", "",
               f"- {wj['reason']}",
               f"- RR CV {wj['features']['rr_cv']}, RMSSD {wj['features']['rmssd_ms']} ms, "
               f"HR {wj['features']['hr_bpm']} bpm",
               f"- flagged intervals {wj['features']['n_flagged']}/{wj['features']['n_intervals']}",
               f"- bSQI {wj['sqi']['bsqi']}, SQI {'PASS' if wj['sqi_passed'] else 'FAIL'} - {wj['sqi_reason']}",
               f"- **axes** - rate: {wj['axes']['rate']['state']} "
               f"| regularity: {wj['verdict']} "
               f"| events: {wj['axes']['events']['burden']} "
               f"({wj['axes']['events']['ectopic_couplets']} couplets, "
               f"{wj['axes']['events']['pause_count']} pauses) "
               f"-> **{wj['axes']['combined']['severity']}**",
               f"- narrative source: **{wj['narrative']['source']}**"
               + (f" (model output rejected: {wj['narrative']['validation_failures']})"
                  if wj['narrative'] and not wj['narrative']['accepted']
                     and wj['narrative']['validation_failures'] else ""),
               "", f"> {wj['narrative']['text']}", ""]
    md += ["", f"> **{SCOPE_STATEMENT}**", ""]
    (out / "report.md").write_text("\n".join(md))

    print(f"windows: {len(windows)}   verdicts: {counts}")
    print(f"hysteresis transitions: {len(transitions)}")
    print(f"wrote {out}/report.json, {out}/report.md, {min(a.max_windows,len(windows))} figures")
    print(f"\n{SCOPE_STATEMENT}")


if __name__ == "__main__":
    main()
