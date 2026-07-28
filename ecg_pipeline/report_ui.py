"""report_ui.py -- minimal, READ-ONLY local visualization of the ECG
pipeline, hosted as a tiny local Flask server.

============================================================================
TWO MODES
============================================================================
1. LIVE REPLAY SERVER (default -- `python -m ecg_pipeline.report_ui`):
   a locally-hosted page where you pick a source (a synthetic scenario or
   a recorded VitalPatch file) and a replay speed, click Start, and watch
   it stream through 4 stages -- RAW -> PREPROCESSED -> ASSESSED (beats +
   rhythm) -> MedGemma REPORT -- with the risk level evolving live as more
   signal "arrives". This is a REPLAY of a recording (real or synthetic),
   never a live device feed -- the header always says "REPLAY (recorded)"
   or "REPLAY (synthetic)".
2. STATIC SINGLE-REPORT PAGE (legacy, `--report <path.json>` / `--list`):
   generates one self-contained HTML file for an already-saved report.
   Kept because it's still the simplest way to look at one specific saved
   report.json without starting a server.

============================================================================
SCOPE / READ-ONLY GUARANTEE
============================================================================
This is a DEMO/VISUALIZATION tool only. It never re-implements detection,
filtering, or classification -- every graph is built from arrays the
EXISTING, unmodified ECGPipeline / agent_bridge actually produced for that
exact input:
  * RAW panel: `recording.signal_mv` as parsed by the existing
    parse_vitalpatch_ecg/parse_sensio_ecg -- untouched.
  * PREPROCESSED panel: the output of `demo_stream.compute_filtered_signal`,
    which calls the SAME public Stage 2-4 functions (`run_sqi_gate`,
    `to_target_rate`, `apply_filter_chain`) that ECGPipeline.run() calls
    internally, with the identical inputs -- deterministic, so the result
    is byte-identical to what the real run computed. (ECGPipeline.run()
    doesn't currently return this intermediate array in PipelineResult, so
    recomputing it via the same public functions -- rather than modifying
    ecg_pipeline_core.py to add a field -- is how this stays read-only.)
  * ASSESSED panel: beats at `PipelineResult.beats[i].r_peak_ms`, colored
    by `PipelineResult.beat_labels[i]` -- the classifier's own output,
    never re-detected/re-classified here. Rhythm-finding regions shaded
    from `PipelineResult.rhythm_findings` / the report's own JSON.
  * REPORT panel: `agent_bridge.build_risk_report_json()` /
    `agent_bridge.run_full_report()`'s own risk_level, deciding_rule,
    beat_summary, and narrative -- verbatim.
Recorded and synthetic inputs are pushed through the exact same
`ECGPipeline` / `agent_bridge` calls; the replay layer only changes how
fast/chunked the input arrives, never which functions process it.

Run:
    python -m ecg_pipeline.report_ui                          # serves the live replay UI
    python -m ecg_pipeline.report_ui --port 5057
    python -m ecg_pipeline.report_ui --list                    # legacy: list saved reports
    python -m ecg_pipeline.report_ui --report <path.json> [--open]   # legacy: static page
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import threading
import time
import traceback
import uuid
import webbrowser
from html import escape
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from ecg_pipeline import agent_bridge
from ecg_pipeline.agent_bridge import load_classifier
from ecg_pipeline.demo_stream import (
    DATA_REPORTS_SYNTHETIC, _load_real_recordings, _sub_recording, build_waveform_block,
    compute_filtered_signal, save_synthetic_report,
)
from ecg_pipeline.ecg_pipeline_core import (
    DATA_RAW, ECGPipeline, MODELS_DIR, TARGET_FS, discover_vitalpatch_files, parse_sensio_ecg,
    parse_vitalpatch_ecg,
)
from ecg_pipeline.synthetic_ecg import SCENARIOS, generate_scenario

DATA_REPORTS = DATA_RAW.parent / "reports"
DEFAULT_OUT_DIR = DATA_RAW.parent / "reports_ui"
DEFAULT_PORT = 5057
MAX_RECORDED_FILES_LISTED = 30  # keep the landing page's dropdown small; see report_ui docstring

RISK_COLORS = {
    "LOW": "#2e7d32",
    "MEDIUM": "#f9a825",
    "HIGH": "#ef6c00",
    "CRITICAL": "#c62828",
    "NOT_ASSESSABLE": "#616161",
}
CLASS_COLORS = {"N": "#2e7d32", "S": "#1565c0", "V": "#c62828", "F": "#8e24aa", "Q": "#9e9e9e"}

_CLASSIFIER_CACHE = None


def _classifier():
    global _CLASSIFIER_CACHE
    if _CLASSIFIER_CACHE is None:
        _CLASSIFIER_CACHE = load_classifier(MODELS_DIR / "five_class_xgb.json")
    return _CLASSIFIER_CACHE


def _find_raw_file(source: str, patient_id: str, segment_id: str) -> Path | None:
    """Reverse-engineers the raw file path from the naming conventions
    used by parse_vitalpatch_ecg / batch_vitalpatch_report.py and
    batch_prorhythm_report.py (segment_id = f"{csv_stem}_seg{n}" for
    vitalpatch; the prorhythm/SeNSiO path is one file == one recording)."""
    if source == "vitalpatch":
        stem = re.sub(r"_seg\d+$", "", segment_id)
        candidate = DATA_RAW / "vitalpatch" / f"Patch_{patient_id}" / f"{stem}.csv"
        return candidate if candidate.exists() else None
    if source == "sensio":
        # parse_sensio_ecg() sets Recording.source="sensio" (see ecg_pipeline_core.py),
        # but these raw files live under data/raw/prorhythm/ -- confirmed directly via
        # batch_prorhythm_report.py, which reuses parse_sensio_ecg() for that directory.
        stem = re.sub(r"_seg\d+$", "", segment_id)
        root = DATA_RAW / "prorhythm"
        if root.exists():
            for p in root.rglob("*.csv"):
                if p.stem in (segment_id, stem):
                    return p
    return None


def _reconstruct_waveform(report_json: dict) -> dict | None:
    if report_json.get("waveform"):
        return report_json["waveform"]

    r = report_json["recording"]
    raw_path = _find_raw_file(r["source"], r["patient_id"], r["segment_id"])
    if raw_path is None:
        return None

    if r["source"] == "vitalpatch":
        recordings = parse_vitalpatch_ecg(raw_path)
        recording = next((rec for rec in recordings if rec.segment_id == r["segment_id"]), None)
    elif r["source"] == "sensio":
        recording = parse_sensio_ecg(raw_path)
    else:
        recording = None
    if recording is None:
        return None

    pipeline = ECGPipeline(classifier=_classifier())
    result = pipeline.run(recording)
    return build_waveform_block(recording, result)


def _render_plot_png(waveform: dict, rhythm_findings: list[dict]) -> str:
    fs = waveform["fs"]
    signal = np.asarray(waveform["signal"], dtype=float)
    t = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=130)
    ax.plot(t, signal, color="#37474f", linewidth=0.6, zorder=1)

    # Blended transform: x in data coords (seconds), y in axes-fraction coords, so the
    # finding-kind label sits just under the top of the plot regardless of signal amplitude.
    label_trans = blended_transform_factory(ax.transData, ax.transAxes)
    for f in rhythm_findings:
        if f.get("start_time_s") is None or f.get("duration_s") is None:
            continue
        x0 = f["start_time_s"]
        x1 = x0 + max(f["duration_s"], 0.1)
        ax.axvspan(x0, x1, color="#ffca28", alpha=0.3, zorder=2)
        # Label the band with the finding kind directly -- the axis is in seconds while
        # rhythm findings are also described by beat index (e.g. "beats 40-59"), which can
        # otherwise read as if the highlighted region doesn't match the stated finding.
        ax.text((x0 + x1) / 2, 0.94, f["kind"], transform=label_trans, ha="center", va="top",
                fontsize=6, color="#7a5c00", zorder=4)

    seen = set()
    for b in waveform["beats"]:
        if b.get("quality_rejected"):
            continue
        r_s = b["r_peak_ms"] / 1000.0
        idx = int(round(r_s * fs))
        if not (0 <= idx < len(signal)):
            continue
        color = CLASS_COLORS.get(b["label"], "#000000")
        label = b["label"] if b["label"] not in seen else None
        ax.scatter([r_s], [signal[idx]], color=color, s=16, zorder=3, label=label)
        seen.add(b["label"])

    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude (a.u.)")
    if seen:
        ax.legend(loc="upper right", fontsize=7, ncol=min(len(seen), 3), frameon=False)
    fig.tight_layout()

    buf = io.BytesIO()
    # bbox_inches="tight" so a legend/label that would otherwise overflow the axes (the
    # "N . S" clipping issue) gets included in the saved image instead of cropped off.
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _beat_summary_table(beat_summary: dict) -> str:
    rows = []
    for cls, info in beat_summary.items():
        flag = " &mdash; LOW CONFIDENCE" if info.get("confidence") == "LOW" else ""
        rows.append(
            f"<tr><td>{escape(cls)}</td><td>{info['count']}</td>"
            f"<td>{info['pct_of_analyzed_beats']}%</td><td>{escape(info['confidence'])}{flag}</td></tr>"
        )
    return "\n".join(rows) or "<tr><td colspan=4>(no beats survived quality gating)</td></tr>"


def _rhythm_findings_list(findings: list[dict]) -> str:
    if not findings:
        return "<li>(none detected)</li>"
    return "\n".join(
        f"<li><b>{escape(f['kind'])}</b>: beats {f['start_beat_idx']}-{f['end_beat_idx']} "
        f"(starts {f.get('start_time_s')}s, duration {f.get('duration_s')}s) &mdash; "
        f"{escape(f.get('evidence_text', ''))}</li>"
        for f in findings
    )


PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>ECG Report -- {segment_id}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }}
  .banner {{ background: #fff3cd; border: 2px solid #d4a017; color: #7a5c00; padding: 0.75rem 1rem;
             border-radius: 6px; font-weight: bold; margin-bottom: 1rem; }}
  .warn {{ background: #eceff1; border: 2px solid #616161; color: #37474f; padding: 0.75rem 1rem;
           border-radius: 6px; font-weight: bold; margin-bottom: 1rem; }}
  .header {{ display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; margin-bottom: 1rem; }}
  .risk-pill {{ display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px; color: white;
                font-weight: bold; background: {risk_color}; }}
  .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; align-items: start; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ddd; padding: 0.3rem 0.5rem; text-align: left; font-size: 0.9rem; }}
  .card {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
  .narrative {{ white-space: pre-wrap; font-size: 0.92rem; line-height: 1.4; }}
  img {{ max-width: 100%; }}
  h2 {{ margin-top: 0; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
{synthetic_banner}
{not_assessable_banner}
<div class="header">
  <h1 style="margin:0">ECG Report</h1>
  <span class="risk-pill">{final_risk_level}</span>
</div>
<div class="card">
  <b>Source:</b> {source} &nbsp; <b>Patient:</b> {patient_id} &nbsp; <b>Segment:</b> {segment_id}<br/>
  <b>Duration:</b> {duration_s}s &nbsp; <b>Beats detected/analyzed:</b> {n_detected}/{n_analyzed}<br/>
  <b>Quality score:</b> {quality_score} &nbsp; <b>MedGemma status:</b> {medgemma_status}
</div>
<div class="card">
  <h2>ECG Waveform</h2>
  {plot_html}
</div>
<div class="grid">
  <div class="card">
    <h2>Findings</h2>
    <h3>Beat classes</h3>
    <table>
      <tr><th>Class</th><th>Count</th><th>% of analyzed</th><th>Confidence</th></tr>
      {beat_rows}
    </table>
    <h3>Rhythm findings</h3>
    <ul>{rhythm_list}</ul>
    <h3>Deciding rule</h3>
    <p>{deciding_html}</p>
  </div>
  <div class="card">
    <h2>MedGemma Clinical Interpretation</h2>
    <div class="narrative">{narrative}</div>
  </div>
</div>
</body>
</html>
"""


def build_page(report_path: Path, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    report_json = json.loads(Path(report_path).read_text())
    r = report_json["recording"]
    is_synthetic = bool(report_json.get("synthetic"))
    assessable = report_json.get("assessable", True)
    final_level = report_json.get("final_risk_level") or report_json.get("risk_level", "?")

    synthetic_banner = ""
    if is_synthetic:
        synthetic_banner = (
            '<div class="banner">SYNTHETIC &mdash; NOT A REAL PATIENT. Generated by '
            'ecg_pipeline/synthetic_ecg.py for pipeline demo/testing only. '
            'Never used for classifier training or evaluation.</div>'
        )

    not_assessable_banner = ""
    if not assessable:
        not_assessable_banner = (
            f'<div class="warn">INSUFFICIENT SIGNAL &mdash; NOT ASSESSABLE. '
            f'{escape(str(report_json.get("not_assessable_reason", "")))}</div>'
        )

    waveform = _reconstruct_waveform(report_json)
    if waveform and waveform.get("signal"):
        plot_b64 = _render_plot_png(waveform, report_json.get("rhythm_findings", []))
        plot_html = f'<img src="data:image/png;base64,{plot_b64}" alt="ECG waveform"/>'
    else:
        plot_html = ("<p><em>No waveform data available for this report -- neither embedded in the JSON "
                     "nor reconstructable from a raw file at the expected path.</em></p>")

    deciding = report_json.get("deciding_rule", {})
    _measured, _threshold = deciding.get("measured_value"), deciding.get("threshold")
    if _measured is None and _threshold is None:
        # Catch-all "nothing fired" rule -- no real measured/threshold value exists, so don't
        # print "None"/"null", and don't call it "FIRED" (that would imply a real threshold
        # was crossed when none was).
        deciding_html = (
            f"No dangerous thresholds exceeded &rarr; "
            f"<b>risk {escape(str(deciding.get('would_set_level') or 'LOW'))}</b>"
        )
    else:
        deciding_html = (
            f"{escape(str(deciding.get('condition')))}: measured {escape(str(_measured))} "
            f"vs threshold {escape(str(_threshold))} &rarr; "
            f"<b>{'EXCEEDED' if deciding.get('fired') else 'not exceeded'}</b>"
        )

    html = PAGE_TEMPLATE.format(
        segment_id=escape(r["segment_id"]),
        risk_color=RISK_COLORS.get(final_level, "#455a64"),
        synthetic_banner=synthetic_banner,
        not_assessable_banner=not_assessable_banner,
        final_risk_level=escape(str(final_level)),
        source=escape(r["source"]),
        patient_id=escape(str(r["patient_id"])),
        duration_s=r.get("duration_s"),
        n_detected=r.get("n_beats_detected"),
        n_analyzed=r.get("n_beats_analyzed"),
        quality_score=r.get("quality_score"),
        medgemma_status=escape(str(report_json.get("medgemma", {}).get("status", "?"))),
        plot_html=plot_html,
        beat_rows=_beat_summary_table(report_json.get("beat_summary", {})),
        rhythm_list=_rhythm_findings_list(report_json.get("rhythm_findings", [])),
        deciding_html=deciding_html,
        narrative=escape(report_json.get("narrative", "(no narrative)")),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{r['segment_id']}.html"
    out_path.write_text(html)
    return out_path


def _list_available(limit: int = 200) -> list[tuple[str, str, str, str]]:
    rows = []
    for manifest_path in sorted(DATA_REPORTS.glob("*_run_manifest.csv")):
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("error"):
                    continue
                rows.append((row.get("source", ""), row.get("patient_id", ""),
                             row.get("segment_id", ""), row.get("final_risk_level", "")))
    synthetic_dir = DATA_REPORTS / "synthetic"
    if synthetic_dir.exists():
        for p in sorted(synthetic_dir.rglob("*.json")):
            try:
                rj = json.loads(p.read_text())
                risk = rj.get("final_risk_level", rj.get("risk_level", ""))
            except (json.JSONDecodeError, OSError):
                risk = ""
            rows.append(("synthetic", p.parent.name, p.stem, risk))
    return rows[:limit]


# ============================================================================
# LIVE REPLAY SERVER
# ============================================================================
# In-memory run registry: run_id -> mutable state dict, updated by a
# background replay thread and read by /api/state/<run_id> polls. A
# process-local dict is adequate for a single-user local demo tool (no
# database, no auth, per the task's "keep it minimal" constraint).
_RUNS: dict[str, dict] = {}
_RUNS_LOCK = threading.Lock()

_CLASSIFIER_FOR_SERVER = None


def _server_classifier():
    global _CLASSIFIER_FOR_SERVER
    if _CLASSIFIER_FOR_SERVER is None:
        _CLASSIFIER_FOR_SERVER = agent_bridge.load_classifier(MODELS_DIR / "five_class_xgb.json")
    return _CLASSIFIER_FOR_SERVER


def _render_stage_plot_png(partial, filtered: np.ndarray, result, report_json: dict) -> str:
    """The 3 signal-stage panels (raw / preprocessed / assessed) as one
    stacked PNG, all drawn from arrays captured off the real pipeline run
    for this exact chunk -- see the module docstring's accuracy guarantee."""
    fs_raw = partial.fs_nominal
    raw = partial.signal_mv
    t_raw = np.arange(len(raw)) / fs_raw
    t_filt = np.arange(len(filtered)) / TARGET_FS
    x_max = max(t_raw[-1] if len(t_raw) else 1.0, t_filt[-1] if len(t_filt) else 1.0, 1.0)

    # Compact layout target: ~1450px wide, short enough that all 3 stacked
    # panels fit one screenshot alongside the findings/report panels below --
    # width (time-axis resolution) is kept GENEROUS (more inches wide than
    # the previous version) specifically so shrinking the HEIGHT doesn't
    # compress/blur the beat markers; only vertical space is reduced.
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 5.6), dpi=130)

    TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 9.5, 8, 7, 6.5

    axes[0].plot(t_raw, raw, color="#455a64", linewidth=0.5)
    axes[0].set_title(f"1. RAW ECG (as parsed, fs={fs_raw:g}Hz)", fontsize=TITLE_FS)
    axes[0].set_ylabel("raw amplitude", fontsize=LABEL_FS)

    axes[1].plot(t_filt, filtered, color="#37474f", linewidth=0.6)
    axes[1].set_title(f"2. PREPROCESSED ECG (filter chain output, resampled {TARGET_FS:g}Hz)", fontsize=TITLE_FS)
    axes[1].set_ylabel("filtered amplitude", fontsize=LABEL_FS)

    axes[2].plot(t_filt, filtered, color="#37474f", linewidth=0.6, zorder=1)
    # Blended transform: x in data coords (seconds), y in axes-fraction coords, so the
    # finding-kind label sits just under the top of the panel regardless of signal amplitude.
    label_trans = blended_transform_factory(axes[2].transData, axes[2].transAxes)
    for f in report_json.get("rhythm_findings", []):
        if f.get("start_time_s") is None or f.get("duration_s") is None:
            continue
        x0 = f["start_time_s"]
        x1 = x0 + max(f["duration_s"], 0.1)
        axes[2].axvspan(x0, x1, color="#ffca28", alpha=0.3, zorder=2)
        # Label the band with the finding kind directly -- the axis is in seconds while
        # rhythm findings are also described by beat index (e.g. "beats 40-59"), which can
        # otherwise read as if the highlighted region doesn't match the stated finding.
        axes[2].text((x0 + x1) / 2, 0.94, f["kind"], transform=label_trans, ha="center", va="top",
                    fontsize=6, color="#7a5c00", zorder=4)
    seen = set()
    for b, lbl in zip(result.beats, result.beat_labels):
        if b.quality_rejected:
            continue
        idx = int(round(b.r_peak_ms / 1000.0 * TARGET_FS))
        if not (0 <= idx < len(filtered)):
            continue
        color = CLASS_COLORS.get(lbl, "#000000")
        axes[2].scatter([b.r_peak_ms / 1000.0], [filtered[idx]], color=color, s=10, zorder=3,
                        label=(lbl if lbl not in seen else None))
        seen.add(lbl)
    axes[2].set_title("3. ASSESSED (pipeline's own beat classes + rhythm findings)", fontsize=TITLE_FS)
    axes[2].set_xlabel("time (s)", fontsize=LABEL_FS)
    axes[2].set_ylabel("filtered amplitude", fontsize=LABEL_FS)
    if seen:
        axes[2].legend(loc="upper right", fontsize=LEGEND_FS, ncol=min(len(seen), 3), frameon=False,
                       handletextpad=0.2, columnspacing=0.6)

    for ax in axes:
        ax.set_xlim(0, x_max)
        ax.tick_params(axis="both", labelsize=TICK_FS)
    fig.tight_layout(pad=0.6, h_pad=0.8)

    buf = io.BytesIO()
    # bbox_inches="tight" so a legend/label that would otherwise overflow the axes (the
    # "N . S" clipping issue) gets included in the saved image instead of cropped off.
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _public_state(state: dict) -> dict:
    """A JSON-safe copy of a run's state (drops the GroundTruth dataclass
    instance, which the worker keeps around only to save the final
    synthetic report correctly)."""
    return {k: v for k, v in state.items() if k != "ground_truth"}


def _replay_worker(run_id: str, recording, classifier, chunk_seconds: float, speed: float) -> None:
    """The ONLY new logic in this tool: a chunk/timing wrapper around the
    EXISTING, unmodified ECGPipeline + agent_bridge calls. Recorded and
    synthetic recordings take this identical path -- only chunk timing
    differs, never the parse/filter/detect/classify/cascade stages."""
    state = _RUNS[run_id]
    fs = recording.fs_nominal
    chunk_samples = max(1, int(round(chunk_seconds * fs)))
    n_total = len(recording.signal_mv)
    n_chunks = max(1, int(np.ceil(n_total / chunk_samples)))

    with _RUNS_LOCK:
        state["n_chunks"] = n_chunks
        state["total_s"] = round(n_total / fs, 1)

    try:
        for i in range(1, n_chunks + 1):
            n_so_far = min(i * chunk_samples, n_total)
            partial = _sub_recording(recording, n_so_far)
            pipeline = ECGPipeline(classifier=classifier)
            result = pipeline.run(partial)
            report_json = agent_bridge.build_risk_report_json(result)
            filtered = compute_filtered_signal(partial)
            plot_png_b64 = _render_stage_plot_png(partial, filtered, result, report_json)

            with _RUNS_LOCK:
                state["chunk_i"] = i
                state["elapsed_s"] = round(n_so_far / fs, 1)
                state["risk_level"] = report_json["risk_level"]
                state["assessable"] = report_json["assessable"]
                state["not_assessable_reason"] = report_json.get("not_assessable_reason")
                state["rhythm_findings"] = report_json["rhythm_findings"]
                state["beat_summary"] = report_json["beat_summary"]
                state["deciding_rule"] = report_json["deciding_rule"]
                state["n_beats_detected"] = report_json["recording"]["n_beats_detected"]
                state["n_beats_analyzed"] = report_json["recording"]["n_beats_analyzed"]
                state["plot_png_b64"] = plot_png_b64
                if not state["risk_history"] or state["risk_history"][-1]["level"] != report_json["risk_level"]:
                    state["risk_history"].append({"t_s": round(n_so_far / fs, 1), "level": report_json["risk_level"]})

            if speed > 0 and i < n_chunks:
                time.sleep(chunk_seconds / speed)

        # One final full pass -- the SAME call the normal (non-replay) batch
        # scripts use -- so the completed report is byte-identical to what
        # running this file/scenario the ordinary way would produce.
        final_report_json, final_result = agent_bridge.run_full_report(recording, classifier)
        ground_truth = state.get("ground_truth")
        if ground_truth is not None:
            json_path, md_path = save_synthetic_report(final_report_json, ground_truth, recording, final_result,
                                                         out_root=DATA_REPORTS_SYNTHETIC)
        else:
            out_dir = DATA_RAW.parent / "reports" / "replay_ui"
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path, md_path = agent_bridge.save_report(final_report_json, out_dir, recording.segment_id)

        with _RUNS_LOCK:
            state["status"] = "done"
            state["final_report"] = final_report_json
            state["saved_json_path"] = str(json_path)
    except Exception as e:  # noqa: BLE001 -- surfaced to the UI, not swallowed
        with _RUNS_LOCK:
            state["status"] = "error"
            state["error"] = f"{type(e).__name__}: {e}"
            state["error_trace"] = traceback.format_exc(limit=6)


LANDING_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>ECG Pipeline -- Replay Demo</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; background: #fafafa; color: #1a1a1a; }}
.card {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; max-width: 640px; }}
label {{ display: block; margin-top: 0.6rem; font-weight: 600; font-size: 0.9rem; }}
select, button {{ font-size: 1rem; padding: 0.4rem; margin-top: 0.2rem; }}
button {{ margin-top: 1rem; padding: 0.5rem 1.2rem; background: #1565c0; color: white; border: none;
          border-radius: 6px; cursor: pointer; }}
.note {{ font-size: 0.85rem; color: #616161; margin-top: 1rem; }}
h1 {{ margin-bottom: 0.2rem; }}
</style></head>
<body>
<h1>ECG Pipeline &mdash; Replay Demo</h1>
<p class="note">This REPLAYS a recording (synthetic or recorded VitalPatch) through the real, unmodified
pipeline in timed chunks &mdash; it is not a live device feed.</p>

<div class="card">
  <form action="/start" method="post">
    <label>Source</label>
    <select name="mode" id="mode" onchange="document.getElementById('synth-row').style.display=(this.value=='synthetic')?'block':'none'; document.getElementById('rec-row').style.display=(this.value=='recorded')?'block':'none';">
      <option value="synthetic">Synthetic scenario</option>
      <option value="recorded">Recorded VitalPatch file</option>
    </select>

    <div id="synth-row">
      <label>Scenario</label>
      <select name="scenario">{scenario_options}</select>
    </div>
    <div id="rec-row" style="display:none">
      <label>Recorded file (first {max_files} shown)</label>
      <select name="file_idx">{file_options}</select>
    </div>

    <label>Replay speed</label>
    <select name="speed">
      <option value="1">1x (real time)</option>
      <option value="5" selected>5x</option>
      <option value="20">20x</option>
      <option value="0">Instant (no delay)</option>
    </select>

    <button type="submit">Start Replay</button>
  </form>
</div>
</body></html>
"""

REPLAY_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Replay -- {run_id}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0.8rem 1.2rem; background: #fafafa;
        color: #1a1a1a; font-size: 14px; }}
.banner {{ background: #fff3cd; border: 2px solid #d4a017; color: #7a5c00; padding: 0.4rem 0.8rem; border-radius: 5px;
           font-weight: bold; margin-bottom: 0.5rem; font-size: 0.85rem; }}
.warn {{ background: #eceff1; border: 2px solid #616161; color: #37474f; padding: 0.4rem 0.8rem; border-radius: 5px;
         font-weight: bold; margin-bottom: 0.5rem; font-size: 0.85rem; }}
.fallback-note {{ background: #e3f2fd; border: 1px solid #64b5f6; color: #0d47a1; padding: 0.35rem 0.7rem;
                  border-radius: 5px; font-size: 0.8rem; margin-bottom: 0.5rem; display: inline-block; }}
.pill {{ display:inline-block; padding:.2rem .8rem; border-radius:999px; color:#fff; font-weight:bold; font-size:0.85rem; }}
.card {{ background: white; border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.6rem; }}
.grid {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 0.8rem; align-items: start; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; }}
td, th {{ border: 1px solid #ddd; padding: 0.15rem 0.4rem; text-align: left; }}
img {{ max-width: 100%; display: block; }}
h2 {{ margin: 0.2rem 0; font-size: 1.2rem; }}
h3 {{ margin: 0.1rem 0 0.4rem 0; font-size: 0.95rem; }}
h4 {{ margin: 0.5rem 0 0.15rem 0; font-size: 0.85rem; }}
ul {{ margin: 0.2rem 0; padding-left: 1.2rem; font-size: 0.82rem; }}
.narrative {{ white-space: pre-wrap; font-size: 0.8rem; line-height: 1.35; max-height: 260px; overflow-y: auto; }}
.timeline span {{ display:inline-block; padding:.1rem .4rem; border-radius:4px; color:#fff; font-size:.72rem;
                   margin-right:.25rem; margin-bottom:.2rem; }}
#js-error {{ display:none; background:#c62828; color:white; padding:0.5rem; border-radius:5px; margin-bottom:0.5rem; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head>
<body>
<div id="js-error"></div>
<div id="synthetic-banner"></div>
<div id="not-assessable-banner"></div>
<h2 id="label">Replay</h2>
<div class="card">
  <b>Segment:</b> <span id="segment_id"></span> &nbsp;
  <b>Elapsed / Total:</b> <span id="elapsed"></span>s / <span id="total"></span>s &nbsp;
  <b>Chunk:</b> <span id="chunk_i"></span>/<span id="n_chunks"></span> &nbsp;
  <b>Risk:</b> <span id="risk_pill" class="pill">?</span>
  &nbsp; <span id="timeline" class="timeline" style="display:inline"></span>
</div>
<div class="card">
  <h3>1-2-3. Raw / Preprocessed / Assessed</h3>
  <img id="plot" src=""/>
</div>
<div class="grid">
  <div class="card">
    <h3>Findings (live)</h3>
    <table><tr><th>Class</th><th>Count</th><th>%</th><th>Confidence</th></tr><tbody id="beat_table"></tbody></table>
    <h4>Rhythm findings</h4>
    <ul id="rhythm_list"></ul>
    <h4>Deciding rule</h4>
    <p id="deciding_rule" style="font-size:0.82rem"></p>
  </div>
  <div class="card">
    <h3>4. MedGemma Clinical Report</h3>
    <p id="medgemma_status_line" style="font-size:0.82rem"><em>Waiting for replay to start...</em></p>
    <div class="narrative" id="narrative"></div>
  </div>
</div>

<script>
const RISK_COLORS = {{"LOW":"#2e7d32","MEDIUM":"#f9a825","HIGH":"#ef6c00","CRITICAL":"#c62828","NOT_ASSESSABLE":"#616161"}};
// Any status other than exactly "ACCEPTED" means MedGemma's own narrative was NOT used --
// either it was never called (SKIPPED_*) or its output was discarded and a deterministic
// template took over (UNAVAILABLE_FALLBACK / REJECTED_FALLBACK:*). All of those are
// legitimate, expected outcomes (see agent_bridge.render_narrative), but the UI must label
// them as fallback so "it never renders" can't be mistaken for a live narrative.
function isFallbackStatus(status) {{
  return status !== "ACCEPTED";
}}
// Mirrors agent_bridge.py's _deciding_rule_sentence(): the catch-all "nothing
// fired" rule has no real measured_value/threshold (both null) -- print it
// plainly instead of "measured null vs threshold null", and never call it
// "FIRED" (that would imply a real threshold was crossed).
function decidingRuleSentence(dr) {{
  if (dr.measured_value === null && dr.threshold === null) {{
    return "No dangerous thresholds exceeded -> risk " + (dr.would_set_level || "LOW") + ".";
  }}
  const verdict = dr.fired ? "EXCEEDED" : "not exceeded";
  if (dr.measured_value !== null && typeof dr.measured_value === "object") {{
    const parts = Object.keys(dr.measured_value).map(k =>
      k + "=" + dr.measured_value[k] + " (threshold " + (dr.threshold ? dr.threshold[k] : "?") + ")");
    return dr.condition + ": " + parts.join(", ") + " -> " + verdict + ".";
  }}
  return dr.condition + ": measured " + dr.measured_value + " vs threshold " + dr.threshold + " -> " + verdict + ".";
}}
const runId = {run_id_js};
let stopped = false;

function esc(s) {{ const d = document.createElement("div"); d.innerText = (s === null || s === undefined) ? "" : s; return d.innerHTML; }}

async function poll() {{
  if (stopped) return;
  try {{
    const resp = await fetch("/api/state/" + runId);
    if (!resp.ok) {{ throw new Error("server returned HTTP " + resp.status); }}
    const s = await resp.json();

    document.getElementById("label").innerText = s.label || "Replay";
    document.getElementById("segment_id").innerText = s.segment_id || "";
    document.getElementById("elapsed").innerText = s.elapsed_s ?? 0;
    document.getElementById("total").innerText = s.total_s ?? 0;
    document.getElementById("chunk_i").innerText = s.chunk_i ?? 0;
    document.getElementById("n_chunks").innerText = s.n_chunks ?? "?";

    const risk = s.risk_level || "?";
    const pill = document.getElementById("risk_pill");
    pill.innerText = risk;
    pill.style.background = RISK_COLORS[risk] || "#455a64";

    document.getElementById("synthetic-banner").innerHTML = s.synthetic
      ? '<div class="banner">SYNTHETIC &mdash; NOT A REAL PATIENT. Replayed from ecg_pipeline/synthetic_ecg.py for demo/testing only.</div>' : '';
    document.getElementById("not-assessable-banner").innerHTML = (s.assessable === false)
      ? '<div class="warn">INSUFFICIENT SIGNAL &mdash; NOT ASSESSABLE. ' + esc(s.not_assessable_reason) + '</div>' : '';

    if (s.plot_png_b64) {{ document.getElementById("plot").src = "data:image/png;base64," + s.plot_png_b64; }}

    const tl = document.getElementById("timeline");
    tl.innerHTML = "";
    (s.risk_history || []).forEach(h => {{
      const el = document.createElement("span");
      el.style.background = RISK_COLORS[h.level] || "#455a64";
      el.innerText = h.t_s + "s:" + h.level;
      tl.appendChild(el);
    }});

    const beatBody = document.getElementById("beat_table");
    beatBody.innerHTML = "";
    Object.entries(s.beat_summary || {{}}).forEach(([cls, info]) => {{
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + esc(cls) + "</td><td>" + info.count + "</td><td>" + info.pct_of_analyzed_beats +
        "%</td><td>" + esc(info.confidence) + (info.confidence === "LOW" ? " (LOW CONF)" : "") + "</td>";
      beatBody.appendChild(tr);
    }});

    const rhythmList = document.getElementById("rhythm_list");
    rhythmList.innerHTML = "";
    if ((s.rhythm_findings || []).length === 0) {{
      rhythmList.innerHTML = "<li>(none so far)</li>";
    }} else {{
      s.rhythm_findings.forEach(f => {{
        const li = document.createElement("li");
        li.innerText = f.kind + ": beats " + f.start_beat_idx + "-" + f.end_beat_idx + " -- " + (f.evidence_text || "");
        rhythmList.appendChild(li);
      }});
    }}

    if (s.deciding_rule) {{
      document.getElementById("deciding_rule").innerText = decidingRuleSentence(s.deciding_rule);
    }}

    if (s.status === "done" && s.final_report) {{
      const mgStatus = s.final_report.medgemma.status;
      const fallback = isFallbackStatus(mgStatus);
      let line = "<b>MedGemma status:</b> " + esc(mgStatus) +
        " &nbsp; <b>Final risk level:</b> " + esc(s.final_report.final_risk_level);
      if (fallback) {{
        line += '<br/><span class="fallback-note">FALLBACK -- MedGemma unavailable/bypassed/rejected for this ' +
                'result; showing the deterministic rule-based report instead of a live LLM narrative.</span>';
      }}
      document.getElementById("medgemma_status_line").innerHTML = line;
      document.getElementById("narrative").innerText = s.final_report.narrative || "(no narrative)";
      stopped = true;
      return;
    }}
    if (s.status === "error") {{
      document.getElementById("medgemma_status_line").innerHTML = "<b style='color:#c62828'>Replay error: " + esc(s.error) + "</b>";
      stopped = true;
      return;
    }}

    // Still running -- make it explicit this is active progress, not stuck.
    document.getElementById("medgemma_status_line").innerHTML =
      "<em>Replay in progress: chunk " + (s.chunk_i ?? 0) + "/" + (s.n_chunks ?? "?") + " (" +
      (s.elapsed_s ?? 0) + "s / " + (s.total_s ?? 0) + "s). MedGemma narrative (or its fallback) " +
      "renders here once the stream completes.</em>";
  }} catch (err) {{
    const box = document.getElementById("js-error");
    box.style.display = "block";
    box.innerText = "UI polling error (server may be down or restarting): " + err;
  }}

  setTimeout(poll, 1000);
}}
poll();
</script>
</body></html>
"""


def _build_flask_app():
    from flask import Flask, jsonify, redirect, request, url_for

    app = Flask(__name__)

    @app.errorhandler(Exception)
    def _handle_any_error(e):
        # Defense-in-depth: an unhandled exception in a route should never
        # surface as a bare closed connection (ERR_EMPTY_RESPONSE) -- always
        # return a real page with the traceback, so a real bug is visible
        # instead of silently swallowed.
        tb = traceback.format_exc()
        print(f"[report_ui] unhandled exception:\n{tb}", flush=True)
        return (
            "<h2>Replay UI error</h2><pre>" + tb.replace("&", "&amp;").replace("<", "&lt;") + "</pre>",
            500,
        )

    @app.route("/")
    def landing():
        files = discover_vitalpatch_files(DATA_RAW / "vitalpatch")[:MAX_RECORDED_FILES_LISTED]
        file_options = "".join(
            f'<option value="{i}">{p.parent.name}/{p.name}</option>' for i, p in enumerate(files)
        ) or "<option disabled>(no recorded VitalPatch files found under data/raw/vitalpatch)</option>"
        scenario_options = "".join(f'<option value="{s}">{s}</option>' for s in SCENARIOS)
        return LANDING_TEMPLATE.format(scenario_options=scenario_options,
                                        file_options=file_options, max_files=MAX_RECORDED_FILES_LISTED)

    @app.route("/start", methods=["POST"])
    def start():
        mode = request.form.get("mode")
        speed = float(request.form.get("speed", 5.0))
        chunk_seconds = 3.0
        classifier = _server_classifier()

        ground_truth = None
        if mode == "synthetic":
            scenario = request.form.get("scenario")
            recording, ground_truth = generate_scenario(scenario)
            label = f"REPLAY (synthetic: {scenario})"
            synthetic = True
        else:
            # file_path (an absolute path under data/raw/vitalpatch) lets a caller pick ANY
            # recorded file, not just the first MAX_RECORDED_FILES_LISTED shown in the
            # landing page's dropdown -- the cap is a UI-cleanliness choice, not a hard
            # limit on which real files this tool can replay.
            explicit_path = request.form.get("file_path")
            if explicit_path:
                path = Path(explicit_path)
            else:
                files = discover_vitalpatch_files(DATA_RAW / "vitalpatch")[:MAX_RECORDED_FILES_LISTED]
                idx = int(request.form.get("file_idx", 0))
                path = files[idx]
            recordings = _load_real_recordings("vitalpatch", path)
            recording = recordings[int(request.form.get("seg_idx", 0))] if len(recordings) > 1 else recordings[0]
            label = "REPLAY (recorded: vitalpatch)"
            synthetic = False

        run_id = uuid.uuid4().hex[:10]
        state = {
            "status": "running", "label": label, "synthetic": synthetic,
            "source": recording.source, "patient_id": recording.patient_id,
            "segment_id": recording.segment_id, "ground_truth": ground_truth,
            "chunk_i": 0, "n_chunks": 1, "elapsed_s": 0.0, "total_s": 0.0,
            "risk_level": "LOW", "risk_history": [], "assessable": True, "not_assessable_reason": None,
            "rhythm_findings": [], "beat_summary": {}, "deciding_rule": {},
            "n_beats_detected": 0, "n_beats_analyzed": 0,
            "plot_png_b64": None, "final_report": None, "error": None,
        }
        with _RUNS_LOCK:
            _RUNS[run_id] = state

        thread = threading.Thread(target=_replay_worker, args=(run_id, recording, classifier, chunk_seconds, speed),
                                   daemon=True)
        thread.start()
        return redirect(url_for("replay_page", run_id=run_id))

    @app.route("/replay/<run_id>")
    def replay_page(run_id):
        if run_id not in _RUNS:
            return "Unknown run_id -- start a new replay from the landing page.", 404
        return REPLAY_TEMPLATE.format(run_id=run_id, run_id_js=json.dumps(run_id))

    @app.route("/api/state/<run_id>")
    def api_state(run_id):
        with _RUNS_LOCK:
            state = _RUNS.get(run_id)
            if state is None:
                return jsonify({"error": "unknown run_id"}), 404
            return jsonify(_public_state(state))

    return app


def serve(port: int = DEFAULT_PORT) -> None:
    app = _build_flask_app()
    url = f"http://127.0.0.1:{port}"
    print(f"ECG Pipeline Replay Demo serving at: {url}")
    print("(read-only over the existing pipeline -- no classifier/threshold/AAMI changes)")
    print("If you're on a remote machine over SSH, tunnel this port to view it in your local "
          f"browser: ssh -L {port}:localhost:{port} <user>@<this-host>")
    app.run(host="127.0.0.1", port=port, threaded=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="[legacy] path to a saved report .json -> static HTML page")
    parser.add_argument("--list", action="store_true", help="[legacy] list available saved reports and exit")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--open", action="store_true", help="[legacy mode only] open the generated page in a browser")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port for the live replay server")
    args = parser.parse_args(argv)

    if args.list:
        rows = _list_available()
        print(f"{'source':<12} {'patient_id':<14} {'segment_id':<48} {'risk':<15}")
        for source, pid, seg, risk in rows:
            print(f"{source:<12} {pid:<14} {seg:<48} {risk or '':<15}")
        print(f"\n({len(rows)} shown) Generate a static page with:")
        print("  python -m ecg_pipeline.report_ui --report <path-to-report.json> [--open]")
        return

    if args.report:
        out_path = build_page(args.report, args.out_dir)
        print(f"Generated -> {out_path}")
        print(f"Open directly in a browser: file://{out_path.resolve()}")
        if args.open:
            webbrowser.open(f"file://{out_path.resolve()}")
        return

    # Default: start the live replay server.
    serve(port=args.port)


if __name__ == "__main__":
    main()
