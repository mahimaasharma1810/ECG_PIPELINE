"""synthetic_ecg.py -- synthetic single-lead ECG generator for DEMO and
PIPELINE-TESTING ONLY.

============================================================================
HARD BOUNDARY -- READ BEFORE USING THIS MODULE
============================================================================
Signals produced here are for (a) demonstrating the end-to-end pipeline
on-demand and (b) testing that the pipeline's deterministic rules/guards
(VT-run detection, PVC-burden threshold, AFib-suspected flag, the SQI/
NOT_ASSESSABLE guard) react correctly to a KNOWN, injected ground truth.

This module, and everything produced by it, must NEVER be:
  - added to any training set, DS1/DS2 split, or build_dataset pipeline,
  - used to fine-tune, reweight, or retrain the beat classifier,
  - used to compute or report the beat classifier's accuracy/F1/etc.
The classifier (`five_class_xgb.json`), the risk cascade thresholds
(`RiskThresholds`/`RISK` in ecg_pipeline_core.py), and the AAMI 5-class
scheme are all imported here read-only and are never modified.

Every Recording produced by this module has `source="synthetic"` and every
saved report derived from it must carry a "SYNTHETIC -- not a real patient"
label (see demo_stream.py / test_pipeline_synthetic.py, which both do this).

============================================================================
MORPHOLOGY DISCLAIMER
============================================================================
These are RHYTHM/PATTERN demos, not a source of realistic per-class beat
morphology. Ventricular ("V") beats here are built from a parametric wide/
tall pulse chosen to trip the same width+amplitude+prematurity criteria a
real PVC trips (and that this project's classifier is comparatively good
at, DS2 held-out V F1=0.830) -- they are NOT a clinically faithful PVC
waveform, and S/F-class morphology is not attempted at all (the classifier
is weak on S/F even on real data; synthesizing fake S/F morphology would
only produce meaningless "detections" of those classes). Do not use any
per-class accuracy number derived from this generator for anything.

============================================================================
BACKEND
============================================================================
Prefers NeuroKit2 (`nk.ecg_simulate`, ECGSYN dynamical-model method) for the
underlying normal-sinus waveform if it's importable; otherwise falls back to
a minimal parametric P-QRS-T simulator defined at the bottom of this file.
`generate_scenario(...)` and each `generate_*` function report which backend
was used in the returned `GroundTruth.backend` field.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from ecg_pipeline.ecg_pipeline_core import ECGPipeline, FiveClassBeatClassifier, MODELS_DIR, RISK, Recording

try:
    import neurokit2 as nk
    _HAVE_NEUROKIT = True
except ImportError:
    _HAVE_NEUROKIT = False

SCENARIOS = ["NORMAL", "PVC_BURDEN", "VT_RUN", "AFIB_LIKE", "NOISY"]

# Device-native sample rates this module targets, per the task spec.
FS_VITALPATCH = 125.0
FS_PRORHYTHM = 100.0


@dataclass
class GroundTruth:
    """Everything that was DELIBERATELY injected into a synthetic signal,
    for the self-test suite (test_pipeline_synthetic.py) to assert the
    pipeline reacted to correctly. Purely descriptive -- never fed back
    into the pipeline itself."""
    scenario: str
    fs: float
    duration_s: float
    heart_rate_bpm: float
    backend: str                                   # "neurokit2" | "fallback_parametric"
    n_cycles: int
    ectopic_cycle_indices: list[int] = field(default_factory=list)
    vt_run_length: int | None = None
    pvc_target_pct: float | None = None
    afib_rr_cv_target: float | None = None
    noise_profile: dict | None = None
    notes: str = ""


# ============================================================================
# Parametric pulse building blocks (used by the fallback simulator AND by
# the ectopic/PVC-like beat injector regardless of backend -- there is no
# "real" ground truth for what a synthetic V beat should look like, so this
# is deliberately a hand-built, controllable shape rather than an attempt at
# per-patient-realistic morphology).
# ============================================================================

def _gauss_bump(fs: float, width_ms: float, amplitude: float) -> np.ndarray:
    n = max(3, int(round(width_ms / 1000.0 * fs)))
    t = np.linspace(-2.2, 2.2, n)
    return amplitude * np.exp(-t ** 2 / 2.0)


def _ricker_pulse(fs: float, width_ms: float, amplitude: float, polarity: float = 1.0) -> np.ndarray:
    """Mexican-hat / Ricker wavelet: sharp central spike + shoulders, a
    reasonable stand-in for a QRS impulse shape without claiming clinical
    fidelity."""
    n = max(3, int(round(width_ms / 1000.0 * fs)))
    t = np.linspace(-3.0, 3.0, n)
    pulse = (1 - t ** 2) * np.exp(-t ** 2 / 2.0)
    pulse = pulse / (np.max(np.abs(pulse)) + 1e-12)
    return polarity * amplitude * pulse


def _add_patch(arr: np.ndarray, patch: np.ndarray, start: int) -> None:
    """In-place add `patch` into `arr` at `start`, clipped to bounds."""
    end = start + len(patch)
    a0, a1 = max(0, start), min(len(arr), end)
    if a1 <= a0:
        return
    p0, p1 = a0 - start, a1 - start
    arr[a0:a1] += patch[p0:p1]


def _make_ectopic_cycle(fs: float, template_len: int, r_idx_in_template: int,
                         normal_amplitude: float, prematurity_frac: float = 0.7,
                         qrs_width_ms: float = 220.0, amp_mult: float = 3.5,
                         rng: np.random.Generator | None = None) -> tuple[np.ndarray, int]:
    """Builds one wide/tall/premature ("PVC-like") cycle from scratch:
    shortened cycle length (prematurity), no preceding P wave, a wide
    high-amplitude QRS, and a discordant T-wave bump. Returns
    (cycle_array, r_peak_index_within_cycle).

    The QRS itself uses a single-lobe Gaussian bump, NOT the Ricker/
    Mexican-hat wavelet used elsewhere in this module: empirically, the
    Ricker's symmetric negative side lobes were large enough (at the
    amplitude needed to read as "V" to the trained classifier) that the
    pipeline's own XQRS R-peak detector sometimes detected a side lobe as a
    second, spurious beat a few tens of ms after the real one -- which
    fragments an intended consecutive V-run with a bogus extra detection.
    A single-lobe bump has no secondary local extremum for XQRS to latch
    onto.
    """
    rng = rng or np.random.default_rng()
    new_len = max(int(fs * 0.3), int(round(template_len * prematurity_frac)))
    cycle = np.zeros(new_len)

    r_pos = max(int(new_len * 0.28), int(round(qrs_width_ms / 1000.0 * fs)))
    r_pos = min(r_pos, new_len - 1)

    qrs = _gauss_bump(fs, width_ms=qrs_width_ms, amplitude=normal_amplitude * amp_mult) * -1.0
    q_start = r_pos - len(qrs) // 2
    _add_patch(cycle, qrs, q_start)
    r_idx_in_cycle = q_start + int(np.argmax(np.abs(qrs)))
    r_idx_in_cycle = max(0, min(new_len - 1, r_idx_in_cycle))

    # Discordant T wave (opposite polarity from the QRS, per classic PVC morphology).
    t_bump = _gauss_bump(fs, width_ms=220.0, amplitude=normal_amplitude * amp_mult * 0.4)
    t_start = q_start + len(qrs) + int(0.04 * fs)
    _add_patch(cycle, t_bump, t_start)

    cycle += rng.normal(0, 0.01 * normal_amplitude, size=new_len)
    return cycle, r_idx_in_cycle


# ============================================================================
# Baseline (normal-sinus) generation -- NeuroKit2 preferred, else fallback.
# ============================================================================

def _cycle_split(signal: np.ndarray, r_peaks: np.ndarray) -> list[tuple[int, int, int]]:
    """Split a signal into one segment per cardiac cycle, boundary = the
    midpoint between consecutive R-peaks. Returns (start, end, r_idx_local)
    per cycle, dropping the (partial) lead-in before the first peak and
    lead-out after the last so every returned cycle contains exactly one
    R-peak, which is required by the injector/concatenation logic below."""
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 3:
        return []
    bounds = [(r_peaks[i] + r_peaks[i + 1]) // 2 for i in range(len(r_peaks) - 1)]
    cycles = []
    for i in range(1, len(bounds)):
        start, end = bounds[i - 1], bounds[i]
        if end <= start:
            continue
        cycles.append((start, end, int(r_peaks[i] - start)))
    return cycles


def _generate_baseline_neurokit(duration_s: float, fs: float, heart_rate_bpm: float,
                                 heart_rate_std: float, seed: int | None) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Returns (full_signal, list_of_cycle_arrays, list_of_r_idx_within_cycle)."""
    pad_s = duration_s * 1.3 + 5.0  # generate extra so cycle-splitting never runs off the end
    sig = np.asarray(nk.ecg_simulate(duration=pad_s, sampling_rate=fs, heart_rate=heart_rate_bpm,
                                      heart_rate_std=heart_rate_std, method="ecgsyn",
                                      random_state=seed), dtype=float)
    _, info = nk.ecg_peaks(sig, sampling_rate=fs)
    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    cycles = _cycle_split(sig, r_peaks)
    target_n_cycles = max(3, int(duration_s / (60.0 / heart_rate_bpm)))
    cycles = cycles[:max(target_n_cycles, 3)]
    if len(cycles) < 3:
        raise RuntimeError("neurokit2 produced too few cycles for the requested duration/heart rate")
    cycle_arrays = [sig[s:e].copy() for s, e, _ in cycles]
    r_idxs = [r for _, _, r in cycles]
    return sig, cycle_arrays, r_idxs


def _generate_baseline_fallback(duration_s: float, fs: float, heart_rate_bpm: float,
                                 heart_rate_std: float, seed: int | None) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Minimal parametric P-QRS-T simulator used only if neurokit2 is not
    importable. NOT physiologically calibrated (no ECGSYN dynamical model)
    -- a plain per-beat P/QRS/T Gaussian/Ricker template train with
    Gaussian-jittered RR intervals, sufficient for rhythm/timing demos
    only."""
    rng = np.random.default_rng(seed)
    mean_rr_s = 60.0 / heart_rate_bpm
    std_rr_s = mean_rr_s * (heart_rate_std / max(heart_rate_bpm, 1.0))
    n_cycles = int(duration_s / mean_rr_s) + 4
    cycle_arrays, r_idxs = [], []
    for _ in range(n_cycles):
        rr_s = max(0.3, rng.normal(mean_rr_s, std_rr_s))
        n = int(round(rr_s * fs))
        cycle = np.zeros(n)
        r_pos = int(n * 0.35)
        p = _gauss_bump(fs, width_ms=80, amplitude=0.15)
        _add_patch(cycle, p, r_pos - int(0.18 * fs) - len(p) // 2)
        qrs = _ricker_pulse(fs, width_ms=90, amplitude=1.0, polarity=1.0)
        _add_patch(cycle, qrs, r_pos - len(qrs) // 2)
        t = _gauss_bump(fs, width_ms=180, amplitude=0.3)
        _add_patch(cycle, t, r_pos + int(0.16 * fs))
        cycle += rng.normal(0, 0.01, size=n)
        cycle_arrays.append(cycle)
        r_idxs.append(r_pos)
    sig = np.concatenate(cycle_arrays)
    return sig, cycle_arrays, r_idxs


def _generate_baseline(duration_s: float, fs: float, heart_rate_bpm: float,
                        heart_rate_std: float, seed: int | None) -> tuple[np.ndarray, list[np.ndarray], list[int], str]:
    if _HAVE_NEUROKIT:
        sig, cycles, r_idxs = _generate_baseline_neurokit(duration_s, fs, heart_rate_bpm, heart_rate_std, seed)
        return sig, cycles, r_idxs, "neurokit2"
    sig, cycles, r_idxs = _generate_baseline_fallback(duration_s, fs, heart_rate_bpm, heart_rate_std, seed)
    return sig, cycles, r_idxs, "fallback_parametric"


def _to_recording(signal: np.ndarray, fs: float, scenario: str, seed: int | None) -> Recording:
    n = len(signal)
    timestamps_ms = np.arange(n, dtype=np.int64) * int(round(1000.0 / fs))
    tag = f"{scenario}_{seed if seed is not None else 'x'}_{uuid.uuid4().hex[:8]}"
    return Recording(
        signal_mv=signal.astype(np.float64),
        timestamps_ms=timestamps_ms,
        fs_nominal=fs,
        source="synthetic",
        patient_id="SYNTHETIC",
        segment_id=f"synthetic_{tag}",
        meta={"synthetic": True, "scenario": scenario, "generator": "ecg_pipeline.synthetic_ecg"},
    )


# ============================================================================
# Public scenario generators
# ============================================================================

def generate_normal(duration_s: float = 60.0, fs: float = FS_VITALPATCH,
                     heart_rate_bpm: float = 72.0, seed: int | None = 1) -> tuple[Recording, GroundTruth]:
    """Clean sinus rhythm, ~60-90 bpm, no injected ectopy or artifacts."""
    sig, cycles, r_idxs, backend = _generate_baseline(duration_s, fs, heart_rate_bpm, heart_rate_std=3.0, seed=seed)
    signal = np.concatenate(cycles)
    recording = _to_recording(signal, fs, "NORMAL", seed)
    gt = GroundTruth(scenario="NORMAL", fs=fs, duration_s=len(signal) / fs, heart_rate_bpm=heart_rate_bpm,
                      backend=backend, n_cycles=len(cycles),
                      notes="Clean sinus rhythm, no injected ectopy/artifacts. Expect risk_level LOW.")
    return recording, gt


_CLASSIFIER_CACHE: FiveClassBeatClassifier | None = None


def _default_classifier() -> FiveClassBeatClassifier:
    """Loads the frozen production classifier read-only, purely so the
    ectopic-beat injector below can VERIFY (never train/fine-tune/save)
    that an injected beat actually reads as "V" to it before a scenario is
    finalized -- the exact same model file and load path
    agent_bridge.load_classifier() uses for real recordings."""
    global _CLASSIFIER_CACHE
    if _CLASSIFIER_CACHE is None:
        clf = FiveClassBeatClassifier()
        clf.load(MODELS_DIR / "five_class_xgb.json")
        _CLASSIFIER_CACHE = clf
    return _CLASSIFIER_CACHE


# (amp_mult, qrs_width_ms, prematurity_frac) configs tried in order for
# ISOLATED ectopic beats (real N beats as neighbours on both sides --
# PVC_BURDEN's spread-out placement), found empirically against the live
# classifier -- see the module's MORPHOLOGY DISCLAIMER.
_ISOLATED_ECTOPIC_CONFIGS = [
    (3.5, 220.0, 0.7),
    (3.0, 220.0, 0.7),
    (4.0, 220.0, 0.7),
    (3.0, 250.0, 0.75),
    (4.5, 220.0, 0.65),
]

# Configs for a CONSECUTIVE run of ectopic beats (VT_RUN). prematurity_frac
# =1.0 (i.e. cycles are NOT shortened) is deliberate: shortening every beat
# in a back-to-back run crowds each beat's fixed-size analysis window
# against its neighbour's, which empirically confuses the RR-timing
# features enough that interior run beats stop reading as V even though
# the identical amplitude/width reads as V reliably in PVC_BURDEN's
# isolated placement. Keeping cycles full-length avoids that crowding; the
# V-run here is signalled by QRS morphology alone, not by an unrealistic
# run rate -- consistent with this module's rhythm/pattern-only scope.
_RUN_ECTOPIC_CONFIGS = [
    (3.0, 220.0, 1.0),
    (3.5, 220.0, 1.0),
    (4.0, 220.0, 1.0),
    (3.0, 250.0, 1.0),
    (2.5, 220.0, 1.0),
]


def _build_injected_signal(cycles: list[np.ndarray], r_idxs: list[int], fs: float,
                            ectopic_indices: list[int], seed: int | None, amp_mult: float,
                            qrs_width_ms: float, prematurity_frac: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    normal_amp = float(np.mean([np.ptp(c) for c in cycles])) or 1.0
    out_cycles = list(cycles)
    for idx in ectopic_indices:
        ectopic_cycle, _ = _make_ectopic_cycle(fs, len(cycles[idx]), r_idxs[idx], normal_amp,
                                                prematurity_frac=prematurity_frac, qrs_width_ms=qrs_width_ms,
                                                amp_mult=amp_mult, rng=rng)
        out_cycles[idx] = ectopic_cycle
    return np.concatenate(out_cycles)


def _verify_readonly(signal: np.ndarray, fs: float, seed: int | None, classifier: FiveClassBeatClassifier):
    """Runs the EXISTING, frozen ECGPipeline read-only, purely to check
    whether a candidate injected signal actually trips the intended
    finding before a scenario is finalized -- the same ECGPipeline.run()
    call every other caller in this project makes. Never fits, saves, or
    otherwise modifies the classifier, any RiskThresholds value, or the
    AAMI scheme."""
    recording = _to_recording(signal, fs, "_TUNE", seed)
    pipeline = ECGPipeline(classifier=classifier)
    return pipeline.run(recording)


def generate_pvc_burden(duration_s: float = 90.0, fs: float = FS_VITALPATCH, pvc_pct: float = 15.0,
                         heart_rate_bpm: float = 72.0, seed: int | None = 2,
                         classifier: FiveClassBeatClassifier | None = None,
                         verify: bool = True) -> tuple[Recording, GroundTruth]:
    """Normal rhythm with a controllable percentage of ventricular (V)-like
    beats, spread out (minimum gap of 2 beats between them) so this
    scenario tests the PVC-BURDEN threshold rule specifically, without
    accidentally also forming a >=3-consecutive VT run.

    If `verify` is True (default), the injected signal is checked against
    the real, frozen production classifier (read-only -- see
    `_verify_readonly`) and the ectopic-beat morphology config that
    achieves the highest actual PVC burden is kept, so this scenario
    reliably trips the rule rather than hoping one fixed parameter set
    always works.
    """
    sig, cycles, r_idxs, backend = _generate_baseline(duration_s, fs, heart_rate_bpm, heart_rate_std=3.0, seed=seed)
    n = len(cycles)
    rng = np.random.default_rng(seed)
    target_count = max(1, int(round(n * pvc_pct / 100.0)))

    candidates = list(range(1, n - 1))
    rng.shuffle(candidates)
    chosen: list[int] = []
    for c in candidates:
        if len(chosen) >= target_count:
            break
        if any(abs(c - x) < 2 for x in chosen):
            continue
        chosen.append(c)
    chosen.sort()

    classifier = classifier or (_default_classifier() if verify else None)
    configs = _ISOLATED_ECTOPIC_CONFIGS if verify else _ISOLATED_ECTOPIC_CONFIGS[:1]
    best_signal, best_pct, best_cfg = None, -1.0, configs[0]
    for cfg in configs:
        signal = _build_injected_signal(cycles, r_idxs, fs, chosen, seed, *cfg)
        if not verify:
            best_signal, best_cfg = signal, cfg
            break
        result = _verify_readonly(signal, fs, seed, classifier)
        achieved_pct = result.risk_report.pvc_burden_pct
        if achieved_pct > best_pct:
            best_signal, best_pct, best_cfg = signal, achieved_pct, cfg
        if achieved_pct > RISK.pvc_burden_high_pct * 1.2:
            break

    recording = _to_recording(best_signal, fs, "PVC_BURDEN", seed)
    verify_note = (f"Verified against the live classifier: achieved pvc_burden_pct~={best_pct:.1f}% "
                   f"(config amp_mult={best_cfg[0]}, qrs_width_ms={best_cfg[1]}, prematurity_frac={best_cfg[2]})."
                   if verify else "NOT verified against the live classifier (verify=False was passed).")
    gt = GroundTruth(scenario="PVC_BURDEN", fs=fs, duration_s=len(best_signal) / fs, heart_rate_bpm=heart_rate_bpm,
                      backend=backend, n_cycles=n, ectopic_cycle_indices=chosen,
                      pvc_target_pct=100.0 * len(chosen) / n,
                      notes=f"Injected {len(chosen)}/{n} isolated V-like beats "
                            f"(~{100.0 * len(chosen) / n:.1f}% of beats, spread >=2 apart). "
                            f"Expect the PVC-burden rule to fire (HIGH, or CRITICAL if >critical threshold). "
                            f"{verify_note}")
    return recording, gt


def generate_vt_run(duration_s: float = 60.0, fs: float = FS_VITALPATCH, run_length: int = 5,
                     heart_rate_bpm: float = 72.0, seed: int | None = 3,
                     classifier: FiveClassBeatClassifier | None = None,
                     verify: bool = True) -> tuple[Recording, GroundTruth]:
    """A run of >=3 consecutive V-like beats in the middle of an otherwise
    normal recording, to trip the CRITICAL VT-run rule.

    If `verify` is True (default), the injected signal is checked against
    the real, frozen production classifier + rhythm engine (read-only --
    see `_verify_readonly`); configs are tried in order until one actually
    produces a VT_RUN rhythm finding, falling back to whichever config
    produced the most V-labelled beats if none does.
    """
    if run_length < 3:
        raise ValueError("run_length must be >=3 to constitute a VT run")
    sig, cycles, r_idxs, backend = _generate_baseline(duration_s, fs, heart_rate_bpm, heart_rate_std=3.0, seed=seed)
    n = len(cycles)
    if n < run_length + 6:
        raise RuntimeError("Recording too short for the requested VT run length; increase duration_s")
    start = n // 2
    ectopic_indices = list(range(start, start + run_length))

    classifier = classifier or (_default_classifier() if verify else None)
    configs = _RUN_ECTOPIC_CONFIGS if verify else _RUN_ECTOPIC_CONFIGS[:1]
    best_signal, best_v_count, best_cfg, confirmed = None, -1, configs[0], False
    for cfg in configs:
        signal = _build_injected_signal(cycles, r_idxs, fs, ectopic_indices, seed, *cfg)
        if not verify:
            best_signal, best_cfg = signal, cfg
            break
        result = _verify_readonly(signal, fs, seed, classifier)
        found = any(f.kind == "VT_RUN" for f in result.rhythm_findings)
        v_count = result.beat_labels.count("V")
        if found:
            best_signal, best_cfg, confirmed = signal, cfg, True
            break
        if v_count > best_v_count:
            best_signal, best_v_count, best_cfg = signal, v_count, cfg

    verify_note = ((f"Verified against the live classifier + rhythm engine: a VT_RUN finding was confirmed "
                    f"(config amp_mult={best_cfg[0]}, qrs_width_ms={best_cfg[1]}, prematurity_frac={best_cfg[2]})."
                    if confirmed else
                    f"NOT confirmed -- no config in this run produced a VT_RUN finding against the live "
                    f"classifier; using the best-scoring config anyway (amp_mult={best_cfg[0]}, "
                    f"qrs_width_ms={best_cfg[1]}). The self-test suite will report this as a real failure "
                    f"if it still doesn't fire.")
                   if verify else "NOT verified against the live classifier (verify=False was passed).")
    recording = _to_recording(best_signal, fs, "VT_RUN", seed)
    gt = GroundTruth(scenario="VT_RUN", fs=fs, duration_s=len(best_signal) / fs, heart_rate_bpm=heart_rate_bpm,
                      backend=backend, n_cycles=n, ectopic_cycle_indices=ectopic_indices,
                      vt_run_length=run_length,
                      notes=f"Injected {run_length} consecutive V-like beats at beat positions "
                            f"{ectopic_indices[0]}-{ectopic_indices[-1]} (of {n} total). "
                            f"Expect a VT_RUN rhythm finding and risk_level CRITICAL. {verify_note}")
    return recording, gt


def generate_afib_like(duration_s: float = 90.0, fs: float = FS_VITALPATCH, heart_rate_bpm: float = 80.0,
                        heart_rate_std: float = 30.0, seed: int | None = 4) -> tuple[Recording, GroundTruth]:
    """Irregular RR timing (high RR coefficient-of-variation), built purely
    from RR-interval variability with otherwise normal beat morphology --
    this is what the pipeline's own AFib heuristic looks at (RR CV over a
    rolling window), so this scenario is the most faithful of the five to
    what it's actually testing.

    NOTE: RR variability is injected here by per-cycle time-resampling
    (stretch/compress each whole cardiac cycle by a random factor), NOT via
    neurokit2's own `heart_rate_std` parameter -- empirically, ECGSYN's ODE
    integrator becomes pathologically slow (>60s, possibly non-terminating
    within a reasonable demo timeout) once `heart_rate_std` is pushed high
    enough to produce AFib-like variability. Generating the baseline at a
    low, fast `heart_rate_std` and then resampling each cycle keeps this
    scenario fast and deterministic while still testing exactly what the
    pipeline's AFib heuristic measures (RR-interval coefficient of
    variation), independent of beat classifier morphology.
    """
    sig, cycles, r_idxs, backend = _generate_baseline(duration_s, fs, heart_rate_bpm, heart_rate_std=1.0, seed=seed)
    rng = np.random.default_rng(seed)
    stretched_cycles = []
    for c in cycles:
        factor = float(np.clip(rng.normal(1.0, 0.35), 0.55, 1.65))
        old_len = len(c)
        new_len = max(int(fs * 0.3), int(round(old_len * factor)))
        old_idx = np.linspace(0, old_len - 1, old_len)
        new_idx = np.linspace(0, old_len - 1, new_len)
        stretched_cycles.append(np.interp(new_idx, old_idx, c))
    cycles = stretched_cycles
    signal = np.concatenate(cycles)
    rr_s = np.array([len(c) for c in cycles]) / fs
    rr_cv = float(np.std(rr_s) / np.mean(rr_s))
    recording = _to_recording(signal, fs, "AFIB_LIKE", seed)
    gt = GroundTruth(scenario="AFIB_LIKE", fs=fs, duration_s=len(signal) / fs, heart_rate_bpm=heart_rate_bpm,
                      backend=backend, n_cycles=len(cycles), afib_rr_cv_target=rr_cv,
                      notes=f"High RR-interval variability injected via heart_rate_std={heart_rate_std} "
                            f"(cycle-length RR CV ~= {rr_cv:.3f}, rule threshold is "
                            f"{RISK.afib_rr_cv_threshold:.2f}). "
                            f"Expect AFIB_SUSPECTED in rhythm_findings.")
    return recording, gt


def generate_noisy(duration_s: float = 60.0, fs: float = FS_VITALPATCH, heart_rate_bpm: float = 72.0,
                    seed: int | None = 5, noise_std_mult: float = 9.0,
                    dropout_frac: float = 0.18) -> tuple[Recording, GroundTruth]:
    """Normal signal + heavy Gaussian noise + slow baseline wander + a
    contiguous flatline dropout, meant to trip the SQI gate and the
    NOT_ASSESSABLE guard (or at minimum heavily depress quality_score)."""
    sig, cycles, r_idxs, backend = _generate_baseline(duration_s, fs, heart_rate_bpm, heart_rate_std=1.0, seed=seed)
    clean = np.concatenate(cycles)
    rng = np.random.default_rng(seed)
    clean_std = float(np.std(clean)) or 1.0

    noisy = clean.copy()
    noisy += rng.normal(0, noise_std_mult * clean_std, size=len(clean))
    t = np.arange(len(clean)) / fs
    wander_freq_hz = 0.12
    noisy += 3.0 * clean_std * np.sin(2 * np.pi * wander_freq_hz * t + rng.uniform(0, 2 * np.pi))

    n = len(noisy)
    drop_len = int(round(dropout_frac * n))
    drop_start = None
    if drop_len > 0:
        drop_start = int(rng.integers(0, max(1, n - drop_len)))
        noisy[drop_start:drop_start + drop_len] = noisy[drop_start] + rng.normal(0, 1e-6, drop_len)

    recording = _to_recording(noisy, fs, "NOISY", seed)
    gt = GroundTruth(scenario="NOISY", fs=fs, duration_s=n / fs, heart_rate_bpm=heart_rate_bpm,
                      backend=backend, n_cycles=len(cycles),
                      noise_profile={"noise_std_mult": noise_std_mult, "wander_freq_hz": wander_freq_hz,
                                     "dropout_frac": dropout_frac, "dropout_start_sample": drop_start,
                                     "dropout_len_samples": drop_len},
                      notes=f"Heavy Gaussian noise ({noise_std_mult}x clean std) + baseline wander + a "
                            f"{dropout_frac * 100:.0f}%-of-duration flatline dropout. Expect either "
                            f"NOT_ASSESSABLE or a heavily quality-degraded result -- never a clean LOW.")
    return recording, gt


_GENERATORS = {
    "NORMAL": generate_normal,
    "PVC_BURDEN": generate_pvc_burden,
    "VT_RUN": generate_vt_run,
    "AFIB_LIKE": generate_afib_like,
    "NOISY": generate_noisy,
}


def generate_scenario(scenario: str, **kwargs) -> tuple[Recording, GroundTruth]:
    if scenario not in _GENERATORS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    return _GENERATORS[scenario](**kwargs)


def backend_in_use() -> str:
    return "neurokit2" if _HAVE_NEUROKIT else "fallback_parametric"


if __name__ == "__main__":
    print(f"synthetic_ecg backend: {backend_in_use()}")
    for scenario in SCENARIOS:
        recording, gt = generate_scenario(scenario)
        print(f"\n=== {scenario} ===")
        print(f"  fs={gt.fs} duration_s={gt.duration_s:.1f} n_samples={len(recording.signal_mv)} "
              f"backend={gt.backend}")
        print(f"  {gt.notes}")
