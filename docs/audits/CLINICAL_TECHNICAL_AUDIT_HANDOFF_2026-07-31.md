# Clinical & Technical Audit Handoff — 2026-07-31

**Written:** 2026-07-31
**Branch:** `mod_1`
**Scope of this pass:** code-level investigation and root-cause analysis only.
**No fixes were implemented in this pass** — per instruction, this is the
"understand and document" phase; fixes are proposed below for a teammate (or a
follow-up session) to implement and then validate.

This document assumes you know Python and can read `ecg_pipeline_core.py`, but
nothing else about this investigation. Every claim below is either (a) read
directly from the code with a file:line citation, or (b) a documented
measurement already in `docs/audits/AUDIT_2026-07-31.md` from the prior session's
R-peak fix, or (c) explicitly labeled as a **reasoned hypothesis** (mechanism
explained, not yet confirmed against a rendered plot — no PDF/PNG reports
existed on this machine at audit time to eyeball directly; `data/` and
generated reports are gitignored and none were present locally). Where
something is a hypothesis, it says so, and says what would confirm it.

---

## 0. What was already fixed before this pass (don't re-diagnose these)

Two prior sessions already fixed and documented:

1. **R-peak over-detection from a disabled T-wave check** — `detect_r_peaks()`
   now sets `XQRS.Conf(t_inspect_period=0.36)`, enabling XQRS's own T-wave
   rejection. CRITICAL rate corrected 290/2,889 (10.0%) → 15/2,889 (0.5%). See
   `docs/audits/AUDIT_2026-07-31.md` §7.
2. **SDNN 0.0-sentinel bug** and **waveform-panel Kalman distortion** — see
   `docs/audits/AUDIT_2026-07-31.md` §8. The waveform-panel fix is directly relevant
   to Problem 2 below — read that section first.

This pass goes further: full preprocessing-chain review (Problem 1), a second,
previously-undocumented R-peak issue that the §8.2 fix likely exposed (Problem
2), the flat-signal-section root cause (Problem 3), and one bug in the
training path not covered by either prior session.

---

## Problem 1 — Preprocessing may be over-attenuating morphology

**Chain** (`apply_filter_chain`, `ecg_pipeline_core.py:906-924`, applied in
this exact order to VitalPatch since it isn't pre-filtered):

```
remove_baseline_median (median filter x2, 200ms window)
  -> highpass_residual (2nd-order Butterworth, 0.5Hz)
  -> powerline_notch (50Hz notch)
  -> bandpass (4th-order Butterworth, 0.5-40Hz)
  -> emg_suppress_kalman (adaptive scalar Kalman)  [skipped for report display only, see §8.2]
```

### Finding 1.1 — Double median filter uses the SAME window twice, not the standard two-stage scheme

`remove_baseline_median()` (`ecg_pipeline_core.py:817-828`):

```python
stage1 = medfilt(x, kernel_size=win_samples)
stage2 = medfilt(stage1, kernel_size=win_samples)
return x - stage2
```

Both passes use the single `FilterChainConfig.median_baseline_window_ms = 200.0`
(`ecg_pipeline_core.py:117`). The classical two-median-filter baseline removal
technique this is clearly modeled on (Sörnmo & Laguna; also the scheme behind
many Pan-Tompkins-family baseline removers) uses **two different window
widths** — typically ~200ms (wide enough to treat the QRS complex, ~80-120ms,
as an outlier and exclude it from the baseline estimate) followed by ~600ms
(wide enough to also exclude the T-wave, ~160-300ms, from the estimate). Using
200ms for **both** passes means the second pass's baseline estimate is not
guaranteed to exclude T-wave energy — a T-wave close to or wider than 200ms can
leak into `stage2`'s median estimate and then get *subtracted out* along with
real baseline drift.

**Root cause, evidence-based:** this is a plausible, mechanistically clear
explanation for "T-waves less visible" / "some wave amplitudes appear
reduced" in Problem 1. **Not yet confirmed on a real waveform** — confirming
it requires plotting `x` vs. `remove_baseline_median(x, fs)` alone (isolated
from the rest of the chain) on a real VitalPatch segment with a clearly visible
T-wave and checking whether T-wave amplitude drops after this one step.

**Fix direction:** give the two passes independent widths, e.g.
`median_baseline_window_ms=200.0` then a second `median_baseline_window_2_ms=600.0`
parameter, matching the literature scheme. This is a config/signature change,
not a rewrite.

### Finding 1.2 — The chain applies a 0.5Hz high-pass TWICE, independently

`highpass_residual()` (0.5Hz, 2nd-order Butterworth, zero-phase via
`filtfilt`) runs, and then `bandpass()`'s own low edge is *also* 0.5Hz
(`FilterChainConfig.bandpass_low_hz = 0.5`, `ecg_pipeline_core.py:121`,
4th-order). These are two **independently-designed filters with the same
cutoff**, run back-to-back on the same signal
(`ecg_pipeline_core.py:917-920`). Their magnitude responses multiply: at any
frequency near but above 0.5Hz (where slow ST-segment/T-wave shape components
live), the combined attenuation is the *product* of both filters' rolloff at
that frequency — meaningfully steeper and more aggressive than either filter
alone was designed for, and steeper than either config value alone suggests to
someone reading `FilterChainConfig`.

**Root cause, evidence-based:** stacking two same-cutoff high-pass filters is
very likely unintentional — `highpass_residual`'s own docstring says it
exists to "catch drift the median filter misses," which is a reasonable
purpose, but nothing in the code or comments indicates the *bandpass* step's
low edge was intended to reinforce it rather than stand alone. Combined with
Finding 1.1, this stacks to a total of **three** low-frequency-attenuating
operations (median x2, highpass, bandpass-low) before the signal reaches
detection or display — a plausible aggregate explanation for "flattened
regions" and "reduced amplitude" beyond what a single-pass standard ECG
bandpass (typically just 0.5-40Hz once, per AHA monitoring-lead convention)
would do.

**Fix direction:** either drop `highpass_residual` as a separate step (let the
median filter + bandpass's own 0.5Hz edge do the job), or lower
`bandpass_low_hz` (e.g. to 0.05-0.1Hz, closer to diagnostic-grade guidance) so
the two stages aren't fighting over the exact same cutoff. Needs a real A/B
comparison on a segment with visible P/T waves before picking one — this is a
tuning decision that trades noise suppression for morphology fidelity, which
is exactly the kind of change Problem 4 asks to justify per-change.

### Finding 1.3 — `emg_suppress_kalman` is a heavy, causal (non-zero-phase) smoother

`emg_suppress_kalman()` (`ecg_pipeline_core.py:858-894`) is a **causal**
(forward-only, sample-by-sample) adaptive Kalman filter — unlike every other
stage in the chain, which uses `filtfilt` (zero-phase, no time delay). Its own
docstring (already updated in the prior session) documents a measured ~12x
peak-to-peak amplitude reduction on a real segment. This step is **still
applied** for beat classification input (by design — see
`docs/audits/AUDIT_2026-07-31.md` §8.2, unchanged deliberately so the classifier's
input distribution matches training) but is now **skipped for report display**.
See Problem 2, Finding 2.1 below — the fact that it's causal (introduces a
time delay on transients) is the mechanism behind a newly-relevant R-peak
timing bug, not just an amplitude one.

### Stage-by-stage verdict (Problem 1's explicit ask)

| Stage | Exists for | Appropriate for VitalPatch 125Hz? | Verdict |
|---|---|---|---|
| Median baseline removal (x2, 200ms) | Remove drift without touching QRS shape | Partially — window scheme deviates from the literature two-width design | **Modify**: use two different window widths (Finding 1.1) |
| Highpass 0.5Hz | Catch residual drift | Redundant given bandpass's own 0.5Hz edge | **Modify or remove** (Finding 1.2) |
| Notch 50Hz | Mains interference | Appropriate, standard, no issue found | Keep as-is |
| Bandpass 0.5-40Hz, order 4 | Standard monitoring-lead ECG band | Standard choice for *monitoring* (not diagnostic-grade ST work); reasonable given the SNR realities of a single-lead patch, but compounds with Finding 1.2 | Keep, but resolve the cutoff-stacking with Finding 1.2 first |
| Kalman EMG suppression | Suppress muscle noise a Butterworth alone can't | Necessary for classification (already justified, kept for training/inference input) but heavy and causal | Keep for classifier input (retraining would be required to change it — see `docs/audits/AUDIT_2026-07-31.md` §8.2); already skipped for display |
| Resampling/interpolation | N/A for VitalPatch (native rate == target 125Hz, function returns early, `ecg_pipeline_core.py:780-781`) | N/A | Not a factor for VitalPatch specifically — see Problem 3 for where interpolation *does* matter (SQI-gate gap-filling, not resampling) |

---

## Problem 2 — R-peak detection: one issue already fixed, one newly found, one residual limitation

### 2.1 — NEW: report waveform plots peak markers at a position derived from a DIFFERENT (delayed) signal than what's displayed

This is the most likely explanation for **"some detections appear after the
actual R wave"** and possibly also for markers that look slightly off the true
peak in general, and it is a direct, mechanistic consequence of the §8.2 fix
from the immediately preceding session — worth reading together.

**The chain of evidence:**

1. For VitalPatch/SeNSiO recordings, `ECGPipeline.run()` sets
   `detection_signal = filtered` (`ecg_pipeline_core.py:2626`) — i.e., R-peak
   detection and windowing both use the **Kalman-included** signal (deliberate;
   see the long comment at `ecg_pipeline_core.py:2604-2620` on why Kalman is
   needed for these sources).
2. Inside `detect_and_segment()`, `_snap_to_local_peak(signal, r_peaks, ...)`
   is called with `signal` = that same Kalman-included array
   (`ecg_pipeline_core.py:1224`, and the call site passes `filtered` as
   `signal` at `ecg_pipeline_core.py:2632`). So the **final, snapped**
   `r_peak_idx` stored on every `Beat` is the index of the local
   `argmax(|amplitude|)` **of the Kalman-smoothed curve**, not of the raw or
   bandpass-only signal.
3. `emg_suppress_kalman` is **causal** (a forward-only per-sample loop, not
   `filtfilt` — `ecg_pipeline_core.py:882-893`). A causal smoother has group
   delay: the smoothed curve's own peak on a sharp transient lags the true
   instantaneous peak, because the estimate keeps rising for several samples
   after the true peak has already passed (the filter is still "catching up").
   Given the measured steady-state gain implied by
   `process_var=1e-4`/`meas_var=1e-2` (roughly gain ≈ sqrt(process_var /
   effective meas_var) ≈ 0.1), this behaves similarly to an EMA with a time
   constant on the order of ~10 samples (~80ms @125Hz) — clinically
   significant lag, comparable to a QRS complex's own width.
4. `generate_clinician_report.py:970` now computes the **displayed** waveform
   via `recompute_filtered_signal()`, which explicitly
   **skips** Kalman (`skip_emg_suppress=True`, the §8.2 fix). Every plotting
   function (`_beat_r_xy`, `_panel_overview_filtered`, `_stacked_strip_pages`,
   `_panel_zoomed_finding` — see `generate_clinician_report.py:1017-1036`)
   reads the marker's y-value off this **Kalman-skipped** array at
   `b.r_peak_idx` (e.g. `generate_clinician_report.py:600`,
   `generate_clinician_report.py:745`).

**Net effect:** the plotted marker sits at whatever the *undelayed, more
accurate* signal's value is at an index that was located on the *delayed,
smoothed* signal. Since the delayed signal's peak occurs later in time than
the true QRS peak, the marker will visually appear **after** the true R-wave
apex in the (now correctly displayed) waveform — exactly the "some detected
peaks appear after the actual R wave" symptom in Problem 2. This is very
likely a **side effect exposed by, not caused by,** the previous session's
otherwise-correct §8.2 waveform-legibility fix: before that fix, the plot
itself was the Kalman'd signal, so the marker and the (also-delayed) visible
peak would have visually agreed; the fix corrected the signal but not the
peak-location math feeding it.

**This is NOT a plotting bug, an indexing bug, or a candidate-vs-final-peak
bug** — it plots exactly one marker per final `Beat`, correctly filtered
(`b.quality_rejected`) and correctly de-duplicated (refractory guard already
ran upstream). It is a **signal-mismatch bug**: the x/y pair drawn is
internally inconsistent about which version of the signal it's describing.

**Fix direction (needs a decision, not just a patch):** either (a) also snap
peaks to the local max of the Kalman-skipped signal for **display purposes
only** (re-snap `r_peak_idx` against `recompute_filtered_signal()`'s output
just before plotting, leaving the classifier-facing `Beat.r_peak_idx`
untouched), or (b) accept a small, explained, documented offset in the report
text. (a) is more correct and does not touch classification.

**Confirming this needs:** regenerating one real report post-fix and visually
checking whether markers now land on the true peak instead of downslope/after
it. This is the single highest-value thing to verify before trusting the rest
of this finding.

### 2.2 — Residual, already-documented limitation: near-tied T-wave/R-peak amplitude mis-snap

Already measured and written up in `docs/audits/AUDIT_2026-07-31.md` §7
("What was investigated and explicitly rejected") — patient 1844AC has 2/13
beats in a 10s window landing on a T-wave trough instead of the true R-spike,
because `_snap_to_local_peak`'s plain `argmax(abs(...))` has no way to break a
near-tied amplitude case in favor of the sharper (QRS) candidate — a
steepness-based tiebreak was tried and measured **worse** overall (83.3% vs.
90.9% correct across 5 test segments) and was reverted. **Do not re-attempt
without re-measuring against the same independent locator** used in that
prior investigation — this is a documented negative result, not an
unexplored option.

### 2.3 — Training-path bug: `training/ecg_pipeline_tools.py` uses the WRONG default `snap_radius` for real-device data

`detect_and_segment()`'s own docstring (`ecg_pipeline_core.py:1213-1220`)
explicitly warns: `snap_radius=8` (the function's default) is
**wfdb-only-validated**; on VitalPatch/SeNSiO, radius=8 causes "catastrophic
beat-level over-culling" (up to 130/133 beats rejected on one real recording).
The **main pipeline correctly overrides this**
(`ecg_pipeline_core.py:2631`, `ecg_inference/pipeline.py:149`: `snap_radius =
8 if recording.source == "wfdb" else 15`).

**But** `training/ecg_pipeline_tools.py:482`, inside
`_windows_from_recording()` — called by `collect_wide_windows()`
(`training/ecg_pipeline_tools.py:444-467`) to build the **unlabeled
pretraining corpus for the self-supervised encoder**, from **both VitalPatch
and SeNSiO files** (`training/ecg_pipeline_tools.py:449-465`) — calls:

```python
beats = detect_and_segment(filtered, TARGET_FS, BEATS)
```

with **no `snap_radius` argument**, silently taking the default `snap_radius=8`
on exactly the sources documented to fail catastrophically at that radius.

**Impact:** this does not affect the shipped classifier (`five_class_xgb.json`)
or the deployed risk pipeline — it only affects `encoder.py`'s self-supervised
pretraining corpus, if/when that encoder is actually trained and used
(`docs/archive/README.md` describes it as a supplementary, optional representation).
Still a real, reproducible bug: the encoder's real-device training windows are
silently starved by beat-level over-culling.

**Fix:** pass `snap_radius=15` explicitly (or replicate the
`recording.source`-based branch) in `_windows_from_recording()`.

---

## Problem 3 — Flat signal sections: root cause is silent SQI-gate interpolation, not a plotting or resampling bug

**Root cause, directly evidenced in code:**

`ECGPipeline.run()`, Stage 2 (`ecg_pipeline_core.py:2564-2571`):

```python
keep_mask, verdicts = run_sqi_gate(recording.signal_mv, ..., clip_value=clip_value)
signal_clean = recording.signal_mv.copy()
signal_clean[~keep_mask] = np.nan
```

Any 5-second window (`SQIThresholds.window_seconds = 5.0`,
`ecg_pipeline_core.py:99`) that fails **any** of six checks (flatline,
clipping, missing samples, QRS-impulse-character kurtosis, baseline-wander
ratio, SNR) is entirely blanked to `NaN` — all 625 samples at 125Hz, not just
the specific bad sub-region.

Then, Stage 3 (`ecg_pipeline_core.py:2584`, and identically in
`generate_clinician_report.py:263` and `training/ecg_pipeline_tools.py:479`):

```python
resampled_filled = np.interp(t_resampled, t_resampled[valid], resampled[valid])
```

`np.interp` performs **straight-line linear interpolation** between the
nearest surviving samples on either side of every NaN-blanked window. If two
or more consecutive 5-second windows fail, the gap can span 10+ seconds,
interpolated as one long straight line (or a single ramp, if the boundary
values differ). **This is the flat section**: not a resampling bug, not
"filtering," not an indexing error — it is a real, working piece of gap-filling
logic doing exactly what it's coded to do, but with a very large blast radius
per rejected window and **zero visual indication in the report**.

**Confirmed no visual indication exists:** neither `_panel_overview_filtered`
nor `_stacked_strip_pages`/`_plot_strip` (`generate_clinician_report.py:640-670`)
shades, hatches, or otherwise marks the time ranges where `keep_mask` was
`False`. The only signal a reader gets is a single aggregate number in the
panel title — `f"...(SQI window rejection rate: {rej_rate:.2%})..."`
(`generate_clinician_report.py:649`) — with no way to tell *where* in the 10s
strip the rejected/interpolated stretch is. This matches the user's report
exactly: raw signal in that time range looks normal (whatever originally
tripped the gate — e.g. a borderline kurtosis or SNR score — is often not
visually obvious to a human eye scanning the raw trace), and the "processed"
version is flat because it was never real signal to begin with past that
point — it's a straight line manufactured by `np.interp` between two distant
anchors.

**Second contributing possibility, also root-caused in code:** if the SQI gate
itself has a false-positive rejection (a genuinely fine window incorrectly
scored as e.g. `BASELINE_WANDER` or `LOW_SNR` — `evaluate_window()`,
`ecg_pipeline_core.py:604-681`), the raw signal in that window really would
look fine to a human, while the processed version is unnecessarily blanked and
interpolated. Whether this is happening on the specific segments the user
looked at is **not yet confirmed** — checking is straightforward:
cross-reference the audit log's `STAGE2_SQI_GATE.reject_codes`
(`ecg_pipeline_core.py:2567-2568`) for the segment against the visually-flat
time range.

**Fix direction (two independent, complementary fixes):**
1. **Visualize the gap.** Shade (`axvspan`) any time range where
   `keep_mask` is `False` in every waveform panel, so a flat/interpolated
   region is unmistakably distinct from real (if noisy) signal. Low-risk,
   pure display change.
2. **Investigate gate false-positive rate.** Pull `reject_code` distribution
   from a batch run and spot-check a sample of each reject code against the
   raw trace to see how many are true rejects (genuinely bad signal) vs.
   false positives (fine signal, wrong verdict). This determines whether
   Fix 1 is sufficient or whether `SQIThresholds` also need retuning.

---

## State-of-the-art comparison (Problem 2's second ask)

This project uses **WFDB's XQRS** (an open-source reimplementation of a
well-known adaptive-threshold QRS detector), not a from-scratch detector. The
existing docstring at `ecg_pipeline_core.py:932-939` already gives the stated
rationale (wearable single-lead electrodes at non-standard chest positions
produce atypical QRS morphology; XQRS's adaptive threshold is more forgiving of
that than a fixed-template method).

| Detector | Relative strength here | Verdict for this project |
|---|---|---|
| **XQRS (current)** | Adaptive threshold, built-in (if easy-to-misconfigure) T-wave rejection, well-tested on MITDB-style data. This project's real bugs were *configuration* bugs (`t_inspect_period` default, `snap_radius` default) — not an algorithmic ceiling. | Keep. The measured before/after (10.0%→0.5% CRITICAL, `docs/audits/AUDIT_2026-07-31.md` §7) shows the *fix*, not a swap, resolved the dominant known issue. |
| **Pan-Tompkins (classic)** | Simpler, well-understood, but its fixed-shape moving-window-integration assumes a QRS morphology closer to standard 12-lead placement. Wearable patches at non-standard positions (per this project's own stated rationale) are exactly the case where Pan-Tompkins' fixed assumptions are more likely to break, not less. | Not recommended as a wholesale replacement without new evidence. |
| **Hamilton** | A Pan-Tompkins variant with adaptive thresholds; similar tradeoffs to Pan-Tompkins above, slightly more robust to noise. | Same as above — not obviously better for this signal without a real A/B. |
| **NeuroKit2 (default `ecg_findpeaks`, `nk.ecg_process`)** | Wraps several methods (Pan-Tompkins-family, `neurokit`'s own, `kalidas2017`, etc.) behind a convenience API; not a fundamentally different core algorithm from what's already available in this project's stack, but does provide multiple algorithm choices behind one interface, cheap to A/B against. | Worth trying as a **second opinion / cross-check detector** on a validation subset (see below), not a wholesale swap. |
| **sleepecg (`detect_heartbeats`)** | Optimized for sleep/wearable long recordings, tends to be fast and reasonably robust to motion artifact — a reasonable comparator specifically *because* it targets wearable-style data, unlike Pan-Tompkins/Hamilton which target clinical 12-lead. | Worth trying as a second comparator for the same reason as NeuroKit2. |

**Recommendation:** the evidence in this repo does not support "XQRS is
fundamentally the wrong choice" — the two real bugs found (T-wave check
disabled by default; wrong snap radius on real-device data in one code path)
are both **configuration bugs in how XQRS was invoked**, not limitations of
XQRS itself, and both have concrete, already-partially-applied fixes. Swapping
detectors is a bigger, riskier change (new dependency, new tuning surface, no
existing validation harness result to compare against) than fixing the
remaining configuration issues (2.1, 2.3) and re-measuring. If the teammate
wants extra confidence, running NeuroKit2 or sleepecg as a **cross-check**
detector on the same validation segments (not a replacement) is a cheap way to
sanity-check XQRS's post-fix output without committing to a rewrite.

---

## Full bug list (ranked by clinical impact)

| # | Bug | File:Line | Impact | Status |
|---|---|---|---|---|
| 1 | R-peak marker plotted using an index located on the Kalman-smoothed (delayed) signal, but read against the now-displayed Kalman-skipped (undelayed) signal | `ecg_pipeline_core.py:2626,2632`, `generate_clinician_report.py:600,745,970` | Peaks visually appear late/off-peak in every VitalPatch/SeNSiO report generated since the §8.2 display fix | **NEW — not fixed** |
| 2 | SQI-gate-rejected windows are NaN-blanked and linearly interpolated with zero visual indication in reports | `ecg_pipeline_core.py:2570-2571,2584`, `generate_clinician_report.py:640-670` | Flat/straight-line sections in "processed" plots that look like data loss but are unmarked gap-fill; could also mask a false-positive-prone SQI gate | **NEW — not fixed** |
| 3 | Double median-filter baseline removal uses the same 200ms window twice instead of a 200/600ms two-stage scheme | `ecg_pipeline_core.py:817-828` | Possible T-wave energy leaking into the baseline estimate and being subtracted (morphology distortion) | **NEW — hypothesis, needs waveform confirmation** |
| 4 | `highpass_residual` (0.5Hz) and `bandpass`'s low edge (also 0.5Hz) stack, compounding low-frequency attenuation beyond either filter's individual design intent | `ecg_pipeline_core.py:831-836,849-855,917-920` | Contributes to reduced ST/T-segment amplitude fidelity | **NEW — needs A/B before fixing** |
| 5 | `training/ecg_pipeline_tools.py`'s encoder-pretraining corpus builder uses the wrong (wfdb-only-validated) `snap_radius=8` default on real VitalPatch/SeNSiO data | `training/ecg_pipeline_tools.py:482` | Silently starves the self-supervised encoder's real-device training windows via beat-level over-culling | **NEW — not fixed** |
| 6 | SDNN 0.0-sentinel bug fixed in `ecg_pipeline_core.py`/`agent_bridge.py` but not mirrored to `ecg_inference/` | `ecg_inference/features.py`, `ecg_inference/classifier.py` | Deployment path can still silently force MEDIUM risk on low-RR-count segments | Carried over from prior session, still open — see `README.md` item 12 |
| 7 | Near-tied R-peak/T-wave amplitude mis-snap on some patients (e.g. 1844AC, 2/13 beats in one window) | `ecg_pipeline_core.py:1121-1168` (`_snap_to_local_peak`) | Small, residual, patient-specific mis-detection | Documented, investigated, explicitly not re-attempted without new evidence (`docs/audits/AUDIT_2026-07-31.md` §7) |

---

## What this pass did NOT do (explicitly deferred)

- **No code changes.** Everything above is diagnosis only, per instruction.
- **No multi-dataset validation run** (MIT-BIH, SVDB, Prorhythm) with
  Sensitivity/PPV/F1 against ground-truth annotations. This is a real,
  separate undertaking (needs the datasets mounted, a scoring harness against
  `.atr` annotation files, and meaningful runtime) — recommended as the
  **next** piece of work once fixes 1-5 above are implemented, so the
  validation measures the fixed pipeline, not the currently-diagnosed one.
- **No before/after plots generated.** No PDF/PNG clinician reports existed
  locally to compare against; regenerating one real report (any VitalPatch
  segment) is the fastest way to visually confirm Findings 2.1 and 3 before
  writing any fix.

## Recommended order of work for whoever picks this up

1. **Regenerate one real clinician report** (`generate_clinician_report.py`)
   on a segment known to have a visible T-wave, to visually confirm/refute
   Findings 1.1, 1.3, 2.1, and 3 before writing fixes for any of them — cheap,
   fast, and turns four hypotheses into four measurements.
2. Fix #1 (peak-marker signal mismatch) and #2 (SQI-gap shading) — both are
   display-only changes, lowest risk, and directly address the two most
   visually alarming symptoms reported.
3. Decide Findings 1.1/1.2 (filter chain tuning) together, since they compound
   — requires an A/B comparison on real morphology, not a blind parameter
   change.
4. Fix #5 (`training/ecg_pipeline_tools.py` snap_radius) — small, isolated,
   no risk to the shipped classifier.
5. Only after 1-4: run the multi-dataset validation (MIT-BIH, SVDB, Prorhythm,
   VitalPatch) with Sensitivity/PPV/F1 against ground truth, so the numbers
   reported reflect the corrected pipeline.
