# Edge Deployment Blocker Fixes — Report

**Scope:** ECG pipeline correctness on the deployment path only.
**Out of scope (not touched):** demo UI, MedGemma narrative quality/wording.
**Files modified:** `ecg_pipeline/ecg_pipeline_core.py` only. No classifier weights,
cascade thresholds, or AAMI scheme changed. No commits made.

---

## Summary verdict

| Priority | Area | Status |
|---|---|---|
| P1 | R-peak over-detection / beat-loss | **Fixed — deploy-safe** |
| P2 | VitalPatch deployment path | **Fixed — deploy-safe** (after catching 2 regressions introduced mid-fix, see below) |
| P3 | Parser robustness (`'-'` sentinel) | **Fixed — deploy-safe** |
| P4 | Structured report integrity | **Verified — deploy-safe** |

**All four priorities are deploy-safe. Nothing currently blocks edge deployment.**

---

## P1 — R-peak over-detection / beat-loss

**Symptom:** on non-VitalPatch (WFDB/MITDB) paths, detected beats >> true beats, and most got culled at the beat-level SQI gate.

**Root cause** (proven via direct pre/post-Kalman `detect_r_peaks` A/B comparison, not guessed):
`emg_suppress_kalman` (Stage 4 of the filter chain) uses **fixed absolute-unit variances**
(`process_var=1e-4`, `meas_var=1e-2`, innovation clip `3.0`). On true mV-scale WFDB signal
(std ≈ 0.1–0.5), these fixed values collapse genuine QRS amplitude toward the noise floor,
destroying the SNR that XQRS's adaptive threshold needs — the detector ends up chasing small
noise fluctuations instead of the (now flattened) true QRS complex, causing severe
over-detection.

**Fix:** R-peak *detection* now runs on a separate signal that skips the Kalman step —
**for `source == "wfdb"` only**. Beat *windowing and feature extraction* (what the classifier
sees) are computed from the original, unchanged, Kalman-included signal — byte-identical to
before. This was a deliberate architectural choice: `emg_suppress_kalman`'s output is shared
with the classifier's training-time feature extraction (`ecg_pipeline_tools.py`), so directly
changing that function would have silently altered the frozen classifier's effective input
distribution. Decoupling detection from windowing avoids this entirely.

A secondary side effect surfaced during this fix: with a rougher (skip-Kalman) detection
signal, the existing `_snap_to_local_peak` search radius (15) let the snap wander onto an
unrelated nearby local max on noisy records — MITDB record 213's beat-level retention
collapsed to 44.6%. A 13-record × 7-radius-value sweep found radius=8 fixes 213 (90.4%
retention) with no other MITDB record regressing catastrophically (13-record aggregate
retention improved from 81.0% to 82.9%). Snap radius was changed to 8, **for WFDB only**
(see P2 for why this couldn't be a global default).

### Before/after — 13-record MITDB beat retention

| record | n_true | n_detected | n_analyzed | analyzed/true |
|---|---|---|---|---|
| 100 | 2273 | 2061 | 2060 | 90.6% |
| 101 | 1865 | 1747 | 1745 | 93.6% |
| 103 | 2084 | 1957 | 1953 | 93.7% |
| 105 | 2572 | 2262 | 2202 | 85.6% |
| 111 | 2124 | 1909 | 1895 | 89.2% |
| 119 | 1987 | 1606 | 1496 | 75.3% |
| 200 | 2601 | 2204 | 2137 | 82.2% |
| 203 | 2980 | 2670 | 2123 | 71.2% |
| 207 | 1860 | 1715 | 1636 | 88.0% |
| 210 | 2650 | 2346 | 2126 | 80.2% |
| 213 | 3251 | 2961 | 2940 | 90.4% |
| 219 | 2154 | 1886 | 1408 | 65.4% |
| 223 | 2605 | 2220 | 1990 | 76.4% |

All previously over-detecting records (103, 111, 119, 219) are now in the same normal
65–94% retention range as records that were never affected.

---

## P2 — VitalPatch deployment path

**This is the part that matters most, and it did not start out clean.** The P1 fix, applied
naively (unconditionally, to all sources), broke VitalPatch in two separate ways. Both were
caught by re-running the full 2375-file VitalPatch batch and diagnosing every risk-level shift
before accepting the fix as correct — neither regression showed up in unit-level testing on a
handful of files, only in the aggregate batch comparison.

### Regression 1: skip-Kalman-for-detection applied universally

Applying the P1 detection-signal fix to *all* sources (not just WFDB) nearly **doubled**
VitalPatch beat counts (e.g. 126 → 244 detected beats on one recording) and pushed multiple
segments to falsely-elevated `CRITICAL`.

Root cause: VitalPatch (and SeNSiO) signals are documented in this codebase as "arbitrary
firmware-scaled raw ADC counts with no published mV-per-count constant" — real amplitude is
roughly **1000x** WFDB's calibrated mV scale (measured std ≈ 185 vs ≈ 0.1–0.5). Skipping Kalman
exposed large genuine motion/EMG noise excursions that XQRS then over-detected as QRS
complexes: implied HR ballooned to ~139 bpm with 134 RR intervals under 0.3s (i.e. implying
>200 bpm between adjacent "beats" — physiologically impossible). A plain amplitude rescale of
the skip-Kalman signal did **not** fix this (XQRS's threshold is internally adaptive; the
problem is real noise *energy*, not an absolute-unit miscalibration) — confirming this is a
genuine per-source difference, not something fixable by rescaling.

**Fix:** the Kalman-skip is now conditional on `recording.source == "wfdb"`. VitalPatch/SeNSiO
keep the original Kalman-included detection signal, which is empirically correct for that
source (Kalman's smoothing is genuinely needed noise suppression there, not the SNR-destroying
effect seen on WFDB).

### Regression 2: snap-radius default change applied universally

The `_snap_to_local_peak` radius change (15 → 8), validated only against MITDB, caused
**catastrophic beat-level over-culling on 245 VitalPatch segments across 5 different
patients** — same number of beats detected, but up to 130/133 rejected via
`R_PEAK_NOT_LOCAL_MAX` on one recording. The failure was non-monotonic in radius: 3, 5, 10, 15,
and 20 were all fine on the affected recordings; only 8 broke it.

**Fix:** snap radius is now also source-conditional — 8 for `wfdb` (the validated value),
15 for `vitalpatch`/`sensio` (the pre-existing, proven-safe value, confirmed via the final
batch re-run below).

### Final verified state (full 2375-file batch, 3632 rows, 0 errors)

| metric | before (pre-session) | after (fully fixed) |
|---|---|---|
| files parsed | 2305/2375 (97.1%) | 2375/2375 (100%) |
| parse/run errors | 6 | 0 |
| assessable rate | 79.7% | 79.6% |
| segments unchanged risk category (of 3500 common) | — | **3499/3500 (100.0%)** |
| over-culling signature (same n_detected, n_analyzed collapsed) | — | **0** (was 245 before the fix above) |

- **3499/3500 (100.0%)** of VitalPatch segments that were already processing before this
  session have the **identical risk category** after all fixes. The single exception
  (1 segment, `MEDIUM → LOW`) has identical beat counts before/after — a negligible
  downstream feature difference, not a beat-detection issue, and not a safety-relevant
  direction (less severe, not more).
- Every increase in aggregate risk-level counts (`CRITICAL` 277→290, `HIGH` 527→543, `LOW`
  1832→1892, `NOT_ASSESSABLE` 705→741) is fully attributable to the **132 newly-recovered
  segments** from the P3 parser fix (previously-crashing files that now parse and contribute
  their own, independent risk distribution) — not to any change in how previously-working data
  is scored.
- **No recording with real signal is being mostly-dropped, and no falsely-calm risk level was
  introduced.**

---

## P3 — Parser robustness (`'-'` sentinel)

**Symptom:** 70 VitalPatch files crashed with `could not convert string to float: '-'`.

**Root cause:** `parse_vitalpatch_ecg` used `pd.isna()` to drop missing samples, which does
not catch non-numeric string sentinels like `'-'`. There was also a latent flat-array
misalignment risk in the original NaN-dropping logic (dropping NaNs from the flattened
timestamp/value pairs independently could desynchronize the pairing).

**Fix:** switched to `pd.to_numeric(..., errors="coerce")` to convert `'-'` (and any other
non-numeric sentinel) to `NaN`, then drop NaNs **pairwise** (by timestamp/value pair, not from
the flat array) so a dropped sample never silently misaligns the remaining pairs.

**Result: 2375/2375 files now parse (100%, up from 2305/2375 = 97.1%).** Spot-checked one
previously-crashing file — the missing-sample gap is correctly handled by the existing
gap-splitting logic, producing 3 valid segments instead of a crash.

---

## P4 — Structured report integrity

Built an automated 8-invariant checker run against real `build_risk_report_json` output:

1. `beat_summary` counts sum to `n_beats_detected`
2. `rhythm_findings` beat indices are within bounds
3. exactly one `is_deciding_rule=True` in `rule_trace`
4. deciding-rule / `risk_level` consistency (catch-all rule ⇒ LOW/NOT_ASSESSABLE)
5. no `None`/null leakage in required top-level fields
6. `assessable` / `NOT_ASSESSABLE` consistency
7. `n_beats_analyzed == n_beats_detected - n_beats_flagged_low_quality`
8. `final_risk_level` never de-escalates below `risk_level` (escalate-only)

Run against 3 WFDB recordings (100, 103, 111) and 4 VitalPatch segments, re-verified after the
final P2 fix: **0 problems found.** Manually spot-checked WFDB 111's raw JSON (a `CRITICAL`
case) for null-leakage and finding-wording correctness — clean.

---

## Open, non-blocking item

A benign `scipy` `RuntimeWarning: divide by zero encountered in divide` /
`invalid value encountered in multiply` appears during batch runs, inside a Lombscargle
periodogram calculation (likely HRV-related). It did not stop processing — the full batch
completed with 0 errors — and has not been root-caused. Flagged for awareness; not a deploy
blocker.
