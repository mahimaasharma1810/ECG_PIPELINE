# Diagnostic Report v3 — `1780171482351_VC2B008BF_184B27_ecg_seg1`

**Generated:** 2026-08-02 · Segment: patient `184B27`, VitalPatch, 115.44 s, 125 Hz, single lead

**What changed since v2:** the 4 reviewed defects were fixed and SHAP attribution
was added. All numbers below come from a fresh run on the fixed code, with
`vitals_root` supplied so NEWS2/qSOFA are genuinely evaluated.

**Regression status: 5/5 synthetic ground-truth scenarios pass** (NORMAL,
PVC_BURDEN, VT_RUN, AFIB_LIKE, NOISY). Verdict unchanged: **CRITICAL**.

---

## Fix status

| # | Defect | Status | Verification |
|---|---|---|---|
| 1 | AFib text said `0.15`, code used `0.10` | **FIXED** | Now renders `> 0.10`, read from the finding, not a literal |
| 2 | `pct_of_analyzed_beats` divided by detected | **FIXED** | Renamed `pct_of_detected_beats`; legacy JSONs still readable |
| 3 | Vitals available but not wired in | **FIXED** | NEWS2/qSOFA now `evaluated: true`, scores 4 and 1 |
| 4 | No HR field; 80 bpm for a 116 bpm patient | **FIXED** | `heart_rate_bpm: 118.1`, within 2.3 bpm of device |
| 5 | No beat-classification evidence | **ADDED** | Exact TreeSHAP on the 5 least-confident beats |

---

### Fix 1 — AFib threshold, single source of truth

The threshold was hoisted into `RiskThresholds` as `afib_rr_cv_threshold = 0.10`
and is now carried *inside each finding* (`rr_cv_threshold`, `window_beats`), so
the display layer renders the threshold that actually produced that finding
rather than restating a literal that can drift.

```
before:  RR coefficient-of-variation 0.299 > 0.15 over a 20-beat rolling window
after:   RR coefficient-of-variation 0.299 > 0.10 over a 20-beat rolling window
```

Two further stale `0.15` strings were found beyond the reported one and also
fixed: `synthetic_ecg.py:504` and `test_pipeline_synthetic.py:104`, both of which
told the same wrong number in test output.

### Fix 2 — `pct_of_detected_beats`

**I took the rename, not the denominator change, and the reason matters.**
Switching the denominator to 119 would have made `beat_summary` report V at
31.93% while the rule trace reports PVC burden at 29.921% — the same quantity,
two different numbers, in one report. `score_recording` computes burden over all
detected beats, so the display must use the same base. Renaming keeps one number
and makes the label true.

The self-contradictory line is gone:

```
before:  Q: 8 beats (6.3% of analyzed beats)     <- Q beats are NOT analyzed
after:   Q: 8 beats (6.3% of detected beats)
```

A `beat_summary_pct()` accessor reads both the new and legacy key, so the ~9,000
already-saved report JSONs under `data/reports/` still render. Verified live
against a 2026-07-26 file: legacy → `N: 5 beats (100.0% of detected beats)`,
new → `N: 72 beats (56.69% of detected beats)`. Mirrored into
`ecg_inference/report.py` so the two copies don't drift.

### Fix 3 — vitals wired in

`run_full_report(recording, classifier, vitals_root=...)` now matches vitals via
the existing containment rule and feeds partial NEWS2 / qSOFA-proxy into the
cascade. Opt-in: omitting `vitals_root` reproduces the previous ECG-only
behaviour exactly, so no existing caller changes silently. `--vitals-root` added
to the VitalPatch batch runner.

The overrides are now genuinely evaluated:

```
before:  "measured_value": null, "evaluated": false   (both rules)
after:   NEWS2  measured 4 vs threshold 7 -> evaluated: true,  fired: false
         qSOFA  measured 1 vs threshold 2 -> evaluated: true,  fired: false
```

Nothing is imputed. Every unavailable component reports `value: null`,
`score: null`, `source: "NOT_AVAILABLE"` and a reason:

| NEWS2 component | Value | Score | Source |
|---|---|---|---|
| heart_rate | 115.8 bpm (18 readings) | 2 | `vitalpatch_col1` |
| respiratory_rate | 23.8 /min (58 readings) | 2 | `vitalpatch_col2` |
| temperature | `null` | `null` | NOT_AVAILABLE — no valid readings in file |
| spo2 | `null` | `null` | NOT_AVAILABLE — VitalPatch has no SpO2 sensor |
| systolic_bp | `null` | `null` | NOT_AVAILABLE — VitalPatch has no BP sensor |
| consciousness | `null` | `null` | NOT_AVAILABLE — no AVPU input path |

**NEWS2 = 4, from 2 of 6 components** (matched by interval containment at **0 ms
offset**). **qSOFA proxy = 1, from 1 of 3.** Neither reaches its threshold, so
neither changed the verdict — ECG alone had already hit CRITICAL.

The `known_limitations` text now states the partial-coverage caveat instead of
the now-false "not wired into this bridge", and the not-evaluated note explains
that it means *this run was ECG-only*, not that no vitals exist.

### Fix 4 — heart rate field

```json
"heart_rate_bpm": 118.1,
"heart_rate_method": "median_nonflagged_rr",
"heart_rate_n_intervals": 106,
"heart_rate_n_intervals_total": 126,
"heart_rate_flagged_fraction": 0.1587,
"heart_rate_warning": null
```

**118.1 bpm vs the device's 115.8 bpm — a 2.3 bpm gap**, against 35.66 bpm for
the mean-of-all-RR reading that v2 exposed. `heart_rate_warning` is `null` here
because both guards passed (gap < 5 bpm; flagged fraction 15.87% < 20%). The
warning fires on either condition and names which one, so an absent warning is an
explicit pass rather than a silent one.

### Fix 5 — SHAP attribution

Uses XGBoost's built-in `pred_contribs=True` (exact TreeSHAP), **not** the `shap`
package, which is not installed here. Numerically identical for this model, no
new dependency.

**Validated:** for beat 52, `sum(SHAP) + base_value = −0.202937` and the model's
own margin is `−0.202937` — agreement to **4.2×10⁻⁷**. These are exact
attributions, not approximations.

---

## 1. Signal quality

| Metric | Value |
|---|---|
| SQI windows | 23 (14 passed, 9 rejected) |
| **SQI rejection rate** | **0.3913** |
| **Quality score** | **0.6087** |
| **Usable signal** | **60.94%** (8,750 / 14,359 samples) |

Rejects: `NO_QRS_IMPULSE_CHARACTER` ×6, `FLATLINE` ×2, `BASELINE_WANDER` ×1.
Clipping `0.0000` in all 23 windows. Min SNR 6.99 dB (above the 5.0 dB floor).
No single "noise %" metric exists; the honest proxies are the 39.13% window
rejection and 39.06% sample discard.

**Verdict: MARGINAL — assessable but degraded.**

## 2. R-peak detection

| Field | Value |
|---|---|
| Detector | WFDB `XQRS`, `Conf(t_inspect_period=0.36)` |
| **`t_inspect_period`** | **0.36 s** |
| Refractory guard | 200 ms |
| **Total detected** | **127** (119 analyzed, 8 `Q`) |
| **Average RR** | **748.70 ms** all / **541.81 ms** non-flagged |
| **Min RR** | **200.00 ms** (exactly the guard floor) |
| **Max RR** | **7784.00 ms** (missed-beat gap, flagged) |
| **Duplicate indices** | **0** — 127 detected, 127 unique, 0 at RR=0 ms |

### Implied HR vs vitals HR

| Method | HR | Δ vs 115.8 | |
|---|---|---|---|
| **`heart_rate_bpm` (median non-flagged) — now reported** | **118.1** | **+2.3** | **PASS** |
| Mean non-flagged RR | 110.7 | −5.1 | fail |
| Mean of all RR (what v2 exposed) | 80.1 | −35.7 | fail |
| Beats ÷ full duration | 66.0 | −49.8 | fail |

**No >5 bpm gap flagged.** The pipeline's reported HR now agrees with the device.

## 3. Beat classification

| Class | Count | Avg Confidence | Min Confidence |
|---|---|---|---|
| **N** | **72** | **0.8910** | 0.3983 |
| **V** | **38** | **0.8318** | **0.3760** |
| **S** | **9** | **0.5709** | 0.3879 |
| **F** | **0** | — | — |
| **Q** | **8** | — (not classified) | — |

Overall: mean 0.8479, median 0.9259, **50 beats ≥0.95**, **25 beats <0.70**.
`Q` carries `null` confidence, not 0.0 — a rejected beat was never shown to the
classifier, which is different from being classified with low confidence.

PVC burden = 38/127 × 100 = **29.921%** · PAC = 9/127 × 100 = **7.087%**

## 4. Lowest-confidence beats, with SHAP

SHAP in margin (log-odds) space. `toward` = pushed the model toward the class it
chose.

**Beat 52 · t=53.240 s · V · conf 0.3760 · neighbors S→S**
(base 0.1025, sum −0.3054; runner-up N at 0.3737 — loses by 0.0023)

| Feature | SHAP | Dir | Value | Meaning |
|---|---|---|---|---|
| `rr_pre_ms` | **+1.0266** | toward | 328.0 | RR before this beat (prematurity) |
| `wavelet_d3_9` | +0.5691 | toward | 0.2211 | shape coefficient |
| `wavelet_d3_8` | +0.5194 | toward | −0.2608 | shape coefficient |
| `wavelet_d4_5` | −0.4438 | away | 1.1390 | shape coefficient |
| `local_hrv_ms` | −0.4107 | away | 160.0 | RR change across beat |

**Beat 79 · t=67.432 s · S · conf 0.3879 · neighbors N→N**

| Feature | SHAP | Dir | Value | Meaning |
|---|---|---|---|---|
| `wavelet_d4_5` | +0.6162 | toward | 1.7992 | shape coefficient |
| `rr_pre_ms` | +0.4557 | toward | 416.0 | prematurity |
| `area_ratio_pre_post` | −0.3082 | away | 0.2297 | QRS asymmetry |
| `wavelet_a4_12` | −0.2699 | away | −2.3610 | shape coefficient |
| `local_hrv_ms` | +0.1523 | toward | 88.0 | RR change |

**Beat 51 · t=52.912 s · S · conf 0.3973 · neighbors V→V**

| Feature | SHAP | Dir | Value | Meaning |
|---|---|---|---|---|
| `local_hrv_ms` | **−0.9466** | away | −80.0 | RR change |
| `rr_pre_ms` | +0.5273 | toward | 408.0 | prematurity |
| `wavelet_a4_9` | +0.2293 | toward | 5.2118 | shape coefficient |
| `wavelet_d3_7` | −0.2093 | away | −0.0691 | shape coefficient |
| `wavelet_d3_14` | −0.2080 | away | 0.0295 | shape coefficient |

**Beat 68 · t=61.968 s · N · conf 0.3983 · neighbors V→V**

| Feature | SHAP | Dir | Value | Meaning |
|---|---|---|---|---|
| `rr_pre_ms` | **−2.6620** | away | 424.0 | prematurity |
| `local_hrv_ms` | +0.5951 | toward | 80.0 | RR change |
| `amplitude_range` | +0.2269 | toward | 3.8439 | QRS size |
| `wavelet_d4_0` | +0.2169 | toward | −0.0278 | shape coefficient |
| `wavelet_d3_9` | −0.2112 | away | 0.2890 | shape coefficient |

**Beat 27 · t=29.680 s · V · conf 0.4809 · neighbors V→N**

| Feature | SHAP | Dir | Value | Meaning |
|---|---|---|---|---|
| `local_hrv_ms` | **+2.9265** | toward | **5104.0** | RR change across beat |
| `rr_pre_ms` | −1.2738 | away | 728.0 | prematurity |
| `wavelet_d3_8` | −1.1046 | away | 2.0775 | shape coefficient |
| `wavelet_d3_7` | −0.4239 | away | 1.0853 | shape coefficient |
| `wavelet_d4_8` | −0.3222 | away | −1.1931 | shape coefficient |

### What SHAP reveals — two findings, one of them new

**(a) The classifier is deciding on timing, not morphology.**
Across all five weakest beats, `rr_pre_ms` and `local_hrv_ms` are the dominant
attributions. Morphology features (`amplitude_range`, `area_ratio_pre_post`)
appear only twice, and small. This model is calling V mainly from
**prematurity**, not from wide-QRS shape. That is a defensible statement to make
in a thesis, and a limitation to state plainly: a short-RR beat of any origin can
be pulled toward V.

**(b) NEW — missed-beat gaps are leaking into the feature vector.**
Beat 27's top attribution is `local_hrv_ms = 5104 ms`, the single largest SHAP
value anywhere in this segment. That is not physiology — it is a **5.1-second
detection gap** across an SQI-rejected stretch. `_morphological_features`
computes `local_hrv = rr_post − rr_pre` with **no `rr_flagged` guard**, even
though `recording_level_hrv()` and the AFib rule both exclude flagged intervals.
Quantified on this segment:

| Measure | Value |
|---|---|
| Beats whose `rr_pre`/`local_hrv` use an out-of-physiological-range interval | **19 of 127** |
| Their labels | N 9, **V 6**, Q 2, S 2 |
| V beats resting on an out-of-range interval | **6 of 38 (15.8%)** |
| PVC burden as reported | **29.921%** (38/127) |
| PVC burden excluding those 6 | **25.197%** (32/127) |
| CRITICAL threshold | 20.0% |

**The verdict survives** — 25.2% is still above 20%, so this segment is CRITICAL
either way. But 6 of the 38 beats driving it are partly justified by detection
gaps, and this mechanism is not specific to this segment. This is a genuine
finding that only became visible once SHAP was added. **I have not fixed it** —
it is upstream of the reviewed defect list, it changes classifier inputs, and it
warrants its own decision.

## 5. Beat timeline

```
  0- 19:  N V N N N N N V Q N S N V V N N N N V N
 20- 39:  V N N V Q N V V N V N Q N N V N N N N Q
 40- 59:  N S V N N V N V N S V S V S N V N V N V
 60- 79:  V N N N V Q S V N V N V V N V V N N N S
 80- 99:  N V V N N N N N N N V N V N N Q N N V N
100-119:  N S Q N N V V N N N S N N Q N N N V N N
120-126:  V V N N N V N
```

All three pattern rules fell **exactly one unit short**:

| Pattern | Longest run | Threshold | Fired |
|---|---|---|---|
| Consecutive `V` (VT) | **2** | ≥3 | NO |
| `N,V` repeats (bigeminy) | **3** | ≥4 | NO |
| `N,N,V` repeats (trigeminy) | **2** | ≥3 | NO |

8 V-pairs, 22 isolated V beats, no triplet anywhere. Pre-fix this segment
reported **3 VT runs** — the `t_inspect_period` fix removed them.

## 6. Rule engine trace

| # | Rule | Measured | Threshold | Fired | Would set |
|---|---|---|---|---|---|
| **1** | **PVC burden > CRITICAL** | **29.921%** | **20.0%** | **YES** | **CRITICAL** |
| 2 | VT run count > 0 | 0 | 0 | no | CRITICAL |
| 3 | PVC burden > HIGH | 29.921% | 10.0% | YES | HIGH |
| 4 | PAC > HIGH OR AFib > HIGH | PAC 7.087% / **AFib 100.0%** | 15.0% / 30.0% | YES | HIGH |
| 5 | HRV suppression: SDNN < thr | 164.351 ms | 20.0 ms | no | MEDIUM |
| 6 | NEWS2 override | **4** *(now evaluated)* | 7 | no | CRITICAL |
| 7 | qSOFA override | **1** *(now evaluated)* | 2 | no | HIGH |

AFib windows (5 examined, 5 flagged → 100.0%), all now rendering the true `0.10`
threshold: CV 0.2994, 0.2741, 0.3200, 0.1785, 0.3513.

## 7. Risk verdict

**FINAL: CRITICAL**

```
★ DECIDING RULE ★
  PVC burden > CRITICAL threshold
    measured  : 29.921   (38 V of 127 detected)
    threshold : 20.0
    fired     : true
    is_deciding_rule : true
```

Three rules fired (1, 3, 4); the cascade is first-match-wins in severity order so
rule 1 wins. Rules 6 and 7 were evaluated against real vitals and did not fire.
**Vitals neither raised nor lowered the verdict** — and now the report says so
from measured values rather than from "no score supplied".

## 8. MedGemma input JSON

Not sent — CRITICAL. Full prompt is 15,151 chars; the structured payload is
saved alongside this report. The blocks that changed since v2:

```json
"recording": {
  "n_beats_detected": 127, "n_beats_analyzed": 119, "n_beats_flagged_low_quality": 8,
  "heart_rate_bpm": 118.1, "heart_rate_method": "median_nonflagged_rr",
  "heart_rate_n_intervals": 106, "heart_rate_n_intervals_total": 126,
  "heart_rate_flagged_fraction": 0.1587, "heart_rate_warning": null
},
"beat_summary": { "V": {"count": 38, "pct_of_detected_beats": 29.92, ...} },
"rule_trace": [
  {"condition": "NEWS2 safety override (>= critical threshold)",
   "measured_value": 4, "threshold": 7, "fired": false, "evaluated": true},
  {"condition": "qSOFA safety override (>= high threshold)",
   "measured_value": 1, "threshold": 2, "fired": false, "evaluated": true}
],
"safety_overrides": [
  {"override": "NEWS2", "score": 4, "threshold": 7, "applied": false},
  {"override": "qSOFA", "score": 1, "threshold": 2, "applied": false}
],
"vitals": {
  "status": "matched", "match_method": "interval_containment",
  "time_offset_ms": 0, "time_offset_minutes": 0.0, "posture": "Standing",
  "news2": {"total_score": 4, "components_available": 2, "components_total": 6,
            "incomplete": true, "missing_components": ["spo2","sbp","consciousness","temperature"]},
  "qsofa": {"total_score": 1, "components_available": 1, "components_total": 3,
            "incomplete": true}
}
```

## 9. MedGemma output

**`SKIPPED_CRITICAL`.** No LLM output exists. `render_narrative()` returns before
`call_medgemma()` when risk is CRITICAL; audit event
`TRANSPARENCY_MEDGEMMA_SKIPPED_CRITICAL` logged, reason *"CRITICAL level bypasses
MedGemma entirely, per safety constraint."* Endpoint
`http://localhost:11434/api/generate`, model `medgemma`. Timeout not applicable —
no call made (`ecg_inference/report.py` constant is 90 s; agent side 120 s;
measured cold start 75.6 s).

The narrative is the deterministic local fallback. Every number in it traces to
the rule trace or the frozen model card — **no unsupported claims**.

## 10. Self-consistency checks

| # | Check | Result |
|---|---|---|
| 1 | V_count / total_detected × 100 = reported PVC burden | **PASS** |
| 2 | Implied HR matches vitals HR | **PASS** (was FAIL) |
| 3 | Risk verdict follows from deciding rule | **PASS** |
| 4 | SDNN is not the 0.0 sentinel | **PASS** |
| 5 | Beat counts sum to total detected | **PASS** |
| 6 | AFib displayed threshold = code threshold | **PASS** (was FAIL) |
| 7 | Field names match their denominators | **PASS** (was FAIL) |
| 8 | NEWS2/qSOFA evaluated when vitals exist | **PASS** (was FAIL) |
| 9 | SHAP values reconcile with model margin | **PASS** |

**1 — PASS.** `38/127 × 100 = 29.92125984…` → reported `29.921`. Exact.
`9/127 × 100 = 7.08661417…` → `7.087`. Exact.

**2 — PASS (was FAIL).** Reported `heart_rate_bpm` 118.1 vs device 115.8 =
**2.3 bpm**, inside the 5 bpm bar. The v2 failure (−35.66 bpm) came from
mean-of-all-RR, which the method field now explicitly rules out.

**3 — PASS.** Rule 1 fired → CRITICAL; marked deciding; `final_risk_level`
CRITICAL. Rules 3 and 4 fired at HIGH and are correctly non-deciding. Rules 6/7
evaluated, did not fire, did not alter the level.

**4 — PASS.** SDNN **164.351 ms** from 106 valid intervals, well above the
3-interval floor. RMSSD 260.811 ms, pNN50 80.952%. High, not suppressed —
coherent with 29.9% ectopy and RR CV 0.18–0.35. Suppression rule correctly did
not fire.

**5 — PASS.** 72+9+38+0+8 = **127** = `n_beats_detected`; 127−8 = 119 =
`n_beats_analyzed`. 127 unique indices, 0 duplicates, 0 at RR=0 ms, min RR
exactly 200.00 ms.

**6 — PASS (was FAIL).** Displayed `> 0.10` = `RISK.afib_rr_cv_threshold` =
`_afib_suspected`'s operative threshold. Carried per-finding, so it cannot drift.

**7 — PASS (was FAIL).** `pct_of_detected_beats` divides by 127 detected. The
"Q: 8 beats (6.3% of analyzed beats)" contradiction is gone.

**8 — PASS (was FAIL).** Both overrides `evaluated: true` with real measured
values (4, 1). Unavailable components marked `NOT_AVAILABLE` with reasons, never
imputed.

**9 — PASS.** Beat 52: `sum(SHAP) + base = −0.202937`; model margin
`−0.202937`; difference **4.2×10⁻⁷**.

## 11. Pipeline timing

| Stage | Time (ms) |
|---|---|
| Ingest / parse | 19.2 |
| SQI gate | 87.9 |
| Resample (125→125, pass-through) | 0.0 |
| Filter chain | 21.2 |
| R-peak detect + segment | 234.6 |
| Feature extraction | 37.3 |
| HRV | 13.2 |
| Beat classification (per-beat loop, 119 beats) | 183.2 |
| Rhythm analysis | 0.7 |
| Rule engine + risk scoring | 0.1 |
| **Pipeline stages 1–9** | **645.9** |
| Vitals match (warm cache) | 7.2 |
| Vitals match (cold, builds per-patient index) | 863.9 |
| `beat_confidence` block without SHAP | 0.4 |
| **`beat_confidence` block with SHAP ×5** | **334.5** |
| ↳ **SHAP marginal cost** | **334.1** (~67 ms/beat) |
| MedGemma call | **0 (SKIPPED_CRITICAL)** |
| **`run_full_report` total (warm)** | **911.5** |

Honest notes:

- **SHAP is now the second-largest cost in the report path** — 334 ms for 5
  beats, ~37% of total. Almost all of it is per-call `DMatrix` + booster setup,
  not the SHAP maths. It is bounded because only 5 beats are explained;
  extending it to all 119 would be roughly 8 s and would need batching first.
- Beat classification still uses the **per-beat loop**
  (`ecg_pipeline_core.py`, `predict_one` per beat). The batched fix exists only
  in `ecg_inference/classifier.py`, which this path does not import. Unchanged
  by this work.
- Cold vitals matching (863.9 ms) is a one-time per-patient index build, not a
  per-segment cost; warm it is 7.2 ms.

---

## Remaining known issues (not fixed here)

1. **Missed-beat gaps leak into classifier features** — `local_hrv`/`rr_pre` have
   no `rr_flagged` guard. 19/127 beats affected, 6 of them V (15.8% of all V).
   Verdict robust here (25.2% still > 20%), but the mechanism is general.
   Discovered by the SHAP work; needs its own decision.
2. **SDNN 0.0 sentinel still live in `ecg_inference/`** — the deployment path
   (`features.py`, `classifier.py`) was never mirrored. Untouched by this work.
3. **`audit_logs` table has 0 rows** (against 6,760 `alerts`/`vitals_snapshots`),
   and `vitals_snapshots` has no segment ID column, so segment-level traceability
   does not exist even in principle.
4. **Agent-side `calculate_qsofa` is not qSOFA** — it scores HR>90 and SpO2<94,
   neither of which is a Seymour 2016 criterion, and escalates to CRITICAL at ≥2.
   Agent-side `calculate_news2` includes diastolic BP, which is not a NEWS2
   component. This is the one clinically wrong thing found, and it is in
   `MedGemma-Agent/`, outside the reviewed defect list.
5. **Risk thresholds remain project-defined** — PVC 20%/10%, PAC 15%, AFib
   burden 30%, SDNN 20 ms are unvalidated. The deciding rule is one of them. Only
   the AFib *detector* threshold (0.10) is swept against labeled data (LTAFDB).
6. **No clinician-annotated VitalPatch labels exist.** SHAP explains the model,
   not the physiology. The 38 V beats remain unconfirmed by any human.
