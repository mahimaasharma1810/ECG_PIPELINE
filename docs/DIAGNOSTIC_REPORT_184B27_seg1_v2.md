# Diagnostic Report — VitalPatch segment `1780171482351_VC2B008BF_184B27_ecg_seg1`

**Generated:** 2026-08-02 · **Commit:** `e0da1f7` (branch `mod_1`) · **No code was modified.**

All numbers below come from a live re-run of the existing post-fix pipeline via
`ecg_pipeline.agent_bridge.run_full_report()`. Nothing is a placeholder.

**Segment selection:** patient `184B27`, the verdict-flip case. Pre-fix
(`multimodal_manifest.json`, 2026-07-30) this segment scored **149 analyzed beats
with 3 VT runs**. Post-fix it scores **119 analyzed beats with 0 VT runs**. It
remained CRITICAL, but the *deciding rule changed* from VT runs to PVC burden.
It also has the highest PVC burden of any post-fix CRITICAL segment with a
substantial beat count, and is the only segment in the 3,632-segment corpus whose
agent-side qSOFA reached 2.

> **Note on saved reports:** the full 2,889-segment corpus *was* re-scored after
> the `t_inspect_period` fix (`docs/audits/AUDIT_2026-07-31.md` §7). What was not re-run
> is report regeneration — the `.json`/`.md` files on disk under
> `data/reports/vitalpatch/` are still pre-fix (mtime 2026-07-26). This report
> supersedes the stale saved report for this segment.

**Recording:** 115.44 s · 14,359 samples · 125.0 Hz · single lead · device
`VC2B008BF` · 2026-05-30T20:04:42.788Z → 20:06:38.228Z

---

## 1. Signal quality metrics

Stage-2 SQI gate, 5.0 s windows, run **before** any filtering.

| Metric | Value |
|---|---|
| SQI windows total | 23 |
| Windows passed | 14 |
| Windows rejected | 9 |
| **SQI window rejection rate** | **0.3913 (39.13%)** |
| **SQI / quality score** | **0.6087** (= 1 − rejection rate) |
| Samples kept | 8,750 / 14,359 |
| **Usable signal percentage** | **60.94%** |
| Samples discarded | 5,609 (39.06%) |

**Reject-code breakdown (9 windows):**

| Reject code | Count | Threshold |
|---|---|---|
| `NO_QRS_IMPULSE_CHARACTER` | 6 | kurtosis < 1.5 |
| `FLATLINE` | 2 | flatline_frac > 0.05 (≥400 ms stuck run) |
| `BASELINE_WANDER` | 1 | ratio > 1.2 |

**Measured per-window ranges (all 23 windows):**

| Metric | min | max | mean | Rejects when |
|---|---|---|---|---|
| `flatline_frac` | 0.0000 | 0.1184 | 0.0095 | > 0.05 |
| `clipping_frac` | 0.0000 | **0.0000** | 0.0000 | > 0.02 |
| `missing_frac` | 0.0000 | 0.0000 | 0.0000 | > 0.03 |
| `morphology_kurtosis` | −0.3047 | 7.8573 | 2.3564 | < 1.5 |
| `baseline_wander_ratio` | 0.2957 | 3.0998 | 0.8526 | > 1.2 |
| `snr_db` | 6.9954 | 21.6874 | 13.6334 | < 5.0 |

**Noise percentage:** the pipeline emits **no single "noise %" metric**. The
honest proxies are the **39.13% window rejection rate** and **39.06% of samples
discarded**. Minimum window SNR was 6.99 dB — above the 5.0 dB floor, so no
window failed on SNR alone.

- Baseline wander detected: **YES** (1 window; max ratio 3.0998 vs 1.2)
- Flatline detected: **YES** (2 windows; max frac 0.1184 vs 0.05)
- Clipping detected: **NO** (`clipping_frac` exactly 0.0000 in all 23 windows)

**Verdict: MARGINAL — assessable but degraded.** The discarded 39% is
concentrated enough to leave five multi-second stretches with zero detected
beats, which is the direct cause of the HR failure in §10.

---

## 2. R-peak detection summary

| Field | Value |
|---|---|
| **Detector** | WFDB `XQRS` adaptive-threshold (`wfdb.processing.XQRS`) |
| **`t_inspect_period`** | **0.36 s** — `XQRS.Conf(t_inspect_period=0.36)`, `ecg_pipeline_core.py:991` |
| Post-detection guard | `_apply_refractory_guard`, 200 ms minimum spacing |
| Snap-to-local-peak radius | 15 samples (non-wfdb sources) |
| **Total detected beats** | **127** |
| Beats analyzed | 119 |
| Beats rejected by beat-level SQI | 8 (labeled `Q`) |
| RR intervals total | 126 |
| RR intervals flagged out-of-range | 20 |
| RR intervals used for HRV/AFib | 106 |

**RR interval statistics:**

| Metric | All RR (n=126) | Non-flagged RR (n=106) |
|---|---|---|
| **Average RR** | **748.70 ms** | **541.81 ms** |
| Median RR | — | 508.00 ms |
| **Min RR** | **200.00 ms** | 304.00 ms |
| **Max RR** | **7784.00 ms** | 1104.00 ms |

The 20 flagged RR values in full:
`200, 232, 280, 280, 288, 288, 312, 448, 456, 504, 512, 552, 584, 640, 648,
5328, 5728, 5832, 6008, 7784` ms.

Two honest qualifications:

1. **The five intervals of 5328–7784 ms are missed-beat gaps, not real pauses.**
   They align with the SQI-rejected stretches. No heart paused for 7.8 s.
2. **8 of the 20 flagged values are physiologically normal** (448–648 ms, inside
   the configured 300–2000 ms range). `segment_beats` flags a **beat** if
   *either* `rr_pre` **or** `rr_post` is out of range
   (`ecg_pipeline_core.py:1050-1053`); downstream, that beat's perfectly good
   `rr_post` is discarded too. One bad interval costs two intervals of HRV data,
   shrinking the sample 126 → 106. Conservative, but undocumented.

### Implied HR from R-peaks vs vitals HR

**Vitals HR: 115.8 bpm** (VitalPatch col 1, 18 readings in the ±2 min window,
range 113–119 bpm, matched at **0 ms offset** by interval containment).

| Method | Implied HR | Δ vs 115.8 | >5 bpm gap? |
|---|---|---|---|
| Beat count / beat span (126 intervals ÷ 94.336 s) | **80.14 bpm** | **−35.66** | **⚠ FLAGGED** |
| Beat count / full duration (127 ÷ 115.44 s) | 66.01 bpm | −49.79 | ⚠ FLAGGED |
| 60000 / mean of all RR (748.70 ms) | 80.14 bpm | −35.66 | ⚠ FLAGGED |
| 60000 / mean of non-flagged RR (541.81 ms) | 110.74 bpm | −5.06 | ⚠ FLAGGED (marginal) |
| 60000 / median of non-flagged RR (508.00 ms) | 118.11 bpm | +2.31 | OK |

**⚠ MISMATCH FLAGGED — 35.66 bpm, 7× over the 5 bpm bar.**
Beat span: first R-peak 15.152 s, last 109.488 s (94.336 s of 115.44 s).
Instantaneous HR from valid RR spans 54.3 – 197.4 bpm.
Cause and interpretation in §10, CHECK 2.

### Duplicate indices

- **Duplicate R-peak indices: 0** (127 detected, 127 unique)
- **Intervals at RR = 0 ms: 0**
- **Intervals below 200 ms: 0** — minimum RR is exactly **200.00 ms**, i.e. the
  refractory guard is holding precisely at its configured boundary.

---

## 3. Beat classification table

Classifier: frozen production XGBoost `five_class_xgb.json`, 56-dim features,
`source = trained_model` for all 119 analyzed beats (no rule-based fallback).

| Class | Count | Avg Confidence | Min Confidence |
|---|---|---|---|
| **N** | **72** | **0.8910** | 0.3983 |
| **V** | **38** | **0.8318** | **0.3760** |
| **S** | **9** | **0.5709** | 0.3879 |
| **F** | **0** | n/a | n/a |
| **Q** | **8** | n/a — not classified | n/a |
| **Total** | **127** | 0.8479 (over 119 classified) | 0.3760 |

`Q` beats are assigned by the deterministic quality gate, not predicted, so they
carry no confidence. Measured `Q` reject reasons: `R_PEAK_NOT_LOCAL_MAX` ×6,
`EXCESS_BASELINE_DRIFT` ×2.

**Confidence distribution over the 119 classified beats:** mean 0.8479, median
0.9259, min 0.3760, max 1.0000. **50 beats (42.0%) at ≥0.95**; **25 beats
(21.0%) below 0.70**.

**Reading this honestly:** the S class averages **0.5709** — barely above chance
for a 4-way decision, matching its held-out F1 of 0.139. The V class averages
0.8318 and drives the CRITICAL verdict, but its *minimum* is 0.3760, and that
lowest-confidence beat is counted in the 38.

**Burden formulas:**
PVC burden = 38 / 127 × 100 = **29.9213%** → reported `29.921`
PAC burden = 9 / 127 × 100 = **7.0866%** → reported `7.087`

> **Denominator warning.** `score_recording` uses `n = len(labels)` = **127**,
> i.e. all detected beats *including the 8 unclassified `Q` beats*, not the 119
> analyzed. Over analyzed beats PVC would be 31.93%. Both exceed the 20%
> threshold so the verdict is unaffected — but the JSON field is **mislabeled**
> `pct_of_analyzed_beats` (`agent_bridge.py:628`) while dividing by the detected
> count. The same block prints "Q: 8 beats (6.3% of analyzed beats)", which is
> self-contradictory: `Q` beats are *by definition* the ones not analyzed.

---

## 4. Top 10 lowest-confidence beats

Neighbors shown as `[n−2 n−1] PRED [n+1 n+2]`.

| # | Beat idx | Timestamp | Predicted | Confidence | Neighboring beats | Runner-up probability |
|---|---|---|---|---|---|---|
| 1 | **52** | 53.240 s | **V** | **0.3760** | `[V S] V [S N]` | N 0.3737 — **loses by 0.0023** |
| 2 | 79 | 67.432 s | S | 0.3879 | `[N N] S [N V]` | V 0.3613 |
| 3 | 51 | 52.912 s | S | 0.3973 | `[S V] S [V S]` | N 0.3065 |
| 4 | 68 | 61.968 s | N | 0.3983 | `[S V] N [V N]` | V 0.3678 |
| 5 | 27 | 29.680 s | V | 0.4809 | `[N V] V [N V]` | N 0.3380 |
| 6 | 41 | 42.688 s | S | 0.4941 | `[Q N] S [V N]` | V 0.4196 |
| 7 | 70 | 63.072 s | N | 0.5109 | `[N V] N [V V]` | V 0.4830 |
| 8 | 101 | 84.736 s | S | 0.5150 | `[N N] S [Q N]` | V 0.4681 |
| 9 | 57 | 56.240 s | V | 0.5253 | `[V N] V [N V]` | S 0.3619 |
| 10 | 117 | 99.888 s | V | 0.5306 | `[N N] V [N N]` | N 0.4135 |

Full probability vector for the lowest, beat 52:
`{V: 0.3760, N: 0.3737, S: 0.2503, F: 0.0000}` — a near-tie between V and N.

**What this means for the verdict:** 4 of the 10 lowest-confidence beats
(52, 27, 57, 117) were called **V**, and all 4 count toward the 38 that produce
CRITICAL. Beats 51, 68, 70 sit in a dense ectopy cluster at 52–63 s where the
model is visibly unstable — the three of them split V/S/N with runner-ups within
0.03–0.11. If beat 52 alone flipped to N, PVC burden would be 37/127 = 29.13%,
still above 20%; **the verdict is not sensitive to any single beat**, but it does
rest on a class whose weakest members are near coin-flips.

---

## 5. Beat timeline

Full sequence of 127 predicted labels, in detection order. Row labels are beat
indices.

```
  0- 19:  N V N N N N N V Q N S N V V N N N N V N
 20- 39:  V N N V Q N V V N V N Q N N V N N N N Q
 40- 59:  N S V N N V N V N S V S V S N V N V N V
 60- 79:  V N N N V Q S V N V N V V N V V N N N S
 80- 99:  N V V N N N N N N N V N V N N Q N N V N
100-119:  N S Q N N V V N N N S N N Q N N N V N N
120-126:  V V N N N V N
```

**Structural read of this sequence — all three pattern rules fell exactly one
unit short:**

| Pattern | Longest run measured | Rule threshold | Fired? |
|---|---|---|---|
| Consecutive `V` (VT run) | **2** | ≥3 | **NO** |
| Repeating `N,V` (bigeminy) | **3** | ≥4 | **NO** |
| Repeating `N,N,V` (trigeminy) | **2** | ≥3 | **NO** |

There are **8 separate V-pairs** and 22 isolated V beats. The ectopy is frequent
but never sustains — no triplet anywhere in 127 beats. This is exactly what the
`t_inspect_period` fix changed: pre-fix, this segment reported **3 VT runs**,
because double-counted T-waves manufactured consecutive V labels. Post-fix the
longest V run is 2.

The densest ectopy is beats 40–79 (t ≈ 42–67 s), which is also where every one of
the ten lowest-confidence beats except 27, 101 and 117 falls.

---

## 6. Rule engine trace

Every rule the cascade evaluates, fired or not.

| # | Rule | Measured value | Threshold | Fired | Would set |
|---|---|---|---|---|---|
| **1** | **PVC burden > CRITICAL threshold** | **29.921%** | **20.0%** | **YES** | **CRITICAL** |
| 2 | VT run count > 0 (run = ≥3 consecutive V) | 0 | 0 | no | CRITICAL |
| 3 | PVC burden > HIGH threshold | 29.921% | 10.0% | **YES** | HIGH |
| 4 | PAC burden > HIGH **OR** AFib burden > HIGH | PAC 7.087% / **AFib 100.0%** | PAC 15.0% / AFib 30.0% | **YES** (on AFib) | HIGH |
| 5 | Sustained HRV suppression: SDNN < threshold | 164.351 ms | 20.0 ms | no | MEDIUM |
| 6 | NEWS2 safety override ≥ critical | `null` — **NOT EVALUATED** | 7 | no | CRITICAL |
| 7 | qSOFA safety override ≥ high | `null` — **NOT EVALUATED** | 2 | no | HIGH |

**Supporting measurements behind rule 4** — 5 AFib windows examined, 5 flagged,
burden = 5/5 = 100.0%:

| Window | Beats | t range | RR CV |
|---|---|---|---|
| 1 | 0–19 | 15.152 – 25.544 s | 0.2994 |
| 2 | 20–39 | 25.968 – 41.800 s | 0.2741 |
| 3 | 40–59 | 42.272 – 57.496 s | 0.3200 |
| 4 | 60–79 | 57.896 – 67.432 s | 0.1785 |
| 5 | 80–99 | 67.936 – 83.728 s | 0.3513 |

Detector threshold: RR CV > **0.10** (`_afib_suspected`,
`ecg_pipeline_core.py:2029`). **⚠ The report text renders every one of these as
`> 0.15`** — a hardcoded literal at `agent_bridge.py:666` that no longer tracks
the code. See §10, CHECK D.

**Rules 6 and 7 were not evaluated.** `run_full_report()` calls
`ECGPipeline.run()` without `news2_score`/`qsofa_score`, so both overrides get
`None` and report `"evaluated": false` rather than being silently scored 0 — the
correct conservative behavior. But real vitals **were** available for this
segment (HR 115.8, RR 23.8, matched at 0 ms offset), giving NEWS2 = 4 and qSOFA
proxy = 1. I re-scored with those fed in: the level stays **CRITICAL**, same
deciding rule, since 4 < 7 and 1 < 2 and neither override can lower a level.

---

## 7. Risk verdict

| Field | Value |
|---|---|
| **FINAL RISK LEVEL** | **CRITICAL** |
| Assessable | true |
| Alert reason | `"PVC burden 29.9% > critical threshold (20.0%)"` |
| Confidence tier | MODERATE-HIGH |
| Confidence basis | Deciding rule depends on V-class counts; held-out V F1 = 0.826 |

### ★★★ DECIDING RULE ★★★

```
Rule 1 — "PVC burden > CRITICAL threshold"

    measured_value                  : 29.921   (38 V beats / 127 detected)
    threshold                       : 20.0
    fired                           : true
    would_set_level                 : CRITICAL
    is_deciding_rule                : true
    depends_on_low_confidence_class : false
```

Three rules fired (1, 3, 4). The cascade is `if/elif`, first-match-wins in strict
severity order, so rule 1 (CRITICAL) takes precedence over rules 3 and 4 (both
HIGH), which are correctly reported as fired-but-not-deciding.

**Did vitals change the verdict? No.** Neither override was evaluated in the
report path, and when fed in manually neither reached its threshold. ECG alone
had already hit the ceiling.

---

## 8. MedGemma input JSON

**This segment was CRITICAL, so no call was made.** Below is exactly what would
have been sent. The full prompt is **15,151 characters**; its embedded structured
payload is reproduced verbatim here (regenerated via
`build_transparency_prompt(report_json)`).

```json
{
  "recording": {
    "source": "vitalpatch",
    "patient_id": "184B27",
    "segment_id": "1780171482351_VC2B008BF_184B27_ecg_seg1",
    "duration_s": 115.44,
    "n_raw_samples": 14359,
    "fs_nominal_hz": 125.0,
    "fs_processed_hz": 125.0,
    "quality_score": 0.6087,
    "sqi_window_rejection_rate": 0.3913,
    "n_beats_detected": 127,
    "n_beats_analyzed": 119,
    "n_beats_flagged_low_quality": 8
  },
  "beat_summary": {
    "N": {"count": 72, "pct_of_analyzed_beats": 56.69, "confidence": "HIGH"},
    "S": {"count": 9,  "pct_of_analyzed_beats": 7.09,  "confidence": "LOW"},
    "V": {"count": 38, "pct_of_analyzed_beats": 29.92, "confidence": "MODERATE-HIGH"},
    "F": {"count": 0,  "pct_of_analyzed_beats": 0.0,   "confidence": "LOW"},
    "Q": {"count": 8,  "pct_of_analyzed_beats": 6.3,   "confidence": "N/A"}
  },
  "rhythm_findings": [
    {"kind": "AFIB_SUSPECTED", "start_beat_idx": 0,  "end_beat_idx": 19, "start_time_s": 15.152, "duration_s": 10.392, "evidence": {"rr_cv": 0.2994186311938437}},
    {"kind": "AFIB_SUSPECTED", "start_beat_idx": 20, "end_beat_idx": 39, "start_time_s": 25.968, "duration_s": 15.832, "evidence": {"rr_cv": 0.27408409540332596}},
    {"kind": "AFIB_SUSPECTED", "start_beat_idx": 40, "end_beat_idx": 59, "start_time_s": 42.272, "duration_s": 15.224, "evidence": {"rr_cv": 0.3199702729566171}},
    {"kind": "AFIB_SUSPECTED", "start_beat_idx": 60, "end_beat_idx": 79, "start_time_s": 57.896, "duration_s": 9.536,  "evidence": {"rr_cv": 0.17849616149015665}},
    {"kind": "AFIB_SUSPECTED", "start_beat_idx": 80, "end_beat_idx": 99, "start_time_s": 67.936, "duration_s": 15.792, "evidence": {"rr_cv": 0.3513061291454986}}
  ],
  "assessable": true,
  "risk_level": "CRITICAL",
  "rule_trace": [
    {"condition": "PVC burden > CRITICAL threshold", "measured_value": 29.921, "threshold": 20.0, "fired": true, "would_set_level": "CRITICAL", "depends_on_low_confidence_class": false, "is_deciding_rule": true},
    {"condition": "VT run count > 0 (a run = >=3 consecutive V beats)", "measured_value": 0, "threshold": 0, "fired": false, "would_set_level": "CRITICAL", "depends_on_low_confidence_class": false, "is_deciding_rule": false},
    {"condition": "PVC burden > HIGH threshold", "measured_value": 29.921, "threshold": 10.0, "fired": true, "would_set_level": "HIGH", "depends_on_low_confidence_class": false, "is_deciding_rule": false},
    {"condition": "PAC burden > HIGH threshold OR AFib burden > HIGH threshold", "measured_value": {"pac_burden_pct": 7.087, "afib_burden_pct": 100.0}, "threshold": {"pac_burden_pct": 15.0, "afib_burden_pct": 30.0}, "fired": true, "would_set_level": "HIGH", "depends_on_low_confidence_class": false, "is_deciding_rule": false},
    {"condition": "Sustained HRV suppression: SDNN < threshold", "measured_value": 164.351, "threshold": 20.0, "fired": false, "would_set_level": "MEDIUM", "depends_on_low_confidence_class": false, "is_deciding_rule": false},
    {"condition": "NEWS2 safety override (>= critical threshold)", "measured_value": null, "threshold": 7, "fired": false, "would_set_level": "CRITICAL", "depends_on_low_confidence_class": false, "evaluated": false, "is_deciding_rule": false},
    {"condition": "qSOFA safety override (>= high threshold)", "measured_value": null, "threshold": 2, "fired": false, "would_set_level": "HIGH", "depends_on_low_confidence_class": false, "evaluated": false, "is_deciding_rule": false}
  ],
  "deciding_rule": {
    "condition": "PVC burden > CRITICAL threshold",
    "measured_value": 29.921,
    "threshold": 20.0,
    "fired": true,
    "would_set_level": "CRITICAL",
    "depends_on_low_confidence_class": false,
    "is_deciding_rule": true
  },
  "confidence": {
    "tier": "MODERATE-HIGH",
    "statement": "The deciding rule depends on V (ventricular) beat counts. The production classifier's held-out V-class F1 is 0.826 -- the most reliable class this model produces, though still not a clinical-grade guarantee.",
    "caveat": "This is a proxy/heuristic confidence assessment produced by this reporting layer, not a statistically calibrated probability (no conformal calibration set has been loaded for this run) and not a clinical guarantee."
  },
  "safety_overrides": [
    {"override": "NEWS2", "applied": false, "note": "No NEWS2 score was supplied to this run -- override not evaluated."},
    {"override": "qSOFA", "applied": false, "note": "No qSOFA score was supplied to this run -- override not evaluated."}
  ],
  "known_limitations": [
    "Beat classifier is the frozen production XGBoost model (five_class_xgb.json). DS2 held-out per-class F1: V=0.826 (reliable), S=0.139 (LOW-CONFIDENCE), F=0.011 (LOW-CONFIDENCE, essentially unsolved).",
    "NEWS2/qSOFA vitals pairing is not wired into this ECG-only bridge -- safety_overrides above are reported as not-evaluated rather than fabricated."
  ],
  "final_risk_level": "CRITICAL"
}
```

The prompt wrapping this JSON instructs the model that the risk level is
already decided, that it may only escalate and never lower, and that it must copy
each rule's pre-computed `EXCEEDS` / `does NOT exceed` verdict verbatim rather
than performing its own comparison.

---

## 9. MedGemma output

**Status: `SKIPPED_CRITICAL`. No LLM output exists.**

| Field | Value |
|---|---|
| Skip reason | Risk level is CRITICAL |
| Code path | `render_narrative()`, `agent_bridge.py:1120-1126` — returns before `call_medgemma()` |
| Audit event logged | `TRANSPARENCY_MEDGEMMA_SKIPPED_CRITICAL` |
| Logged rationale | *"CRITICAL level bypasses MedGemma entirely, per safety constraint"* |
| Endpoint that would have been used | `http://localhost:11434/api/generate` |
| Model | `medgemma` |
| Timeout vs cold start | **N/A — no call made.** For reference (`SYSTEM_AUDIT.md` §0/§5): the original `ecg_inference/report.py` timeout was **10 s** against a **measured 75.6 s cold start** (7.5× over); since raised to `MEDGEMMA_TIMEOUT_S = 90.0`. The MedGemma-Agent call site uses 120 s. Neither applied here. |

This is a deliberate safety design: a CRITICAL verdict must not be softened or
reworded by an LLM.

The narrative in the report is the **deterministic local fallback**
(`_deterministic_narrative()`), not LLM-generated:

```
*** CRITICAL -- IMMEDIATE CLINICIAN REVIEW REQUIRED ***
Risk level: CRITICAL.
Deciding rule: PVC burden > CRITICAL threshold: measured 29.921 vs threshold 20.0 -> EXCEEDED.
Other rules that also fired: PVC burden > HIGH threshold (29.921 vs 10.0); PAC burden > HIGH
threshold OR AFib burden > HIGH threshold ({'pac_burden_pct': 7.087, 'afib_burden_pct': 100.0}
vs {'pac_burden_pct': 15.0, 'afib_burden_pct': 30.0})
Confidence (MODERATE-HIGH): The deciding rule depends on V (ventricular) beat counts. The
production classifier's held-out V-class F1 is 0.826 -- the most reliable class this model
produces, though still not a clinical-grade guarantee.
Limitation: Beat classifier is the frozen production XGBoost model (five_class_xgb.json).
DS2 held-out per-class F1: V=0.826 (reliable), S=0.139 (LOW-CONFIDENCE), F=0.011
(LOW-CONFIDENCE, essentially unsolved). Any finding driven primarily by S or F counts is a
screening flag, not a diagnosis.
Limitation: NEWS2/qSOFA vitals pairing is not wired into this ECG-only bridge -- safety_overrides
above are reported as not-evaluated rather than fabricated.
Clinician review suggested -- this is decision support, not a diagnosis.
```

**Unsupported-statement check:** every number in it (29.921, 20.0, 10.0, 7.087,
100.0, 15.0, 30.0, 0.826, 0.139, 0.011) traces directly to the rule trace or the
frozen model card. **No unsupported claims.** Expected, since it is
template-generated from the trace.

---

## 10. Self-consistency checks

| # | Check | Result |
|---|---|---|
| 1 | V_count / total_detected × 100 = reported PVC burden | **PASS** |
| 2 | Implied HR from RR intervals matches vitals HR | **FAIL** |
| 3 | Risk verdict follows from deciding rule | **PASS** |
| 4 | SDNN is not the 0.0 sentinel | **PASS** |
| 5 | Beat counts sum to total detected | **PASS** |
| D | (additional) Internal contradictions | **3 DEFECTS FOUND** |

### CHECK 1 — PVC burden formula → **PASS**

```
38 / 127 × 100 = 29.92125984251968…%   →  reported 29.921   ✓ exact
 9 / 127 × 100 =  7.08661417322835…%   →  reported  7.087   ✓ exact
```

Qualification (not an arithmetic failure): the denominator is **detected** beats
(127, including 8 `Q`), not **analyzed** (119, which would give 31.93%). The
verdict is unchanged either way. But the field is **named**
`pct_of_analyzed_beats` while dividing by the detected count — a labeling defect
carried into CHECK D.

### CHECK 2 — Implied HR vs vitals HR → **FAIL**

Vitals HR **115.8 bpm**. Implied HR by the natural reading (beat count ÷ elapsed
time, equivalently 60000 ÷ mean of all RR) = **80.14 bpm**.

**Δ = −35.66 bpm, seven times the 5 bpm tolerance. FAIL.**

| Method | HR | Δ | Verdict |
|---|---|---|---|
| Beat count ÷ beat span | 80.14 | −35.66 | **FAIL** |
| Beat count ÷ full duration | 66.01 | −49.79 | **FAIL** |
| Mean of all 126 RR | 80.14 | −35.66 | **FAIL** |
| Mean of 106 non-flagged RR | 110.74 | −5.06 | **FAIL** (marginal) |
| Median of 106 non-flagged RR | 118.11 | +2.31 | PASS |

**Root cause: not a detector failure.** 39% of the signal was SQI-gated,
producing five stretches of 5.3–7.8 s with no detectable beats. Those inflate
mean RR without representing slow beating. Excluding them puts HR at
110.7–118.1 bpm, bracketing the device's 115.8.

**But this is a real reporting hazard, not a benign artifact.** The pipeline
emits **no heart-rate field at all** — not mean, min, or max. Any consumer
computing HR the obvious way gets **80 bpm for a patient the device measures at
116 bpm**. On a segment with 39% signal loss that gap is large enough to change
clinical impression, and nothing in the output warns about it.

### CHECK 3 — Risk verdict follows from deciding rule → **PASS**

Rules fired: 1 (PVC > 20% → CRITICAL), 3 (PVC > 10% → HIGH), 4 (AFib 100% > 30%
→ HIGH). Rules 2 and 5 did not fire; 6 and 7 not evaluated.
The cascade is first-match-wins in severity order; rule 1 is the
highest-precedence rule that fired; `deciding_rule` points at rule 1;
`final_risk_level` is CRITICAL. Consistent end to end. Rules 3 and 4 are correctly
marked fired-but-not-deciding.

### CHECK 4 — SDNN is not the 0.0 sentinel → **PASS**

**SDNN = 164.351 ms**, computed from **106 valid RR intervals** — far above the
3-interval minimum, so the sentinel branch was never reached. Post-`e0da1f7` that
branch returns `None` (not `0.0`), and `score_recording` guards with
`sdnn_ms is not None and sdnn_ms < threshold`.

Supporting HRV: RMSSD 260.811 ms, pNN50 80.952%, LF/HF 0.1255.

Physiologically coherent: 164 ms SDNN is far *above* normal sinus (~50 ms), which
is exactly what 29.9% ectopy plus RR CV of 0.18–0.35 should produce. RMSSD > SDNN
confirms the variability is beat-to-beat (ectopy-driven), not slow drift. The
suppression rule correctly did **not** fire.

*Precision caveat:* because `rr_flagged` is per-beat, 8 physiologically normal
intervals were excluded alongside the 12 genuinely abnormal ones — SDNN uses 106
of 126 intervals. Bias direction is not obvious and is undocumented.

### CHECK 5 — Beat counts sum to total detected → **PASS**

```
N 72 + S 9 + V 38 + F 0 + Q 8 = 127 = n_beats_detected   ✓
127 − 8 (Q) = 119 = n_beats_analyzed                     ✓
```

Also verified: 127 unique R-peak indices, **0 duplicates**, 0 intervals at
RR = 0 ms, 0 intervals below 200 ms (minimum exactly 200.00 ms — the refractory
guard holding at its configured floor).

### CHECK D — Internal contradictions → **3 REAL DEFECTS**

1. **AFib evidence text states the wrong threshold.** All five findings print
   `> 0.15`; the rule fired on **0.10** (`agent_bridge.py:666` hardcodes the
   literal; `ecg_pipeline_core.py:2029` sets `cv_threshold=0.10`). The lowest CV
   here is 0.178, so each printed statement is *coincidentally* true — but
   clinicians are being shown the wrong decision boundary. A window at CV 0.12
   would print the flatly self-contradictory `"0.120 > 0.15 … fired=True"`.

2. **`pct_of_analyzed_beats` divides by detected beats.** The same JSON reports
   `n_beats_analyzed: 119` and then `"Q": {"count": 8, "pct_of_analyzed_beats": 6.3}`.
   `Q` beats are by definition excluded from analysis; they cannot be 6.3% of the
   analyzed set. Divisor is 127.

3. **NEWS2/qSOFA reported "not evaluated" while real vitals existed.** The report
   states *"vitals pairing is not wired into this ECG-only bridge"*, yet matched
   vitals were available and loadable (HR 115.8, RR 23.8, 0 ms offset). Not
   fabricating them is correct; but a reader could reasonably conclude no vitals
   existed. They did.

**No case was found of a rule firing that the verdict ignores, or of a verdict
asserting something no rule supports.**

---

## 11. Pipeline timing per stage

Wall-clock, `time.perf_counter()`, single run, 10-vCPU host.

| Stage | Function | Time (ms) | % of stages 1–8 |
|---|---|---|---|
| 1. Ingest / parse | `parse_vitalpatch_ecg` (whole file, both segments) | 19.2 | 3.2% |
| 2. SQI gate | `run_sqi_gate` | 87.9 | 14.7% |
| 3. Resample | `to_target_rate` (125 → 125 Hz, pass-through) | 0.0 | <0.1% |
| 4. Filter chain | `apply_filter_chain` | 21.2 | 3.5% |
| 5. R-peak detect + beat segment | `detect_and_segment` (XQRS + snap + guard) | **234.6** | **39.3%** |
| 6a. Feature extraction | `batch_feature_matrix` (119 × 56) | 37.3 | 6.2% |
| 6b. HRV | `recording_level_hrv` | 13.2 | 2.2% |
| 7a. Beat classification | `predict_one` loop, 119 beats | **183.2** | **30.7%** |
| 7b. Rhythm analysis | `RhythmContextEngine.analyze` | 0.7 | 0.1% |
| 8. Rule engine + risk scoring | `score_recording` | 0.1 | <0.1% |
| **Subtotal, stages 1–8** | | **597.4** | 100% |
| — | Vitals load + match (`load_real_vitals`, cold index) | 863.9 | — |
| 9. Report generation (full re-run) | `run_full_report` end-to-end | 486.0 | — |
| **MedGemma call** | — | **0 (SKIPPED_CRITICAL)** | — |
| **Total incl. vitals matching** | | **1,461.3** | — |

Honest notes:

- **Stages 5 and 7a are 70% of compute** (417.8 ms combined), matching
  `SYSTEM_AUDIT.md` §4's identification of R-peak detection and the per-beat
  classifier loop as the two bottlenecks.
- **This run used the per-beat LOOP classifier**, not the batched version.
  `ecg_pipeline_core.py:2662` calls `predict_one()` once per beat. The batched
  fix was applied to `ecg_inference/classifier.py`'s `predict_batch()`, which
  **this path does not import**. Measured here: 183.2 ms for 119 beats
  (~1.54 ms/beat). At the audit's measured 153× ratio the batched equivalent
  would be ~1–2 ms.
- **The 863.9 ms vitals load is a cold-cache figure.** `_build_vitals_index()`
  reads every vitals file for the patient once, then caches per patient
  directory; subsequent segments for `184B27` cost near zero. Not a per-segment
  cost in a batch run.
- **Beat segmentation is not separately timable** — `detect_and_segment` fuses
  detection and segmentation into one call, so 234.6 ms covers both.
- **The 486.0 ms `run_full_report` figure overlaps rather than adds to** the
  per-stage numbers — it re-runs stages 1–9 from scratch. It is faster than the
  instrumented sum because the vitals index is warm and NumPy caches are hot on
  the second pass.

---

## Final check summary

| Check | Result |
|---|---|
| 1 — PVC burden formula matches beat counts | **PASS** |
| 2 — Implied HR matches vitals HR within 5 bpm | **FAIL** (−35.66 bpm) |
| 3 — Risk verdict follows deciding rule | **PASS** |
| 4 — SDNN is not the 0.0 sentinel | **PASS** |
| 5 — Beat counts sum to total detected | **PASS** |
| D — No internal contradictions | **FAIL** (3 defects) |

**Failures are stated, not hidden:** one HR-consistency failure with an
identified root cause and an unaddressed reporting gap (no HR field exists), and
three internal-contradiction defects (AFib threshold text, `pct_of_analyzed_beats`
denominator, vitals-available-but-unwired).
