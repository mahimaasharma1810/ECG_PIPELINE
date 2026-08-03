# Full Diagnostic Report — one real VitalPatch segment

**Generated:** 2026-08-02
**Produced by:** re-running the existing (post-fix) pipeline on one real segment.
No pipeline code was modified to produce this report. Numbers come from a
live re-run via `ecg_pipeline.agent_bridge.run_full_report()` at HEAD `e0da1f7`,
plus `load_real_vitals()` / `compute_partial_news2()` / `compute_qsofa_proxy()`.

## Why this segment

Selected from the **15 post-fix CRITICAL segments** in
`data/reports/clinician_review_sample.csv` (the full 2,889-segment corpus *was*
re-scored after the `t_inspect_period` fix — see "Correction" below). Among
those 15, this one is the richest single case:

- highest PVC burden of any post-fix CRITICAL segment with a substantial beat
  count (29.92%, 127 beats)
- the only segment in the corpus whose agent-side qSOFA reached 2
- NEWS2 = 4, the joint-highest among the 15
- both HR and respiratory rate really present in the paired vitals file
- **verdict-relevant change after the fix**: pre-fix it was scored with 149
  analyzed beats and 3 VT runs; post-fix it is 119 analyzed beats and **0 VT
  runs**. It stayed CRITICAL, but for a different reason (PVC burden, not VT).

### Correction to the brief

The brief stated "t_inspect_period fix validated on 50 segments and synthetic
suite — full corpus not yet re-run." That is **out of date**.
`docs/audits/AUDIT_2026-07-31.md` §7 records a full 2,889-segment re-score (scoring
only, no report regeneration), 0 errors. CRITICAL went 290 (10.0%) → 15 (0.5%).
What has *not* been re-run is **report regeneration** — the saved `.json`/`.md`
reports on disk under `data/reports/vitalpatch/` are still pre-fix (mtime
2026-07-26). This report is a fresh post-fix run, so its numbers do not match
the stale saved report for this segment.

---

## SECTION 1 — RECORDING METADATA

| Field | Value |
|---|---|
| Patient ID | `184B27` |
| Segment ID | `1780171482351_VC2B008BF_184B27_ecg_seg1` |
| Source file | `data/raw/vitalpatch/Patch_184B27/1780171482351_VC2B008BF_184B27_ecg.csv` |
| Recording duration | 115.44 s |
| Raw samples | 14,359 |
| Sampling rate (nominal) | 125.0 Hz |
| Sampling rate (processed) | 125.0 Hz (no rate change) |
| Device | VitalPatch, device serial `VC2B008BF`, patch `184B27` |
| Lead count | **1** (single-lead; CSV is flat `timestamp,value` pairs, one channel) |
| Segment start (epoch ms) | 1780171482788 → **2026-05-30T20:04:42.788Z** |
| Segment end (epoch ms) | 1780171598228 → **2026-05-30T20:06:38.228Z** |
| Segments in this file | 2 (`_seg0`, `_seg1`); this is seg1 |
| Data gaps flagged | 0 within the segment |
| Pipeline | `ecg_pipeline/ecg_pipeline_core.py` stages 1–9 via `agent_bridge.run_full_report()` |
| Git commit | `e0da1f7`, branch `mod_1` |
| Classifier | `ecg_pipeline/models/five_class_xgb.json` (frozen production XGBoost) |
| **Fix applied — `t_inspect_period`** | **`0.36`** (`detect_r_peaks`, `ecg_pipeline_core.py:991`, `wp.XQRS.Conf(t_inspect_period=0.36)`) |
| Companion fix — refractory guard | `_apply_refractory_guard(..., refractory_s=0.2)` (200 ms), `ecg_pipeline_core.py:1171` |

Note: the file's *first* timestamp is 1780169852460 (20:37 earlier); seg1 begins
after a large inter-segment gap, which is why the segment splitter produced two
segments from one file.

---

## SECTION 2 — SIGNAL QUALITY

Stage-2 SQI gate, 5.0 s windows, run **before** any filtering.

| Metric | Value |
|---|---|
| SQI windows total | 23 |
| Windows passed | 14 |
| Windows rejected | 9 |
| **SQI window rejection rate** | **0.3913 (39.13%)** |
| **SQI / quality score** | **0.6087** (defined as 1 − rejection rate) |
| Samples kept | 8,750 / 14,359 |
| **Usable signal percentage** | **60.94%** |

Reject-code breakdown (9 rejected windows):

| Reject code | Count |
|---|---|
| `NO_QRS_IMPULSE_CHARACTER` (kurtosis < 1.5) | 6 |
| `FLATLINE` (≥400 ms stuck run, frac > 0.05) | 2 |
| `BASELINE_WANDER` (ratio > 1.2) | 1 |

Per-window measured ranges across all 23 windows:

| Metric | min | max | mean | threshold |
|---|---|---|---|---|
| `flatline_frac` | 0.0000 | 0.1184 | 0.0095 | > 0.05 rejects |
| `clipping_frac` | 0.0000 | **0.0000** | 0.0000 | > 0.02 rejects |
| `missing_frac` | 0.0000 | 0.0000 | 0.0000 | > 0.03 rejects |
| `morphology_kurtosis` | −0.3047 | 7.8573 | 2.3564 | < 1.5 rejects |
| `baseline_wander_ratio` | 0.2957 | 3.0998 | 0.8526 | > 1.2 rejects |
| `snr_db` | 6.9954 | 21.6874 | 13.6334 | < 5.0 rejects |

Direct answers:

- **Baseline wander detected: YES** — 1 window rejected for it; max ratio 3.0998 vs threshold 1.2.
- **Flatline detected: YES** — 2 windows rejected; max flatline fraction 0.1184 vs threshold 0.05.
- **Clipping detected: NO** — `clipping_frac` is exactly 0.0000 in every one of the 23 windows.
- **Noise percentage** — the pipeline has no single "noise %" metric. The closest real numbers are the 39.13% window rejection rate and the 39.06% of samples discarded. Minimum window SNR was 6.99 dB (above the 5.0 dB floor, so no window failed on SNR alone).

**Overall quality verdict: MARGINAL — assessable but degraded.** 39% of the
recording was gated out. The segment passes the assessability bar (119 beats
analyzed, well above the 5-beat floor), but the discarded 39% is concentrated
enough to leave five multi-second stretches with no detected beats at all (see
Section 4), which is the root cause of the HR mismatch in Section 12.

---

## SECTION 3 — PREPROCESSING

Filters applied, in order (`apply_filter_chain`, `ecg_pipeline_core.py:906`):

| # | Filter | Type | Cutoff / parameter | Order |
|---|---|---|---|---|
| 1 | `remove_baseline_median` | Rolling-median baseline removal | 200.0 ms window | n/a (median) |
| 2 | `highpass_residual` | High-pass | 0.5 Hz | Butterworth |
| 3 | `powerline_notch` | IIR notch | 50.0 Hz | notch |
| 4 | `bandpass` | Band-pass | 0.5 – 40.0 Hz | Butterworth |
| 5 | `emg_suppress_kalman` | Kalman EMG suppression | fixed `process_var`/`meas_var` | n/a |

- **Polarity correction applied: NO.** There is no polarity-correction stage in
  this pipeline. Polarity is handled implicitly: `_snap_to_local_peak` selects on
  `abs()` amplitude, so negative-going QRS (as on patient 184B27) is detected
  without inversion. Documented as a deliberate choice in `docs/audits/AUDIT_2026-07-31.md` §7.
- **Resampling: NO net change.** Stage 3 ran (`to_target_rate`, 14,359 → 14,359
  samples) but input and output rates are both 125.0 Hz, so this was a pass-through.
- **Segments dropped: 0.** Both segments in the file were parsed; this run
  processed seg1. No sub-segment was dropped. What *was* dropped is 5,609 of
  14,359 samples (39.06%), NaN-masked by the SQI gate then linearly
  interpolated across before filtering (`np.interp` over valid samples).
- **Kalman smoothing applied: YES — and for this source it is NOT display-only.**
  This is a real subtlety worth stating precisely:
  - For `source="vitalpatch"` (this segment), R-peak **detection runs on the
    full Kalman-filtered signal** (`detection_signal = filtered`,
    `ecg_pipeline_core.py:2626`). Kalman is load-bearing here — the code
    documents that skipping it causes XQRS over-detection on ADC-scale signals.
  - For `source="wfdb"` only, detection uses a Kalman-skipped signal.
  - The **display-only Kalman skip** (commit `e0da1f7`, `generate_clinician_report.py`'s
    `recompute_filtered_signal`) applies **only to the waveform PNG panels**, not
    to detection or features. No PNG was generated in this run, so that path did
    not execute.

---

## SECTION 4 — R-PEAK DETECTION

| Field | Value |
|---|---|
| Detector | WFDB `XQRS` adaptive-threshold (`wfdb.processing.XQRS`) |
| Detector config | `XQRS.Conf(t_inspect_period=0.36)` |
| **`t_inspect_period` used** | **0.36 s** |
| Post-detection guard | `_apply_refractory_guard`, 200 ms minimum peak spacing |
| Snap-to-local-peak radius | 15 samples (non-wfdb sources) |
| **Total R-peaks detected** | **127** |
| **Beats analyzed** | **119** |
| Beats rejected by beat-level SQI | 8 |
| RR intervals total | 126 |
| RR intervals flagged out-of-range | 20 |
| RR intervals used for HRV/AFib | 106 |

**Beats analyzed vs detected — why they differ:** all 127 detected R-peaks are
segmented into beats. 8 are then rejected by the *beat-level* quality check
`_beat_level_sqi` (a per-beat check distinct from the window-level SQI gate) and
labeled `Q` instead of being classified. Measured reasons for the 8:

| Reject reason | Count | Beat times (s) |
|---|---|---|
| `R_PEAK_NOT_LOCAL_MAX` | 6 | 19.280, 37.144, 41.800, 60.528, 81.304, 92.520 |
| `EXCESS_BASELINE_DRIFT` | 2 | 27.856, 97.576 |

(Times listed are the 8 rejected beats in ascending order; the two reason groups
are interleaved in time.)

### RR intervals

| Metric | All RR (n=126) | Non-flagged RR (n=106) |
|---|---|---|
| Mean | **748.70 ms** | **541.81 ms** |
| Median | — | 508.00 ms |
| Min | **200.00 ms** | 304.00 ms |
| Max | **7784.00 ms** | 1104.00 ms |

The 20 flagged RR values, in full:
`200, 232, 280, 280, 288, 288, 312, 448, 456, 504, 512, 552, 584, 640, 648, 5328, 5728, 5832, 6008, 7784` ms.

Two things to flag here honestly:

1. **The five multi-second intervals (5328–7784 ms) are missed-beat gaps**, not
   real pauses. They line up with the SQI-rejected stretches. The heart did not
   stop for 7.8 seconds; the detector had no usable signal there.
2. **8 of the 20 "flagged" values are physiologically normal** (448, 456, 504,
   512, 552, 584, 640, 648 ms — all inside the 300–2000 ms configured range).
   They are flagged because `segment_beats` sets `rr_flagged` on a **beat** if
   *either* its `rr_pre` **or** its `rr_post` is out of range
   (`ecg_pipeline_core.py:1050-1053`). Downstream, `recording_level_hrv` and the
   AFib rule then discard that beat's `rr_post` too. So a single bad interval
   silently costs two intervals of HRV data. This is conservative rather than
   dangerous, but it is undocumented and it materially shrinks the HRV sample
   (126 → 106).

### Implied heart rate — five ways, because they disagree

| Method | Value | vs vitals HR (115.8) |
|---|---|---|
| Beat count over beat span (126 intervals / 94.336 s) | **80.14 bpm** | −35.66 |
| Beat count over full duration (127 / 115.44 s) | 66.01 bpm | −49.79 |
| 60000 / mean of all RR (748.70 ms) | 80.14 bpm | −35.66 |
| 60000 / mean of non-flagged RR (541.81 ms) | **110.74 bpm** | −5.06 |
| 60000 / median of non-flagged RR (508.00 ms) | **118.11 bpm** | **+2.31** |

Beat span: first R-peak 15.152 s, last R-peak 109.488 s (94.336 s of the 115.44 s
recording). Instantaneous HR from non-flagged RR ranges 54.3 – 197.4 bpm.

- **Vital signs HR for cross-check: 115.8 bpm** (VitalPatch col 1, 18 readings in
  the ±2 min window, range 113–119 bpm).
- **Mismatch flag (>5 bpm): FLAGGED.** The headline detected-HR figure (80.14 bpm)
  is **35.66 bpm below** the device's own HR. See CHECK B for the resolution — this
  is a coverage artifact, not a detector failure, but it is a real reporting hazard.

### Duplicate indices

- **Duplicate R-peak indices: 0** (127 detected, 127 unique).
- **RR = 0 ms intervals: 0.**
- **RR < 200 ms intervals: 0.** The minimum RR is exactly 200.00 ms, i.e. the
  refractory guard is holding precisely at its configured 200 ms boundary.

### First 20 R-peak timestamps (seconds from segment start)

| # | t (s) | RR-post (ms) | RR flagged | Q-rejected |
|---|---|---|---|---|
| 0 | 15.152 | 288.0 | yes | no |
| 1 | 15.440 | 648.0 | yes | no |
| 2 | 16.088 | 440.0 | no | no |
| 3 | 16.528 | 608.0 | no | no |
| 4 | 17.136 | 512.0 | no | no |
| 5 | 17.648 | 584.0 | no | no |
| 6 | 18.232 | 536.0 | no | no |
| 7 | 18.768 | 512.0 | no | no |
| 8 | 19.280 | 480.0 | no | **yes** |
| 9 | 19.760 | 456.0 | no | no |
| 10 | 20.216 | 616.0 | no | no |
| 11 | 20.832 | 504.0 | no | no |
| 12 | 21.336 | 416.0 | no | no |
| 13 | 21.752 | 1104.0 | no | no |
| 14 | 22.856 | 472.0 | no | no |
| 15 | 23.328 | 504.0 | no | no |
| 16 | 23.832 | 536.0 | no | no |
| 17 | 24.368 | 320.0 | no | no |
| 18 | 24.688 | 856.0 | no | no |
| 19 | 25.544 | 424.0 | no | no |

Note the recording's first 15.152 s produced no detected beats — that stretch was
SQI-rejected.

---

## SECTION 5 — BEAT CLASSIFICATION

Classifier: frozen production XGBoost, `five_class_xgb.json`, 56-dimensional
feature vector, `is_trained=True`, source `trained_model` for all 119 analyzed
beats (no rule-based fallback was used).

### Total beats by class

| Class | Count | % of 127 detected | % of 119 analyzed |
|---|---|---|---|
| N | 72 | 56.69% | 60.50% |
| S | 9 | 7.09% | 7.56% |
| V | **38** | **29.92%** | 31.93% |
| F | 0 | 0.00% | 0.00% |
| Q | 8 | 6.30% | n/a (Q = rejected, not classified) |
| **Total** | **127** | 100.00% | — |

### Burden percentages

**PVC burden = V_count / total_detected × 100 = 38 / 127 × 100 = 29.9213%**
→ reported as **29.921%**. Formula confirmed exactly.

**PAC burden = S_count / total_detected × 100 = 9 / 127 × 100 = 7.0866%**
→ reported as **7.087%**. Confirmed exactly.

> **Denominator warning.** `score_recording` uses `n = len(labels)` = **127**,
> i.e. *all detected beats including the 8 unclassifiable `Q` beats*, not the 119
> analyzed. Using the analyzed count would give PVC 31.93%. Both are above the
> 20% CRITICAL threshold here so the verdict is unaffected, but the report's own
> `beat_summary` field is **mislabeled**: it calls this
> `pct_of_analyzed_beats` (`agent_bridge.py:628`) while dividing by the detected
> count. The same block reports "Q: 8 beats (6.3% of analyzed beats)", which is
> self-contradictory — Q beats are by definition the ones *not* analyzed.

### Confidence distribution (119 classified beats)

| Metric | Value |
|---|---|
| Mean confidence | **0.8479** |
| Median confidence | 0.9259 |
| Min | 0.3760 |
| Max | 1.0000 |
| **Beats with confidence ≥ 0.95** | **50** (42.0%) |
| **Beats with confidence < 0.70** | **25** (21.0%) |

By class:

| Class | n | mean | min | max |
|---|---|---|---|---|
| N | 72 | 0.8910 | 0.3983 | 1.0000 |
| S | 9 | **0.5709** | 0.3879 | 0.8020 |
| V | 38 | 0.8318 | 0.3760 | 0.9999 |

The S class averages 0.57 confidence — barely above chance for a 4-way choice,
consistent with its held-out F1 of 0.139.

**Lowest-confidence beat:** index 52, **t = 53.240 s**, predicted **V**,
confidence **0.3760**. Full probability vector:
`{V: 0.3760, N: 0.3737, S: 0.2503, F: 0.0000}` — V beat N by 0.0023. This beat
is counted in the 38 that produce the CRITICAL verdict.

### Top 20 individual beat predictions

| # | t (s) | Predicted | Confidence | Source |
|---|---|---|---|---|
| 0 | 15.152 | N | 0.8492 | trained_model |
| 1 | 15.440 | V | 0.9258 | trained_model |
| 2 | 16.088 | N | 0.9947 | trained_model |
| 3 | 16.528 | N | 0.6502 | trained_model |
| 4 | 17.136 | N | 0.9977 | trained_model |
| 5 | 17.648 | N | 0.6645 | trained_model |
| 6 | 18.232 | N | 0.5976 | trained_model |
| 7 | 18.768 | V | 0.5773 | trained_model |
| 8 | 19.280 | **Q** | n/a | quality_gate |
| 9 | 19.760 | N | 0.8953 | trained_model |
| 10 | 20.216 | S | 0.5804 | trained_model |
| 11 | 20.832 | N | 0.9611 | trained_model |
| 12 | 21.336 | V | 0.9323 | trained_model |
| 13 | 21.752 | V | 0.9990 | trained_model |
| 14 | 22.856 | N | 1.0000 | trained_model |
| 15 | 23.328 | N | 0.7185 | trained_model |
| 16 | 23.832 | N | 0.8777 | trained_model |
| 17 | 24.368 | N | 0.9949 | trained_model |
| 18 | 24.688 | V | 0.9781 | trained_model |
| 19 | 25.544 | N | 0.9989 | trained_model |

**Classifier mode: per-beat loop.** This run went through
`ecg_pipeline_core.py:2662`, `self.classifier.predict_one(...)` once per beat.
See Section 11.

---

## SECTION 6 — RHYTHM ANALYSIS

### Heart rate

The pipeline does **not** emit mean/min/max HR as fields. Derived from the RR
series in this run (non-flagged intervals, n=106):

| Metric | Value |
|---|---|
| Mean HR (from mean RR 541.81 ms) | 110.74 bpm |
| Median HR (from median RR 508.00 ms) | 118.11 bpm |
| Min instantaneous HR (from max RR 1104 ms) | 54.3 bpm |
| Max instantaneous HR (from min RR 304 ms) | 197.4 bpm |

Including flagged intervals the mean drops to 80.14 bpm — see Section 4.

### HRV

| Metric | Value |
|---|---|
| **SDNN** | **164.351 ms** |
| **RMSSD** | **260.811 ms** |
| pNN50 | 80.952% |
| LF/HF ratio | 0.1255 |
| QRS width trend | −0.0341 |

**Sentinel check: NO — SDNN was NOT the 0.0 sentinel.** 106 valid RR intervals
were available, far above the 3-interval floor, so `recording_level_hrv`
returned a genuine measurement. The post-`e0da1f7` code returns `sdnn_ms=None`
(not `0.0`) in the too-little-data case; that branch was not taken here.

SDNN 164 ms and RMSSD 261 ms are **very high**, not suppressed — consistent with
a genuinely chaotic rhythm (29.9% ectopy plus AFib-range RR variability), and the
reason the HRV-suppression rule did not fire.

### Rhythm findings — 5 total, all one kind

| Kind | Beats | t start (s) | t end (s) | Evidence (RR CV) |
|---|---|---|---|---|
| AFIB_SUSPECTED | 0–19 | 15.152 | 25.544 | 0.2994 |
| AFIB_SUSPECTED | 20–39 | 25.968 | 41.800 | 0.2741 |
| AFIB_SUSPECTED | 40–59 | 42.272 | 57.496 | 0.3200 |
| AFIB_SUSPECTED | 60–79 | 57.896 | 67.432 | 0.1785 |
| AFIB_SUSPECTED | 80–99 | 67.936 | 83.728 | 0.3513 |

Direct answers:

- **Bigeminy episodes: 0.** Count 0, no timestamps, no repeats. The rule requires
  ≥4 consecutive literal `N,V` pairs; the actual label sequence is irregular ectopy,
  not a locked alternating pattern.
- **Trigeminy episodes: 0.** Rule requires ≥3 consecutive literal `N,N,V` triplets.
- **VT runs detected: 0.** Requires ≥3 consecutive `V` labels. **This changed with
  the fix** — the pre-fix multimodal manifest recorded `vt_run_count: 3` for this
  same segment. The over-detection bug was manufacturing consecutive-V sequences.
- **AFib suspected: YES.** Burden **100.0%** — all 5 of 5 examined 20-beat rolling
  windows were flagged. (Burden = flagged windows / windows examined = 5/5.)
- **Bradycardia episodes: NOT COMPUTED.** The pipeline has **no bradycardia rule**.
  There is no HR-threshold rule of any kind in `score_recording`. The lowest
  instantaneous HR observed was 54.3 bpm, but nothing evaluates it.
- **Tachycardia episodes: NOT COMPUTED.** Same reason — no HR-threshold rule
  exists in the ECG cascade. Rate only enters risk indirectly, via NEWS2's HR
  component. Max instantaneous HR observed was 197.4 bpm, unevaluated by any rule.

---

## SECTION 7 — RULE ENGINE TRACE

Every rule the system evaluates, as emitted in `rule_trace`. Threshold
provenance assessed against `docs/` and in-code citations.

| # | Rule | Threshold | Provenance | Measured | Fired | Would set |
|---|---|---|---|---|---|---|
| **1** | **PVC burden > CRITICAL threshold** | **20.0%** | **[PROJECT-DEFINED — unvalidated]** | **29.921%** | **YES** | **CRITICAL** |
| 2 | VT run count > 0 (run = ≥3 consecutive V) | 0 (runs); 3 beats/run | [CLINICAL LITERATURE] for the ≥3-beat definition of a VT run; [PROJECT-DEFINED] for "any run → CRITICAL" | 0 | no | CRITICAL |
| 3 | PVC burden > HIGH threshold | 10.0% | [PROJECT-DEFINED — unvalidated] | 29.921% | YES | HIGH |
| 4 | PAC burden > HIGH **OR** AFib burden > HIGH | PAC 15.0% / AFib 30.0% | [PROJECT-DEFINED — unvalidated] (both burden cutoffs). The underlying AFib *detector* threshold RR-CV > 0.10 **is** [CLINICAL LITERATURE-VALIDATED] — swept against LTAFDB, 84 records, 449,749 windows | PAC 7.087% (no) / **AFib 100.0% (yes)** | YES | HIGH |
| 5 | Sustained HRV suppression: SDNN < threshold | 20.0 ms | [PROJECT-DEFINED — unvalidated] ("conservative low-HRV cutoff", in-code comment) | 164.351 ms | no | MEDIUM |
| 6 | NEWS2 safety override ≥ critical | 7 | [CLINICAL LITERATURE] — RCP NEWS2 2017 | `null` — **NOT EVALUATED** | no | CRITICAL |
| 7 | qSOFA safety override ≥ high | 2 | [CLINICAL LITERATURE] — Seymour 2016 threshold value | `null` — **NOT EVALUATED** | no | HIGH |

### ★ DECIDING RULE ★

```
Rule 1 — "PVC burden > CRITICAL threshold"
  measured_value : 29.921
  threshold      : 20.0
  fired          : true
  would_set_level: CRITICAL
  is_deciding_rule: true
  depends_on_low_confidence_class: false
```

Three rules fired (1, 3, 4). The cascade is ordered, first-match-wins, so rule 1
(CRITICAL) wins over rules 3 and 4 (both HIGH). Rules 3 and 4 are reported as
fired but non-deciding — correctly.

**Rules 6 and 7 were not evaluated in this run.** `run_full_report()` calls
`ECGPipeline.run()` without `news2_score`/`qsofa_score`, so both overrides receive
`None` and are reported `"evaluated": false` rather than being silently scored 0.
That is the correct conservative behavior, but it means the ECG-only report path
**never applies vitals**, even though real vitals exist for this segment. See
Section 8.

---

## SECTION 8 — RISK VERDICT

| Field | Value |
|---|---|
| **Final risk level** | **CRITICAL** |
| Assessable | true |
| **Deciding rule** | PVC burden > CRITICAL threshold (29.921% vs 20.0%) |
| Alert reason | `"PVC burden 29.9% > critical threshold (20.0%)"` |
| Confidence tier | MODERATE-HIGH |
| Confidence rationale | Deciding rule depends on V-class counts; held-out V F1 = 0.826, the model's most reliable class |

**Did vitals raise or lower the risk? NEITHER — vitals were not applied in this
run.** Verified two ways:

1. In the report path (`run_full_report`), both overrides report
   `"applied": false, "No NEWS2 score was supplied to this run"`.
2. I separately re-scored the same segment *with* the real vitals fed in
   (`score_recording(..., news2_score=4, qsofa_score=1)`): the result is still
   **CRITICAL**, same deciding rule. Neither override can lower a level — the
   code only ever escalates — and neither reached its threshold (4 < 7, 1 < 2).

So for this segment the answer is unambiguous: **vitals changed nothing**, and
they could not have, because ECG alone had already reached the ceiling.

### NEWS2

Computed by `compute_partial_news2()` from the matched real vitals file.

| Component | Value | Score | Source |
|---|---|---|---|
| Heart rate | **115.8 bpm** | **2** | `vitalpatch_col1`, 18 readings, range 113–119 |
| Respiratory rate | **23.8 /min** | **2** | `vitalpatch_col2`, 58 readings, range 16–28 |
| Temperature | **null** | 0 | col 3 present but **no valid readings** in window |
| SpO2 | null | 0 | **NOT AVAILABLE — VitalPatch has no SpO2 sensor** |
| Systolic BP | null | 0 | **NOT AVAILABLE — VitalPatch has no BP sensor** |
| Consciousness (AVPU) | null | 0 | **NOT AVAILABLE — "assumed alert"** |
| **TOTAL** | | **4** | |

**NEWS2 = 4, scored from 2 of 6 components** (`coverage` string:
`"2/6 NEWS2 components scored from real data"`).
`missing_components: ["spo2", "sbp", "consciousness", "temperature"]`.
Interpretation label: MEDIUM. Below the ≥7 critical override threshold.

**Components unavailable from VitalPatch:** SpO2, systolic BP, consciousness —
structurally unavailable (no sensor / no input). Temperature is *normally*
available on this device but had zero valid readings in this particular ±2 min
window, so it is missing for a different, incidental reason.

> Honest caveat carried by the code itself: *"Partial NEWS2 — missing SpO2, BP,
> consciousness. Score is a lower bound. Full NEWS2 requires bedside SpO2 and BP
> measurement."* The three structurally-missing components each contribute **0**
> to the total while being listed as missing. That is conservative (can only
> under-count) but the total does not distinguish "measured and normal" from
> "never measured". See the audit section.

### qSOFA

| Criterion | Value | Flag | Score |
|---|---|---|---|
| HR > 90 (this codebase's proxy criterion) | 115.8 bpm | true | 1 |
| SBP ≤ 100 | null | **unevaluated** | not scored |
| Altered mentation / GCS < 15 | null | **unevaluated** | not scored |
| **TOTAL (local proxy)** | | | **1** |

**qSOFA (local) = 1, from 1 of 3 criteria.** Below the ≥2 threshold.

The code is explicit that this is not a real qSOFA:
*"1-of-3-criteria proxy, not a real qSOFA score… this proxy's maximum possible
value is 1, so it can never independently trigger the ECG cascade's qSOFA
override."* SBP is deliberately **not** scored 0 — the docstring states that
scoring it 0 would assert "not hypotensive", which is unverifiable.

**Discrepancy with the stored corpus figure — flagged.** The multimodal manifest
records `agent_qsofa_score: 2` for this segment, and it is the only segment in
3,632 to reach 2. That 2 comes from the **agent-side** `calculate_qsofa`
(`MedGemma-Agent/guardrails/clinical_rules.py:240`), which is a **different
formula** — see Section 11 and the audit. Local proxy and agent disagree: 1 vs 2.

---

## SECTION 9 — MEDGEMMA INPUT JSON

**This segment was CRITICAL, so MedGemma was skipped.** Below is what *would*
have been sent. The prompt is 15,151 characters. Its embedded structured payload
is the report JSON; the key blocks are reproduced here (full prompt regenerated
via `build_transparency_prompt(report_json)`).

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
    {"condition": "PVC burden > CRITICAL threshold", "measured_value": 29.921, "threshold": 20.0, "fired": true, "would_set_level": "CRITICAL", "is_deciding_rule": true},
    {"condition": "VT run count > 0 (a run = >=3 consecutive V beats)", "measured_value": 0, "threshold": 0, "fired": false, "would_set_level": "CRITICAL", "is_deciding_rule": false},
    {"condition": "PVC burden > HIGH threshold", "measured_value": 29.921, "threshold": 10.0, "fired": true, "would_set_level": "HIGH", "is_deciding_rule": false},
    {"condition": "PAC burden > HIGH threshold OR AFib burden > HIGH threshold", "measured_value": {"pac_burden_pct": 7.087, "afib_burden_pct": 100.0}, "threshold": {"pac_burden_pct": 15.0, "afib_burden_pct": 30.0}, "fired": true, "would_set_level": "HIGH", "is_deciding_rule": false},
    {"condition": "Sustained HRV suppression: SDNN < threshold", "measured_value": 164.351, "threshold": 20.0, "fired": false, "would_set_level": "MEDIUM", "is_deciding_rule": false},
    {"condition": "NEWS2 safety override (>= critical threshold)", "measured_value": null, "threshold": 7, "fired": false, "evaluated": false, "would_set_level": "CRITICAL", "is_deciding_rule": false},
    {"condition": "qSOFA safety override (>= high threshold)", "measured_value": null, "threshold": 2, "fired": false, "evaluated": false, "would_set_level": "HIGH", "is_deciding_rule": false}
  ],
  "deciding_rule": {
    "condition": "PVC burden > CRITICAL threshold",
    "measured_value": 29.921, "threshold": 20.0, "fired": true,
    "would_set_level": "CRITICAL", "is_deciding_rule": true
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
  "final_risk_level": "CRITICAL"
}
```

**Why it was skipped:** `render_narrative()` (`agent_bridge.py:1120-1126`) short-
circuits before any network call when `report_json["risk_level"] == "CRITICAL"`,
logging audit event `TRANSPARENCY_MEDGEMMA_SKIPPED_CRITICAL` with reason
*"CRITICAL level bypasses MedGemma entirely, per safety constraint."* This is a
deliberate safety design: a CRITICAL verdict must not be softened or reworded by
an LLM. Note the prompt itself also contains a bug worth flagging — see Section 11
on the AFib evidence text.

---

## SECTION 10 — MEDGEMMA OUTPUT

**Status: `SKIPPED_CRITICAL`.** No narrative was generated by MedGemma.

- Exact reason: risk level was CRITICAL; `render_narrative()` returns before
  calling `call_medgemma()`.
- Endpoint that would have been used: `http://localhost:11434/api/generate`
- Model: `medgemma`
- **Timeout value vs cold start:** not applicable to this run, since no call was
  made. For the record, the relevant measured numbers from `SYSTEM_AUDIT.md` §0/§5
  are: `ecg_inference/report.py`'s original timeout was **10 s** against a
  **measured 75.6 s cold start** (7.5× over); that constant was since raised to
  `MEDGEMMA_TIMEOUT_S = 90.0`. The separate MedGemma-Agent call site uses 120 s.
  Neither applied here.

The narrative in the report is the **deterministic fallback**, generated locally
by `_deterministic_narrative()`, not by any LLM:

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

**Unsupported-statement check on this narrative:** every number in it
(29.921, 20.0, 10.0, 7.087, 100.0, 15.0, 30.0, 0.826, 0.139, 0.011) traces
directly to the rule trace or the frozen model card. **No unsupported claims
found.** The narrative is template-generated from the trace, so this is expected
rather than impressive.

---

## SECTION 11 — KNOWN BUGS STATUS

| Bug | Source | Status in this run |
|---|---|---|
| **audit_log rows not written** | SYSTEM_AUDIT §1, §12 | **STILL PRESENT** |
| **SDNN 0.0 sentinel** (`ecg_pipeline_core.py`) | AUDIT_2026-07-31 §8.1 | **FIXED — and NOT TRIGGERED here** |
| **SDNN 0.0 sentinel** (`ecg_inference/`) | AUDIT_2026-07-31 §8.1 | **STILL PRESENT** (different code path, not used by this run) |
| **AFib evidence_text says 0.15, code uses 0.10** | this report | **STILL PRESENT — confirmed** |
| **Beat classifier batch vs loop** | SYSTEM_AUDIT §4 | **LOOP mode ran** (fix not in this path) |
| R-peak over-detection / `t_inspect_period` | AUDIT_2026-07-31 §7 | **FIXED and active** (`t_inspect_period=0.36`) |
| Display-only Kalman for waveform PNGs | AUDIT_2026-07-31 §8.2 | **NOT TRIGGERED** (no PNG generated) |
| MedGemma 10 s timeout vs 75.6 s cold start | SYSTEM_AUDIT §0 | **NOT TRIGGERED** (call skipped; constant since raised to 90 s in `ecg_inference/report.py`) |

### Specifically requested checks

**1. `audit_log` row count — was this segment recorded? NO.**
Queried `MedGemma-Agent/data/medgemma_agent.db` directly:

| Table | Rows |
|---|---|
| `patients` | 2,540 |
| `vitals_snapshots` | 6,760 |
| `alerts` | 6,760 |
| **`audit_logs`** | **0** |
| `abg_results` | 0 |

The HIPAA-relevant audit table is empty, exactly as SYSTEM_AUDIT §1 reported.
Patient `184B27` **is** in the DB (1 patient row, 213 `alerts`, 213
`vitals_snapshots`), but **this specific segment is not identifiable** —
`vitals_snapshots` keys on `patient_id` only (`184B27`), with no segment ID
column, and a query for the segment's timestamp `1780171482351` returns 0 rows.
So: the patient is recorded, the segment is not, and the audit trail is empty.

**2. SDNN sentinel — was 0.0 returned? NO.** SDNN = 164.351 ms from 106 valid RR
intervals. The `< 3 intervals` branch was not reached. Post-`e0da1f7` that branch
returns `None`, not `0.0`, and `score_recording` guards with
`sdnn_ms is not None and sdnn_ms < threshold`. Verified in source at
`ecg_pipeline_core.py:1577-1587` and `:2131`.
**Caveat:** the same fix was **not** mirrored into `ecg_inference/features.py` /
`ecg_inference/classifier.py`, which `run_inference.py` (the deployment path)
uses. That copy still has the 0.0 sentinel. This run did not touch it.

**3. AFib evidence_text threshold mismatch — CONFIRMED, STILL PRESENT.**

- Code threshold: `_afib_suspected(..., cv_threshold: float = 0.10)`
  (`ecg_pipeline_core.py:2029`), changed 0.15 → 0.10 on the LTAFDB sweep.
- Displayed text: `agent_bridge.py:666` hardcodes the string
  `f"RR coefficient-of-variation {f.detail['rr_cv']:.3f} > 0.15 over a 20-beat rolling window"`.

This run makes the bug concrete and consequential. The 4th AFib window has
**RR CV = 0.178**. The report tells a clinician it is *"0.178 > 0.15"* — which
happens to be arithmetically true, so it reads fine. But the rule actually fired
on `0.178 > 0.10`. Had a window come in at, say, CV 0.12, the report would have
printed the self-contradictory *"0.120 > 0.15"* while claiming the rule fired.
The displayed threshold is a hardcoded literal that no longer tracks the code.
All five AFib findings in this segment carry the wrong threshold in their text.

**4. Beat classifier batch vs loop — LOOP mode ran.**
This run went through `ecg_pipeline_core.py:2662`:
`result = self.classifier.predict_one(feat_lookup.get(i), beat, mean_rr)` inside
`for i, beat in enumerate(beats)` — one `predict_proba()` call per beat.
The batched fix (SYSTEM_AUDIT §0) was applied to `ecg_inference/classifier.py`'s
new `predict_batch()`, which **this path does not import**. Measured cost here:
183.2 ms for 119 beats (~1.54 ms/beat). At the audit's measured 153× ratio, the
batched equivalent would be roughly 1–2 ms total. Small in absolute terms at this
beat count; the audit's 3,026 ms figure was for 2,000 beats.

---

## SECTION 12 — SELF-CONSISTENCY CHECKS

| # | Check | Result |
|---|---|---|
| 1 | Detected HR vs vitals HR within 5 bpm | **FAIL** (headline), PASS on median-of-valid-RR |
| 2 | PVC burden formula matches beat counts | **PASS** |
| 3 | Risk level matches deciding rule | **PASS** |
| 4 | SDNN physiologically plausible | **PASS** |
| 5 | Beat counts sum to total detected | **PASS** |
| 6 | No duplicate R-peak indices (RR = 0 ms) | **PASS** |
| 7 | MedGemma output does not contradict rule trace | **N/A** (no LLM output) |
| 8 | NEWS2 score matches component values | **PASS** |

**1 — Detected HR vs vitals HR: FAIL as reported, with a known cause.**
Vitals HR 115.8 bpm. Detected HR by the natural reading (beat count / elapsed
time, or mean of all RR) = **80.14 bpm** → **−35.66 bpm, 7× over the 5 bpm bar.**
Cause: 39% of the recording was SQI-gated out, producing five 5.3–7.8 s stretches
with no detected beats. Those stretches inflate mean RR without representing slow
beating. Restricting to non-flagged RR gives 110.74 bpm (−5.06, still marginally
fails) and the median gives 118.11 bpm (+2.31, passes). **The detector is
probably fine; the aggregate HR statistic is not.** The pipeline emits no HR
field at all, so nothing currently surfaces this — a consumer computing HR the
obvious way gets 80 bpm for a patient the device says is at 116 bpm.

**2 — PVC burden: PASS.** 38 / 127 × 100 = 29.9212598…% → reported 29.921%.
Exact. PAC: 9 / 127 × 100 = 7.0866…% → 7.087%. Exact.
*Caveat carried forward:* denominator is detected (127), not analyzed (119),
while the field is named `pct_of_analyzed_beats`. The arithmetic is
self-consistent; the label is wrong.

**3 — Risk verdict follows the trace: PASS.** Rule 1 fired → CRITICAL; it is
marked `is_deciding_rule: true`; final level is CRITICAL. The cascade is
first-match-wins and rule 1 is the highest-precedence rule that fired. Rules 3
and 4 also fired at HIGH and are correctly not deciding. No contradiction.

**4 — SDNN plausible: PASS.** 164.351 ms from 106 valid RR intervals — a real
computed value, not the 0.0 sentinel, and the ≥3-interval precondition was met
with margin. The value is *high*, not low, which is physiologically coherent for
30% ectopy plus AFib-range irregularity. It correctly did **not** trip the
suppression rule.

**5 — Beat counts sum: PASS.** 72 (N) + 9 (S) + 38 (V) + 0 (F) + 8 (Q) =
**127** = `n_beats_detected`. Also consistent: 127 − 8 Q = 119 = `n_beats_analyzed`.

**6 — No duplicate R-peaks: PASS.** 127 detected, 127 unique indices, 0
duplicates, 0 intervals at RR = 0 ms, 0 intervals below 200 ms. Minimum RR is
exactly 200.00 ms — the refractory guard's configured floor, holding precisely.

**7 — MedGemma vs rule trace: N/A.** `SKIPPED_CRITICAL`; no LLM text exists to
check. The deterministic fallback narrative was checked instead (Section 10) and
contains no unsupported statements.

**8 — NEWS2 matches components: PASS.** HR 115.8 → RCP band 111–130 → **2**.
RR 23.8 → RCP band 21–24 → **2**. Temperature null → **0**. SpO2/SBP/consciousness
null → **0** each. Sum = **4** = reported `partial_news2_score`. Component scoring
functions verified against RCP NEWS2 2017 bands (`_news2_hr_score`,
`_news2_rr_score`, `_news2_temp_score`) — all three match the published tables
exactly.

---

## SECTION 13 — PIPELINE TIMING

Wall-clock, `time.perf_counter()`, single run on this segment, 10-vCPU host.

| Stage | Function | Time (ms) |
|---|---|---|
| 1. Ingest / parse | `parse_vitalpatch_ecg` (whole file, both segments) | 19.2 |
| 2. SQI gate | `run_sqi_gate` | 87.9 |
| 3. Resample | `to_target_rate` | 0.0 (<0.05, pass-through 125→125 Hz) |
| 4. Preprocessing / filter chain | `apply_filter_chain` | 21.2 |
| 5. R-peak detection + beat segmentation | `detect_and_segment` (XQRS + snap + refractory guard) | 234.6 |
| 6a. Feature extraction | `batch_feature_matrix` (119 × 56) | 37.3 |
| 6b. HRV | `recording_level_hrv` | 13.2 |
| 7a. Beat classification | `predict_one` loop, 119 beats | 183.2 |
| 7b. Rhythm analysis | `RhythmContextEngine.analyze` | 0.7 |
| 8. Rule engine + risk scoring | `score_recording` | 0.1 |
| — | Vitals load + match (`load_real_vitals`, cold index build) | 863.9 |
| 9. Report generation (full path) | `run_full_report` end-to-end re-run | 486.0 |
| **MedGemma call** | — | **0 (skipped, `SKIPPED_CRITICAL`)** |
| **Total (stages 1–8, instrumented)** | | **597.4 ms** |
| **Total incl. vitals matching** | | **1,461.3 ms** |

Notes on honest interpretation:

- **Stages 5 and 7a dominate** (234.6 + 183.2 = 417.8 ms, 70% of stage 1–8 time),
  matching SYSTEM_AUDIT §4's finding that R-peak detection and the per-beat
  classifier loop are the two bottlenecks.
- **Rule engine and risk scoring are effectively free** (0.1 ms) — consistent
  with the audit's 0.12 ms.
- **The 863.9 ms vitals load is a cold-cache number.** `_build_vitals_index()`
  reads every vitals file for the patient once and caches it per patient
  directory; the second and subsequent segments for `184B27` cost near zero.
  It is not a per-segment cost in a batch run.
- "Beat segmentation" is not separately timable — `detect_and_segment` fuses
  detection and segmentation in one call, so the 234.6 ms covers both.
- The 486.0 ms `run_full_report` figure **re-runs stages 1–9 from scratch**, so
  it overlaps the per-stage numbers rather than adding to them. It is faster than
  the instrumented sum mainly because the vitals index is already warm and
  Python/NumPy caches are hot on the second pass.

---

## SECTION 14 — WHAT THIS REPORT CANNOT TELL YOU

**1. No clinician-annotated VitalPatch labels exist. Beat-classification accuracy
on this device is unknown.** The 38 V beats driving the CRITICAL verdict have
never been confirmed by a human on this hardware. Every accuracy number quoted
anywhere in this report (V F1 = 0.826, S = 0.139, F = 0.011) is **MIT-BIH DS2
held-out performance**, measured on a different device, different electrodes,
different patient population, different sampling rate. Applying it to VitalPatch
is an assumption, not a measurement.

**2. Pseudo-labels are not ground truth.** Nothing in this report was checked
against a reference annotation. The classifier's output *is* the label here.

**3. MIT-BIH validation does not guarantee VitalPatch performance.** Beyond
domain shift: this is a **single-lead** device, whereas MIT-BIH is 2-lead.
Morphology-based V/N discrimination is materially harder with one lead, and none
of the quoted F1 scores account for that.

**4. NEWS2 is incomplete — 2 of 6 components scored here.** SpO2 and blood
pressure are **not measurable by VitalPatch at all** (no sensor). Consciousness
has no input path. Temperature was additionally missing in this window.
The three structurally-absent components each contribute 0 to the total, so
**NEWS2 = 4 is a lower bound, not a score.** A patient hypotensive at SBP 85 and
desaturating at SpO2 89% would score identically. NEWS2 was calibrated assuming
all components are observed; a 2-of-6 subscore has no validated interpretation.

**5. qSOFA can never be fully scored from this device.** SBP (no sensor) and
altered mentation (no input) are both unavailable, permanently. The local proxy
substitutes HR > 90 — **which is not a qSOFA criterion at all**. Its maximum
possible value is 1, below the ≥2 threshold, so it can never fire. Sepsis risk
cannot be ruled out from this data.

**6. The `t_inspect_period` fix was validated at scale for *scoring*, but reports
on disk are stale.** Correcting the brief: the full 2,889-segment corpus **was**
re-scored post-fix (`docs/audits/AUDIT_2026-07-31.md` §7, 0 errors). What was **not**
re-run is report regeneration — every saved `.json`/`.md` under
`data/reports/vitalpatch/` is still pre-fix (2026-07-26). Anyone reading those
files is reading pre-fix numbers.

**7. The largest unexplained effect of the fix is not accounted for.** 503
segments went LOW → HIGH after the fix — the single biggest transition in either
direction, and an *escalation*, not a false-positive reduction. The audit offers
a plausible mechanism (regular double-counting masked genuine irregularity) but
explicitly labels it unproven. This segment cannot speak to those 503.

**8. Risk thresholds are mostly unvalidated.** PVC 20%/10%, PAC 15%, AFib burden
30%, SDNN 20 ms are **project-defined**, not drawn from clinical literature. The
deciding rule for this segment is one of them. Only the AFib *detector* threshold
(RR CV > 0.10) has been swept against labeled data (LTAFDB).

**9. This report describes one 115-second segment.** No trend, no comparison to
this patient's other 212 recordings, no clinical context, no medication history,
no baseline.

---

# APPENDED CHECKS

## CHECK A — Does PVC burden match V_count / total_detected × 100?

**PASS — exact.**
`38 / 127 × 100 = 29.92125984251968…%`; reported `29.921`. Matches to the
rounding precision used.
Cross-check on PAC: `9 / 127 × 100 = 7.086614173…%`; reported `7.087`. Matches.

One qualification, not a failure: the denominator is **detected** beats (127,
including the 8 `Q` beats that were never classified), not **analyzed** beats
(119). Using analyzed would give 31.93%. The verdict is unchanged either way
(both exceed 20%), but the report field is **named** `pct_of_analyzed_beats`
while computing over the detected count (`agent_bridge.py:628`, `n = len(labels)`).
That is a labeling defect, not an arithmetic one.

## CHECK B — Does implied HR from R-peak timestamps match vitals HR within 5 bpm?

**FAIL for the headline figure. Cause identified.**

Vitals HR: **115.8 bpm**.

| Method | HR | Δ vs 115.8 | Verdict |
|---|---|---|---|
| Beat count over beat span | 80.14 | −35.66 | **FAIL** |
| Beat count over full duration | 66.01 | −49.79 | **FAIL** |
| Mean of all 126 RR | 80.14 | −35.66 | **FAIL** |
| Mean of 106 non-flagged RR | 110.74 | −5.06 | **FAIL** (marginal) |
| Median of 106 non-flagged RR | 118.11 | +2.31 | **PASS** |

**This is not a detector failure.** The five multi-second RR intervals
(5328–7784 ms) are stretches where the SQI gate discarded the signal and no beats
could be detected. They drag the mean without representing real bradycardia.
Once excluded, HR lands at 110.7–118.1 bpm, bracketing the device's 115.8.

**But it is a real reporting hazard.** The pipeline emits **no heart-rate field
whatsoever** — not mean, min, or max. Any downstream consumer computing HR the
obvious way (beats ÷ duration) gets **80 bpm for a patient the device measures at
116 bpm**. On a segment with 39% signal loss, that error is large enough to
change clinical impression. Nothing in the current output warns about it.

## CHECK C — Does the risk verdict logically follow from the rule trace?

**PASS.**

Rules that fired: 1 (PVC > 20% → CRITICAL), 3 (PVC > 10% → HIGH), 4 (AFib burden
100% > 30% → HIGH). Rules 2 and 5 did not fire. Rules 6 and 7 were not evaluated.

The cascade in `score_recording` is `if/elif` — first match wins, in strict
severity order. PVC-critical is first, it fired, so the level is CRITICAL. Rules
3 and 4 are lower-severity and correctly reported as fired-but-not-deciding.
`deciding_rule` points at rule 1. `final_risk_level` is CRITICAL. Consistent
throughout.

## CHECK D — Any self-contradictions in the report?

**Four found. None invalidate the verdict; three are real defects.**

1. **AFib evidence text states the wrong threshold — REAL DEFECT.**
   All five AFib findings print *"> 0.15"*; the rule that fired uses **0.10**
   (`agent_bridge.py:666` hardcodes the literal; `ecg_pipeline_core.py:2029` sets
   `cv_threshold=0.10`). The lowest CV here is 0.178, so every printed statement
   is *coincidentally* still true — but the report is telling clinicians the wrong
   decision boundary. A window at CV 0.12 would print the flatly self-contradictory
   *"0.120 > 0.15 … fired=True"*.

2. **`pct_of_analyzed_beats` is computed over detected beats — REAL DEFECT.**
   The same JSON reports `n_beats_analyzed: 119` and then
   `"Q": {"count": 8, "pct_of_analyzed_beats": 6.3}`. Q beats are *by definition*
   the ones excluded from analysis; they cannot be 6.3% of the analyzed set. The
   divisor is 127.

3. **Rule 4 fires on AFib but the report says AFib "raises risk to HIGH", while
   the actual level is CRITICAL — COSMETIC, not a contradiction.**
   The evidence text *"this crosses the burden threshold and raises risk to HIGH"*
   describes what that rule alone would do. The rule trace correctly shows rule 1
   overrode it. Confusing phrasing in isolation, correct in context.

4. **NEWS2/qSOFA reported "not evaluated" while real vitals exist — DESIGN GAP,
   honestly disclosed.** The report says *"NEWS2/qSOFA vitals pairing is not wired
   into this ECG-only bridge"*, and matched vitals (HR 115.8, RR 23.8) were in fact
   available and loadable. The report does not fabricate them, which is the right
   call, but a reader could reasonably assume no vitals existed. They did.

**No case was found of a rule firing that the verdict ignores, or a verdict
asserting something no rule supports.**

## CHECK E — Does the SDNN value make sense for this recording?

**PASS, with an important read.**

SDNN = **164.351 ms**, from 106 valid RR intervals (well above the 3-interval
minimum). It is a genuinely computed value, **not** the 0.0 sentinel.

Physiological plausibility: 164 ms SDNN and 261 ms RMSSD are far above normal
sinus values (typically ~50 ms SDNN over short recordings). That is **coherent
with, not contradictory to,** the rest of the picture: 29.9% ventricular ectopy
plus RR coefficient-of-variation of 0.18–0.35 across all five windows is exactly
the substrate that produces extreme beat-to-beat variability. RMSSD > SDNN
further indicates the variability is short-term and beat-to-beat (ectopy-driven),
not slow drift.

The HRV-suppression rule correctly did **not** fire (164.351 ≥ 20.0).

One caveat on precision: because `rr_flagged` is set per-*beat* when *either*
adjacent interval is out of range, 8 physiologically normal intervals
(448–648 ms) were excluded alongside the 12 genuinely abnormal ones. SDNN is
computed from 106 of 126 intervals. The direction of that bias is not obvious,
and it is undocumented.

## CHECK F — Was the audit log row written for this segment?

**FAIL — NO.**

`MedGemma-Agent/data/medgemma_agent.db`:

- **`audit_logs`: 0 rows** (against `alerts`: 6,760 and `vitals_snapshots`: 6,760)
- Patient `184B27` **is** present: 1 `patients` row, 213 `alerts`, 213 `vitals_snapshots`
- Query for this segment's identifier `1780171482351`: **0 rows**

Two distinct failures:

1. **The compliance audit table is empty**, exactly as `SYSTEM_AUDIT.md` §1
   reported. The code path exists; nothing has ever written to it in the
   environment that produced this database.
2. **Segment-level traceability does not exist.** `vitals_snapshots` has columns
   `(id, patient_id, systolic_bp, diastolic_bp, heart_rate, spo2,
   snapshot_timestamp, created_at)` — **no segment ID column**. Even if the audit
   table were populated, this specific 115-second segment could not be located in
   it. Rows are attributable to a patient, not to a recording.

Separately, the **in-memory** `AuditLog` for this run did populate normally
(`STAGE1_INGEST` … `STAGE8_RISK`, plus
`TRANSPARENCY_MEDGEMMA_SKIPPED_CRITICAL`), and it is hash-chained. But it is
discarded when the process exits — it is never persisted to the database.

---

## Summary of failures found

| Check | Result |
|---|---|
| A — PVC burden formula | PASS (denominator naming defect noted) |
| B — HR vs vitals within 5 bpm | **FAIL** (−35.66 bpm headline; cause identified; no HR field exists) |
| C — Verdict follows rule trace | PASS |
| D — Self-contradictions | **3 real defects found** (AFib threshold text; `pct_of_analyzed_beats`; vitals-not-wired gap) |
| E — SDNN plausible | PASS |
| F — Audit log row written | **FAIL** (`audit_logs` = 0 rows; no segment-level key exists) |
