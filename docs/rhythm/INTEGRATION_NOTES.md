# Integration notes — rhythm pipeline into the live path

**Read before touching any threshold in this system.**

> Rhythm regularity indicator. Not a diagnosis. Not validated for clinical use.
> Cannot distinguish atrial fibrillation from other causes of irregularity.

---

## 1. THE INTEGRATION WILL LOOK BROKEN. IT IS NOT.

When this runs against a live patch it will emit **`UNABLE_TO_DETERMINE` for
roughly 38 of every 39 windows**. Measured, not estimated
(`1789/2026-08-17_08.csv`: 38/39, dominant reason `bSQI 0.882-0.938 < 0.950`).

**This is the safety gate working. It is not a plumbing failure.**

### DO NOT loosen `bsqi_min` to make verdicts appear

The temptation will be to lower `BSQI_MIN` in `rhythm/pipeline.py` from 0.95
until the dashboard fills with verdicts. Here is what that produces, measured
on the same capture with the gate removed entirely:

| Verdict | Windows |
|---|---|
| REGULAR | **0** |
| UNABLE (refusal band) | 5 |
| **IRREGULAR** | **34** |

**34 of 39 windows would be labelled IRREGULAR on a healthy adult, and none
REGULAR.** The RR CV distribution for that capture is min 0.116, median 0.191 —
every window sits at or above the 0.1275 threshold.

For scale: genuinely healthy windows in expert-labelled public data have a
median RR CV of **0.0305**. The device figure is **6x higher on a healthier
population**. That gap is R-peak detection error, not physiology.

There is no `bsqi_min` value that produces correct answers, because the RR
series feeding the threshold is corrupted upstream.

### The honest framing

This integration is **plumbing, not enablement**. It gets the path built and
testable end to end so that when detection is fixed, verdicts flow with no new
integration work. **It does not make the feature work.**

---

## 2. What is gated, and by what

`rhythm/domain_gate.py` reads `reports_rhythm/device_domain_validation.json`.
No file, or a failed validation, means:

| Output | State |
|---|---|
| Rhythm verdict + evidence | permitted (ENGINEERING) |
| Clinical severity (Normal/Abnormal/Critical) | **BLOCKED** |
| MedGemma report | **BLOCKED** |

Permission comes from an artifact written by the validation harness, **not a
config flag**, so it cannot be switched on by editing code. Reports are stamped
`output_class: ENGINEERING`.

**The gate opens only when `rhythm/reference.py` passes on a controlled
recording**, which requires ALL of:

```
sensitivity     >= 0.95
PPV             >= 0.95
|RR bias|       <= 10 ms
RR CV error p95 <= 0.027     <- BINDING criterion
```

The last one matters most. Simulating today's device behaviour (4.4% spurious
peaks) gives Se 1.0000 and **PPV 0.9585 — which passes a "95% agreement" gate**
— while CV error is **0.1741, 6.4x the budget**. Beat-level agreement alone
would wave through a detector that destroys the quantity the threshold consumes.

---

## 3. WebSocket schema — use the real one

`rhythm/ws_schema.py` parses the actual device schema:

```
top-level: b, app_type, pid, mac, cid, clid, test_id, sNo, start_time,
           end_time, duration, status, vitals, ecg_clean, vitals_status,
           match_id, os, new_score, new_score_freq, alerts, vital_alerts, alert
vitals:    t, hr, skt, rr, spo2, hrv, sp, dp, finger_spo2
ecg_clean: [{e, t}, ...]        e = amplitude, t = timestamp
```

An earlier client assumed `{"samples": [{"timestamp_ms", "amplitude"}]}`. On a
real packet it would have **connected successfully, logged no error, and
yielded zero samples** — the same silent-failure class as the MITDB NUL-byte
bug. The parser now **raises** on an unrecognised packet, and aborts after 20
failures, because a silently empty stream is worse than stopping.

### `sNo` gives the live path what the offline path cannot have

No historical capture CSV contains `sNo`, so replay and packet loss were
*inferred* from timestamps. The live stream carries it, so the client now
reports **measured** values:

- replay: sequence number seen before
- packet loss: summed gaps in the sequence

Keep `sNo`, `start_time`, `end_time`, `duration`, `vitals_status` — capture
them, do not discard.

---

## 4. Do not add a second filter chain

The field is named `ecg_clean` because the firmware already filtered it.
A second chain leaves 6-7% of the amplitude. Independently confirmed on public
data: the **unfiltered** path scored best at device resolution (F1 0.9890 vs
0.9831 filtered).

---

## 5. Amplitude sanity check — use p95, not min/max

An earlier criterion of "amplitude ~±200" came from a single file. Across 37
captures the pooled range is **[-1439.5, +998.9]**, with 35/37 exceeding ±300 —
because **max is artefact-dominated** (median max/p95 ratio **7.6x**, up to 27x).

Robust statistics instead:

| Statistic | Median | Range |
|---|---|---|
| p95\|x\| | 65.6 | 20.6-403.4 |
| IQR | 17.6 | 6.8-90.6 |
| \|max\| | 603.8 | 141.5-1438.1 |

Even p95 varies ~20x, and **within-subject spread (1.4-17.4x) exceeds
between-subject spread (5.5x)** — so it tracks session conditions (contact,
motion), not a fixed per-device gain.

**Recommendation:** treat p95|x| as a wide per-session sanity check (10-500),
not a tight pass/fail band. Within-session quality is already handled by the
SQI gate, which uses only scale-relative measures and needs no amplitude
constant — which is why the unknown amplitude units have never propagated
into the rhythm maths.

---

## 6. Threshold status

`RR CV >= 0.1275` — **PROVISIONAL, NOT SHIPPABLE**, and this status travels in
every `report.json` and every MedGemma JSON.

| | |
|---|---|
| Derived on | LTAFDB, 16 records, 21,920 windows, 89.70 Hz |
| Robustness | LORO spread 0.0125; CI95 [0.1150, 0.1525] |
| Held-out database | MITDB: Se 0.9839, Sp 0.8660, PPV 0.4692, NPV 0.9978 |
| Validated on device | **NO** |

**Only the REGULAR verdict is actionable** (NPV ~0.998). IRREGULAR has
PPV ~0.47: it prompts a look and must **never** fire an alarm or reach an
escalation desk on its own.

**HR limits (40/60/100/150 bpm) have NO provenance** — conventional defaults,
never derived or validated in this project. They are marked
`PROVISIONAL - NOT SHIPPABLE` and are only reachable once the gate opens.

---

## 8. Three axes now, not one verdict

Each window emits `rate_state`, `verdict` (regularity), `ectopy_burden` /
`pause_count`, and a combined `severity`. Each axis refuses independently.

**On patch data only the RATE axis works.** Measured on 1,210 windows of healthy
adults: rate median 85.1 bpm with zero extremes (sound), regularity RR CV median
0.1681 against 0.0305 for healthy public data (broken), ectopy present in 95.8%
of windows and pauses in 7.5% (broken).

**With the gate open this would raise 91 false CRITICAL alarms on healthy
people**, all from manufactured pauses. Do not open the gate per-axis to "get
the rate working" — rate is already emitted as ENGINEERING output.

**Rate uses its own quality test**, deliberately excluding bSQI. Do not
re-couple them: doing so marks rate UNDETERMINED in 97% of windows for a reason
that does not apply to a mean.

## 7. Known gap the rhythm verdict cannot cover

**Sustained supraventricular tachycardia can read as REGULAR at an abnormal
rate.** High-burden atrial activity is fast but evenly spaced: at >16 PACs per
64-beat window the IRREGULAR flag rate *drops* to 60% from 77.6%.

**Consequence for the UI: the rhythm verdict must never be surfaced without the
rate beside it.** A REGULAR badge alone is unsafe. This is a display constraint,
not a footnote.
