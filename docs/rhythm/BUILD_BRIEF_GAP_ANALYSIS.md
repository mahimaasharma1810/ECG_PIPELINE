# BUILD_RHYTHM_MODEL.md — what is already done, what is new, what conflicts

Assessment of `prorithm_ecg/1789/BUILD_RHYTHM_MODEL.md` (dated 2026-09-01)
against the system as built.

> Rhythm regularity indicator. Not a diagnosis. Not validated for clinical use.
> Cannot distinguish atrial fibrillation from other causes of irregularity.

**Headline: 10 of the brief's 12 steps are already built and validated. One
instruction contradicts a measured finding and must not be followed. One step
(the waveform CNN) is genuinely new and is blocked behind the device problem.**

---

## 1. Step-by-step status

| Brief step | Status | Where |
|---|---|---|
| 1. Define output labels | **DONE** | `verdict.py`, `axes/`, `layers.py` |
| 2. Prepare patch dataset | **DONE** | `ingest.py`, `resample.py` |
| 3. Clean signal pipeline | **DONE — except one instruction, see §2** | `resample.py` |
| 4. Extract rhythm features | **DONE** | `features.py`, `sqi.py` |
| 5. Waveform windows | **DONE** (64-beat, not fixed-second — see §3) | `features.py` |
| 6. Waveform + timing CNN fusion | **NOT BUILT** — see §4 | — |
| 7. Public datasets | **DONE** (4 of 6 named; see §5) | `degrade.py`, `s10`, `s13` |
| 8. Match public data to device | **PARTLY** — see §6 | `degrade.py` |
| 9. Patient-level split | **DONE** | grouped CV throughout |
| 10. Uncertainty / refusal | **DONE** | `sqi.py`, refusal band, `domain_gate.py` |
| 11. Real-time websocket | **DONE** | `streaming.py`, `ws_schema.py`, `s17` |
| 12. Validate before claiming | **DONE** | held-out-database throughout |

The brief's §6 "what to avoid" list is already enforced in code: no training on
unlabelled patch data, no AF claim, patient-level splits, held-out validation.

---

## 2. CONFLICT: "filter noise using light ECG bandpass"

**Do not do this.** It is the one instruction in the brief that contradicts a
measured result — **and the device team's own filter document agrees.** That
document states the system "needed no additional filters to reject unwanted
noises from hardware and the human body" (see `PATCH_TEAM_ANSWERS.md`).

The device field is named `ecg_clean` because **the firmware already filters**.
A second filter chain was measured to leave **6–7% of the signal amplitude**:

```
device ecg_clean : p95|x| = 55.02 , peak-to-peak = 217.08
after 2nd chain  : p95|x| =  3.89 , peak-to-peak =  13.23
```

Confirmed independently on public data at device resolution — the **unfiltered**
path scored best:

| Arm | Se | PPV | F1 |
|---|---|---|---|
| 360 Hz, filtered | 0.9883 | 0.9821 | 0.9852 |
| 89.7 Hz, filtered | 0.9877 | 0.9785 | 0.9831 |
| **89.7 Hz, unfiltered (device path)** | **0.9915** | **0.9865** | **0.9890** |

Skipping the filter won on 20 of 46 records, biggest gain +0.131 F1 on record
113. At 89.7 Hz the Nyquist limit is 44.85 Hz, so a standard chain sits on the
edge of the band it is meant to preserve.

**The pipeline correctly applies no filtering. This must stay.**

---

## 3. Windowing: beats, not seconds

The brief suggests fixed 5 / 10 / 30 s windows. The system uses **64 beats
(= 63 intervals, ~45 s)**.

Reason: rhythm features are per-interval statistics. A fixed-second window
contains a variable number of intervals, so RR CV computed over 5 s at 50 bpm
(4 intervals) is not comparable to 5 s at 150 bpm (12 intervals). A fixed-beat
window makes the statistic comparable across rates — which matters because rate
and regularity are separate axes here.

Streaming already emits every 8 beats (~6 s at 75 bpm), so the responsiveness
the brief wants is present without the comparability problem.

---

## 4. Step 6, the waveform CNN — the one genuinely new item

**Not built, and correctly blocked.** Three reasons, in order of how hard they
are to remove:

1. **Device beat detection is wrong by ~4.4%.** A waveform model trained or run
   on windows aligned to wrong beat positions inherits that error. This is the
   same blocker as everything else device-side.
2. ~~The lead configuration is OPEN.~~ **RESOLVED 2026-09-02: Lead II,
   confirmed by the patch team.** A CNN trained on modified-Lead-II reference
   data now has a defined relationship to the device signal. This blocker is
   removed.
3. **Torch is CPU-only on this host** (`2.12.1+cpu`, `cuda=False`) despite a
   GTX 1080 Ti being present. A CUDA build is needed before CNN training is
   practical.

**What the morphology work already established**, and which should inform any
CNN attempt:

- Hand-built lead-robust features reached held-out-database F1 **0.63–0.73**,
  deployment PPV **0.62**.
- They are **2–8× worse** on the hard negatives (supraventricular, fusion) than
  on normal beats — and F1 hides this.
- Performance **survives downsampling to 89.7 Hz** at a cost of under 2 F1
  points, so resolution is not the obstacle.
- Grouped-CV F1 SD is **0.12–0.21**: per-patient performance is close to
  unpredictable.

A CNN would plausibly beat 0.63–0.73 on clean public data. It would not remove
the lead dependence or the detection error, and it would be **harder to audit**
— which matters for a system whose deterministic core is its main clinical
argument.

**Recommendation:** keep the deterministic verdict primary. A CNN, if built,
annotates; it does not decide. Build it only after the device blocker clears.

---

## 5. Datasets

| Named in brief | On host |
|---|---|
| MITDB | **48 records** |
| SVDB | **78 records** |
| LTAFDB | **16 of 84** (download stalled) |
| INCARTDB | **75 records** (not yet used) |
| PhysioNet challenge | challenge2017 present — **median 30 s, too short for 64 beats** |
| PTB-XL | **NOT on host** |

PTB-XL is 10-second records, so like challenge2017 it **cannot produce a single
64-beat window**. It is useful for morphology work, not for rhythm windows.

INCARTDB is the one genuinely useful unused source — 75 records, 12-lead, with
beat annotations. Worth adding as a third held-out database.

---

## 6. Step 8: noise and jitter simulation is only half done

**Done:** public data is resampled to the device's measured 89.70 Hz before any
comparison, and RR timing is snapped to that grid.

**Not done:** the brief asks to *simulate wearable noise and time jitter*. This
has been characterised but not injected:

- BLE packet-arrival jitter: ~6.5 samples share one stamp, one burst per ~73 ms,
  giving **±36 ms** of per-sample timing uncertainty
- Replay: up to **49.3%** duplicate rows in 28 of 37 captures
- Dropout: **10.2%** of recorded span, gaps to 22 minutes
- Motion artefact: reaching **~8× baseline** amplitude

**This is buildable now and is genuinely useful.** Injecting measured device
artefacts into labelled public data would let us estimate how much performance
the device domain costs — *without* needing the controlled recording. It would
not replace that recording, because simulated artefacts are not ground truth,
but it would bound the expected degradation.

**Recommended as the next public-data task.**

---

## 7. What the brief does not cover, and the system does

Findings that post-date it, all measured:

- **Rate must never feed the regularity decision.** HR discriminates
  irregularity at AUC 0.9221 — a cohort artefact (irregular episodes run at 112
  vs 75 bpm). A model given rate learns "fast means irregular".
- **The SVT gap.** 350 windows (7.7%) read REGULAR at HR > 100, up to 161 bpm;
  339 more read REGULAR below 60 bpm. Enforced structurally: the schema
  **refuses to serialise a verdict without its rate**.
- **Ectopy burden cannot attribute cause.** It fires on 91.3% of AFIB windows
  containing zero ectopic beats, so it is interpretable only when the rhythm is
  otherwise regular.
- **The system is a false-reassurance-reduction instrument.** Layering does not
  make "irregular" actionable (PPV 0.4957 → 0.2183 on a matched target) but cuts
  missed-significant windows from 34.2% to 11.6%.
- **Per-patient performance is unpredictable** across layers — one systemic
  property of per-beat inference on single-lead data.

---

## 8. Recommended order

**Buildable now, no new data:**

1. Device-artefact simulation (§6) — bounds expected device degradation
2. INCARTDB as a third held-out database (§5)
3. Finish the LTAFDB download (68 records outstanding)

**Blocked on the patch team** — see `PATCH_TEAM_QUESTIONS.md`

**Blocked on the controlled recording:** everything device-side, including any
waveform CNN.
