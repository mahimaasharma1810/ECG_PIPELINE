# ProRhythm Lead-II Rhythm Regularity Pipeline

Classifies a single-lead ECG stream from a wearable patch as **REGULAR /
IRREGULAR / UNABLE_TO_DETERMINE**, with evidence a clinician can check for every
verdict.

> **Rhythm regularity indicator. Not a diagnosis. Not validated for clinical
> use. Cannot distinguish atrial fibrillation from other causes of
> irregularity.**

---

## Status in one line

The software is built, tested and validated on public data. **It is not
validated on the target device**, because R-peak detection on patch data finds
~4.4% too many beats, and one controlled recording is needed to fix that.

| | |
|---|---|
| Threshold | RR CV ≥ 0.1275 — **PROVISIONAL, NOT SHIPPABLE** |
| Held-out database (MITDB) | Se 0.9839, Sp 0.8660, PPV 0.4692, NPV 0.9978 |
| Validated on device | **No** |
| Test suite | 76 passing, 3 xfailing **by design** |

The three xfailing tests are the ones that will report when the device blocker
clears. They must not be skipped or weakened.

---

## Layout

```
rhythm/                  the package
  ingest.py              parse captures, measure the delivered sample rate
  resample.py            dedupe replays, segment, uniform grid
  degrade.py             resolution matching; device signal path
  features.py            RR screen + per-window features (64 beats = 63 intervals)
  sqi.py                 signal-quality gate; the system's ability to refuse
  verdict.py             L3 regularity verdict + refusal band  [PRIMARY]
  smoothing.py           hysteresis (N=5)
  axes/                  L2 rate, L4 ectopy events, combination layer
  layers.py              how L1-L5 combine; disagreement is reported, not averaged
  morphology/            L5 PVC shape features (public data only)
  track_b/               supervised RR model (challenges L3, never sets it)
  fusion/                waveform+timing CNN — BUILT AND REJECTED, see below
  narrative.py           MedGemma wording + structural validation
  medgemma_schema.py     evidence JSON (allow-list; refuses verdict without rate)
  domain_gate.py         blocks clinical output until the device is validated
  streaming.py           real-time engine
  ws_schema.py           real device packet parser
  replay.py              replay captures as device packets

scripts_rhythm/          numbered analysis scripts, s01 … s29
tests_rhythm/            test suite — run: bash tests_rhythm/run_all.sh
reports_rhythm/          measurement outputs (bulk + subject images gitignored)
docs/rhythm/             all project documentation
prorithm_ecg/            subject captures — NEVER COMMITTED
data/                    public datasets — not committed, regenerable
archive/                 superseded beat-classification system
```

---

## Where to start reading

| If you want | Read |
|---|---|
| What happens to the data, step by step, in plain English | `docs/rhythm/PIPELINE_WALKTHROUGH.md` |
| Every measurement and result | `docs/rhythm/ecg_rhythm_classification.md` |
| The device, its data, and its traps | `docs/rhythm/PATCH_AND_DATA.md` |
| What we need from the patch team | `docs/rhythm/PATCH_TEAM_QUESTIONS.md` |
| Design rationale and standing practices | `docs/rhythm/CLASSIFIER_DESIGN.md` |
| Before wiring this into the live path | `docs/rhythm/INTEGRATION_NOTES.md` |

---

## Running it

```bash
bash tests_rhythm/run_all.sh                       # full suite + golden diff

python3 scripts_rhythm/s17_realtime.py \
        --replay prorithm_ecg/1789/2026-08-17_08.csv --speed 0

python3 scripts_rhythm/s12_generate_report.py \
        --input prorithm_ecg/1789/2026-08-17_08.csv --out reports_rhythm/review

# GPU work (venv lives on the SSD; see docs/rhythm/GPU_ENVIRONMENT.md)
/ssd_scratch/mahimakopalley/venv-gpu/bin/python scripts_rhythm/s28_train_fusion.py
```

---

## Five rules that are not negotiable

1. **Do not filter the signal.** The firmware already does (`ecg_clean`); a
   second chain leaves 6–7% of the amplitude. Independently confirmed: the
   unfiltered path scores best (F1 0.9890 vs 0.9831). The device team's own
   filter document agrees.
2. **Do not loosen the quality gate to produce more verdicts.** Removing it on
   healthy subjects yields 34 of 39 windows labelled IRREGULAR and none REGULAR.
3. **Never show a rhythm verdict without the heart rate.** 350 windows in public
   data read REGULAR at over 100 bpm, up to 161. The evidence schema *refuses*
   to serialise a verdict without its rate — enforced in code, not documentation.
4. **Never route IRREGULAR to an alarm.** PPV ≈ 0.47. This is a
   false-reassurance-reduction instrument, not an alerting one.
5. **The LLM never decides.** The verdict is fixed before any model is called;
   its prose is structurally validated or discarded whole.

---

## Things that were built and then rejected by measurement

Recorded because the negative results are load-bearing:

| Idea | Why it was dropped |
|---|---|
| Ectopy-vs-AF distinction | Fires on 91.3% of AFIB windows with zero ectopic beats |
| Learned ectopy-burden model | Loses to a counting rule already in the codebase |
| Waveform + timing CNN | **Worse** than RR alone (0.9633 vs 0.9865 held-out) |
| Beat-free rhythm (autocorr/spectral) | Collapses to near-chance under motion artefact |

The last two matter most: **rhythm is timing**, and the beat detector cannot be
routed around.

---

## The one blocker

Device R-peak detection finds ~4.4% too many beats. Healthy subjects score RR CV
**0.19** where genuinely healthy reference data scores **0.03**.

**What clears it:** one recording — a subject wearing the patch *and* a chest
strap with per-beat RR export (Polar H10 or equivalent), simultaneously. Five
minutes rest, ten minutes seated still, two minutes deliberate movement,
wall-clock noted at both starts. Under an hour.

Nothing already held substitutes: the device's own heart rate is a smoothed
average that can show the beat *count* is wrong but not where each beat truly
is.
