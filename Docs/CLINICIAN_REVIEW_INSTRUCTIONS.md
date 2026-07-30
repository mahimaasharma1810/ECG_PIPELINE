# Clinician Review Instructions — CRITICAL Segment Sample

**Written:** 2026-07-30
**Sample file:** `data/reports/clinician_review_sample.csv` (50 segments, gitignored — local only)

## Why this exists

Of 2,889 assessable ECG segments in the current batch
(`data/reports/vitalpatch_run_manifest.csv`), 290 (10.0%) came back CRITICAL.
That is a high rate for a recovering post-op cohort, and there is no ground
truth on this corpus to say whether it reflects real events or classifier
false positives — the ventricular-beat classifier's precision is 0.754
(roughly 1 in 4 "ventricular beat" calls is wrong), and ventricular-beat
burden drives both CRITICAL rules. Only clinical review can settle this. See
`Docs/HANDOFF.md` §Task 5 and §8 for the full background.

This is not a diagnostic tool and no output here should be treated as one.
The system produces decision support, not a diagnosis.

## What is in the sample

`data/reports/clinician_review_sample.csv` is a random sample of 50 of the
290 CRITICAL segments (seed 42, reproducible), one row per segment, with the
same columns as the full manifest (`patient_id`, `segment_id`,
`n_beats_analyzed`, etc.).

For every sampled segment, two report files already exist at:

```
data/reports/vitalpatch/<patient_id>/<segment_id>.md      <- human-readable report
data/reports/vitalpatch/<patient_id>/<segment_id>.json     <- structured data (same info, machine-readable)
```

All 50 were verified present (0 missing) before this file was written.

## What to do

1. For each of the 50 segments in the sample CSV, open its `.md` report. It
   contains:
   - Beat classification summary (N/S/V/F/Q counts and percentages, with
     confidence caveats already attached to the unreliable classes S and F)
   - Any rhythm findings (AFIB_SUSPECTED, VT runs, sustained PVC burden, etc.)
     with the exact rule that fired and the numbers behind it
   - The final risk level and which threshold triggered it

2. Judge only one question per segment: **does the CRITICAL call look
   clinically plausible given the beat classifications and rhythm findings
   described?** You do not need raw waveform access to make this call from
   the report alone, but if you have access to the original ECG file
   (`data/raw/vitalpatch/<patient_id>/<segment_id-prefix>_ecg.csv`) and want to
   inspect the waveform directly, that is welcome and more reliable.

3. Record your verdict in a simple CSV, one row per segment, with at minimum:

   ```
   segment_id,verdict,notes
   1778744267337_VC2B008BF_183594_ecg_seg0,plausible,clear V-run with...
   ```

   Suggested `verdict` values: `plausible`, `implausible`, `uncertain`. Free
   text in `notes` for anything that would help tune thresholds or classifier
   behavior later (e.g. "V-beats look like noise artifacts, not real PVCs").

4. Do not feel obliged to give a verdict on all 50 — a partial review (e.g.
   10–20 segments) is still useful evidence and better than none.

## What NOT to do

- Do not guess what a clinician would say and fill this in yourself if you
  are not a clinician reviewing real waveforms — that would defeat the
  purpose of this step entirely (see `Docs/HANDOFF.md`'s critical rules on
  fabrication).
- Do not treat a "plausible" verdict as validation of the whole 10.0% CRITICAL
  rate — 50 segments is a sample for directional signal, not a statistically
  powered validation study.
