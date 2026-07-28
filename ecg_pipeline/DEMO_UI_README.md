# ECG Pipeline Replay Demo — what it is and how to run it

This documents the demo/visualization layer built on top of the existing,
unmodified ECG pipeline (`ecg_pipeline_core.py` + `agent_bridge.py`). All
four files below are additive tooling only — no classifier, threshold, or
AAMI-scheme code was touched.

## What was built

| File | Purpose |
|---|---|
| `report_ui.py` | Flask app. Default mode: a **live replay server** — pick a recording (real VitalPatch file or synthetic scenario), click Start, and watch it stream through RAW → PREPROCESSED → ASSESSED (beats + rhythm) → MedGemma REPORT panels, with risk level updating live. Also has a legacy `--report <path.json>` mode that renders one saved report as a static HTML page. |
| `demo_stream.py` | Streaming harness used by `report_ui.py`'s replay worker (and runnable standalone from the CLI). Re-runs the real `ECGPipeline.run()` on a growing prefix of the signal in chunks, "as if" it were arriving live from a patch. Can source from a synthetic scenario or a real recording file. |
| `synthetic_ecg.py` | Synthetic single-lead ECG generator (`NORMAL`, `PVC_BURDEN`, `VT_RUN`, `AFIB_LIKE`, `NOISY` scenarios) with known injected ground truth. Strictly for demo/testing — never used for classifier training, never mixed into DS1/DS2, and every output is tagged `source="synthetic"` with a "SYNTHETIC — not a real patient" banner. |
| `test_pipeline_synthetic.py` | Self-test suite: runs each synthetic scenario through the real, frozen pipeline and asserts the expected rule fires (e.g. `VT_RUN` → CRITICAL, `PVC_BURDEN` → HIGH/CRITICAL, `AFIB_LIKE` → `AFIB_SUSPECTED` finding, `NORMAL` → LOW/no rule fired). Tests the rule engine's reaction to known input, not real-world classifier accuracy. |

Recent fixes made to the report wording (in `agent_bridge.py` and
`report_ui.py`, wording-only — no thresholds/classifier changed):
- The "catch-all" (no dangerous rule fired) case now reads plainly, e.g.
  *"No dangerous thresholds exceeded → risk LOW."* — no more `None`/`null`
  or misleading `FIRED` text.
- `AFIB_SUSPECTED` evidence text now states the recording's actual AFib
  burden vs. the 30% risk-raising threshold, so a LOW risk level next to
  an AFib-suspected flag is no longer self-contradictory to a reader.
- The highlighted rhythm-finding band on the ASSESSED plot is now labeled
  directly with the finding's kind (e.g. `AFIB_SUSPECTED`, `VT_RUN`).
- Plot legends no longer clip (`bbox_inches="tight"`, capped column count).

## How to run the live replay server

```bash
cd /home2/mahimakopalley/projects
python3 -m ecg_pipeline.report_ui --port 5057
```

This prints:
```
ECG Pipeline Replay Demo serving at: http://127.0.0.1:5057
```

Then in a browser go to `http://127.0.0.1:5057/`, pick either:
- **Synthetic scenario**: `NORMAL`, `PVC_BURDEN`, `VT_RUN`, `AFIB_LIKE`, or `NOISY`, or
- **Recorded VitalPatch file**: from the dropdown of files under `data/raw/vitalpatch/`,

set a replay speed, and click Start. You'll be redirected to
`/replay/<run_id>`, which polls `/api/state/<run_id>` and renders the four
panels live.

### If you're on a remote machine (this host is `gnode079`)

Tunnel the port from your local machine before opening the URL:
```bash
ssh -L 5057:localhost:5057 <your-username>@gnode079
```
Then open `http://127.0.0.1:5057/` in your local browser.

### Run in the background (so it survives your shell closing)

```bash
cd /home2/mahimakopalley/projects
nohup python3 -m ecg_pipeline.report_ui --port 5057 > /tmp/report_ui.log 2>&1 &
disown
```

Check it's up:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5057/
```

Stop it:
```bash
pkill -f "ecg_pipeline.report_ui"
```

### Starting a replay from the command line (no browser)

```bash
# synthetic
curl -s -X POST http://127.0.0.1:5057/start -d "mode=synthetic" -d "scenario=AFIB_LIKE" -d "speed=8" -i | head -5

# real recording, by dropdown index
curl -s -X POST http://127.0.0.1:5057/start -d "mode=real" -d "file_idx=0" -d "speed=8" -i | head -5
```
The response is a redirect to `/replay/<run_id>`; poll completion with:
```bash
curl -s http://127.0.0.1:5057/api/state/<run_id> | python3 -m json.tool
```

## How to run the legacy static single-report page

```bash
# list saved reports
python3 -m ecg_pipeline.report_ui --list

# render one saved report.json to a static HTML file and open it
python3 -m ecg_pipeline.report_ui --report path/to/report.json --open
```

## How to run the streaming harness standalone (no server/UI)

Useful for quick CLI checks without Flask.

```bash
# synthetic scenario, printed live per-chunk
python3 -m ecg_pipeline.demo_stream --scenario VT_RUN --speed 8

# a real recording file
python3 -m ecg_pipeline.demo_stream --file data/raw/vitalpatch/Patch_183594/<file>.csv --source vitalpatch --speed 8

# generate a self-refreshing static HTML snapshot instead of console printout
python3 -m ecg_pipeline.demo_stream --scenario PVC_BURDEN --live --live-out /tmp/live.html
```

## How to run the synthetic self-tests

```bash
python3 -m ecg_pipeline.test_pipeline_synthetic
```
Runs all 5 scenarios through the real pipeline and prints PASS/FAIL against
each scenario's known ground truth.

## Scope / guarantees

- Every panel is built from arrays the existing, unmodified
  `ECGPipeline`/`agent_bridge` actually produced — nothing here
  re-implements detection, filtering, or classification.
- Synthetic recordings are clearly labeled (`source="synthetic"`, banner
  text) and are never usable as training/eval data for the classifier.
- MedGemma (Ollama) is optional — if `localhost:11434` isn't reachable, the
  report panel falls back to a deterministic narrative rather than
  blocking or showing blank content.

## Known open item (not fixed, out of scope for wording-only work)

`RhythmContextEngine._afib_suspected` in `ecg_pipeline_core.py` indexes
`AFIB_SUSPECTED` findings into the RR list with `None`s filtered out
(`clean_rr`), not the original `beats` list — a latent index-mapping bug
that would only manifest when RR-flagged beats exist upstream of a
flagged window. Not encountered in the recordings tested so far, but
worth a real fix in a future pass since it affects the true beat range
reported for that finding.
