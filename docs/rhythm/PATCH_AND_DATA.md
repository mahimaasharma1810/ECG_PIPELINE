# The ProRhythm / SeNSiO Patch and Its Data

> **UPDATE 2026-09-02 — see `PATCH_TEAM_ANSWERS.md`.** The patch team confirmed
> **Lead II** for both raw and clean, and a native sampling rate of
> **120–140 Hz**. The historical captures' 89.70 Hz is a DELIVERY rate. Their
> filter document also confirms no additional filtering is needed.

Everything known about the device and the data it produces. Every number here
was measured on the 37 captures we hold, unless marked otherwise.

**Confidence labels used throughout:**

| Label | Meaning |
|---|---|
| **CONFIRMED** | Stated by the device team, or verified from device configuration |
| **MEASURED** | We computed it from real capture files; the method is named |
| **UNCONFIRMED** | Believed but not verified — treat as an assumption |
| **OPEN** | We need an answer from the device team |

> Rhythm regularity indicator. Not a diagnosis. Not validated for clinical use.
> Cannot distinguish atrial fibrillation from other causes of irregularity.

---

## 1. The hardware

| Property | Value | Confidence |
|---|---|---|
| Device | ProRhythm / SeNSiO wearable patch | CONFIRMED |
| Electrodes | **3 — RA, LA, LL** | **CONFIRMED (device team, 2026-08-30)** |
| Output | **2-lead ECG** | **CONFIRMED (device team, 2026-08-30)** |
| Which lead(s) reach us | **Lead II** (raw and clean) | **CONFIRMED (patch team, 2026-09-02)** |
| Nominal sample rate | 100 Hz | device command parser default |
| Amplitude units | **UNKNOWN** — not mV, probably ADC counts | UNCONFIRMED |
| Firmware filtering | Yes — the data field is literally named `ecg_clean` | CONFIRMED |

### 1.1 What three electrodes means

With RA (right arm), LA (left arm) and LL (left leg) you have Einthoven's
triangle. Three leads are definable, any two of which are independent:

```
Lead I   = LA − RA
Lead II  = LL − RA
Lead III = LL − LA
```

Lead II usually gives the tallest upward R-wave, which is why it is the
conventional choice for beat detection.

> **CORRECTION, 2026-08-30.** Our earlier documents recorded "2-electrode
> bipolar, RA + LL → Lead II", taken from the original project brief. That is
> **wrong**. The device has three electrodes and produces two leads. All prior
> documents describing a single Lead II need this correction applied.

### 1.2 The consequence we have not resolved

Every file we hold carries **one** amplitude channel. The live packet schema
has **one** `ecg_clean` array. So either the second lead is dropped somewhere
between device and capture, or only one lead is transmitted.

**This matters more than it looks.** Our signal-quality gate currently runs two
*algorithms* over the *same* waveform. That is failing on device data, because
both algorithms make the same mistake on the same noise. Two *physically
independent leads* would be a much stronger check — a real heartbeat appears in
both, a movement artefact usually does not. That is the standard way multi-lead
systems suppress precisely the false-beat problem that is currently blocking
this project.

It would not replace the controlled reference recording, but it might reduce
the over-detection using data we already have.

---

## 2. Data formats

### 2.1 Live capture CSV — what we actually hold

`prorithm_ecg/{subject}/{YYYY-MM-DD_HH}.csv`

```
timestamp_ms,amplitude,received_at_utc
1786618155941.0,-0.6370956301689148,2026-08-13T10:49:31.849051+00:00
```

One channel only. No lead label. No sequence number.

### 2.2 Live WebSocket packet — the real schema

CONFIRMED from device logs:

```
top-level: b, app_type, pid, mac, cid, clid, test_id, sNo, start_time,
           end_time, duration, status, vitals, ecg_clean, vitals_status,
           match_id, os, new_score, new_score_freq, alerts, vital_alerts, alert

vitals:    t, hr, skt, rr, spo2, hrv, sp, dp, finger_spo2
ecg_clean: [{e, t}, ...]        e = amplitude, t = timestamp
```

`sNo` is a packet sequence number. **It exists in the live stream but in none of
the captured CSVs**, so historical replay and packet loss had to be inferred
from timestamps rather than counted.

> **A silent-failure warning.** An earlier client assumed
> `{"samples": [{"timestamp_ms", "amplitude"}]}`. Fed a real packet it would
> **connect successfully, log no error, and yield zero readings.** The parser
> now refuses an unrecognised packet rather than returning nothing quietly.

### 2.3 Device vitals — paired snapshots

`data/raw/cliniaura_live/vitals/{subject}/{same filename}.csv`

```
received_at_utc,snapshot_timestamp,heart_rate,spo2,systolic_bp,
diastolic_bp,respiration_rate,temperature,hrv
```

All 37 ECG captures have a matching vitals file. 6,836 snapshots, one every
~12.5 s.

**`hrv` is mean RR interval in milliseconds** (MEASURED: `60000/hrv` tracks
`heart_rate`, correlation 0.724 across 6,754 paired snapshots).

**Two cautions:**

- `hrv` is heavily smoothed (lag-1 autocorrelation +0.656). It can check the
  **average rate** and can **never** check beat-to-beat variability — which is
  the quantity our classifier actually uses. It is **not beat-level ground
  truth.**
- `temperature` spans 36.8 to 99.8, mixing Celsius and Fahrenheit. Unusable.
  Not needed by this pipeline.

---

## 3. What we hold

| | |
|---|---|
| Captures | **37 files, 4 subjects** (1049, 1789, 1793, 1794) |
| Total samples | 7,508,399 |
| Total span | 64,541 s (17.9 h) |
| Active recording time | 57,978 s (16.1 h) — 10.2% is dropout |
| Date range | 2026-08-07 to 2026-08-20 |
| Rhythm labels | **NONE** |
| Population | Healthy adults, self-recorded, protocol unknown |

**The central data problem:** every recording is from one class — healthy. You
cannot derive a decision boundary from one class. The other side must come from
public labelled data, which makes that dependency the methodological core of the
whole project.

---

## 4. Measured signal properties

All MEASURED across all 37 captures.

| Property | Value | Note |
|---|---|---|
| **True sample rate** | **89.70 Hz** | after removing re-sent rows; range 87.96–90.50, CV 0.6% |
| Apparent rate, raw | 51–167 Hz | wrong — inflated by re-sent data |
| Apparent rate, `median(dt)` | 1000 Hz or infinite | wrong — half of readings share a timestamp |
| Re-sent rows | up to **49.3%**, in **28 of 37 files** | exact duplicate `(timestamp, amplitude)` pairs |
| Duplicate timestamps | 15.6%–74.8% | distinct readings sharing one packet stamp |
| Non-monotonic timestamps | **29 of 37 files** | time runs backwards by ~13,100 ms |
| Dropout | 10.2% of span; gaps to 22 min | |
| Amplitude, p95\|x\| | median 65.6 (range 20.6–403.4) | robust measure |
| Amplitude, absolute max | median 604 (range 141–1438) | **artefact-dominated** |
| Packet burst size | ~6.5 readings per packet | |
| Inter-packet interval | ~73 ms | |

### 4.1 Amplitude: use p95, not min/max

Median max/p95 ratio is **7.6×** (up to 27×), because the maximum is dominated
by movement artefact. Motion spikes reach ~8× baseline.

Within-subject spread (1.4–17.4×) **exceeds** between-subject spread (5.5×), so
amplitude tracks session conditions — electrode contact, movement — not a fixed
per-device gain. A tight absolute band is therefore the wrong shape for this
quantity. Use p95|x| as a wide per-session sanity check (10–500).

This is also why the unknown amplitude units have never caused a problem: every
quality measure we use is scale-relative, and the rhythm calculation uses only
beat *timing*.

---

## 5. The three traps in this data

**Trap 1 — the sample rate is not `rows ÷ duration`.**
That gives up to 167 Hz. You must remove exact duplicate rows first, sort,
exclude dropout gaps over 1 s, and only then divide. 28 of 37 files are affected.

**Trap 2 — the timestamps are not measurement times.**
They are Bluetooth packet-arrival stamps. About 6.5 readings share one stamp,
one packet per ~73 ms, so a reading's true time is uncertain by **±36 ms**.

MEASURED consequence, simulating a *perfectly steady* heartbeat:

| Timing method | Variability floor |
|---|---|
| Using per-reading timestamps | **0.095** |
| Counting readings on a 89.7 Hz grid | **0.006** |

Our decision threshold is 0.1275. The first method puts the noise floor almost
on top of it — the system would be measuring Bluetooth behaviour, not the heart.

**Trap 3 — do not filter it again.**
The field is named `ecg_clean` because the firmware already filtered it. A
second filter chain leaves only 6–7% of the signal amplitude. Independently
confirmed on public data: the **unfiltered** path scored best at this device's
sample rate (F1 0.9890 vs 0.9831 filtered).

---

## 6. Known problem with detection on this device

**The beat detector finds approximately 4.4% too many beats on real patch data.**

MEASURED by comparing our median RR interval against the device's own reported
median RR across all 37 captures: ratio **0.9578** (IQR 0.9495–0.9652); our
heart rate exceeded the device's in **33 of 37** captures.

Why it matters: each false beat splits one interval into two, manufacturing
irregularity. Around 3 false beats per 64-beat window reproduces a variability
score of ~0.156 — which is what we observe.

The result: healthy adults on this device score a median variability of **0.19**,
where genuinely healthy windows in expert-labelled public data score **0.03**.
Six times higher, on a healthier group.

Visual inspection confirms the cause: the detector marks low-amplitude
deflections (~20) alongside real R-waves (~80–100), concentrated where signal
quality swings within a recording.

> **Important distinction.** The device's reported heart rate is what revealed
> this, but it is **not beat-by-beat ground truth**. It is a smoothed average.
> It can show that the beat *count* is wrong; it cannot show where each beat
> truly is, and therefore cannot be used to fix the detector.

---

## 7. OPEN QUESTIONS for the device team

> **These are now maintained as a standalone document to send:
> `PATCH_TEAM_QUESTIONS.md`.** It marks Q1 (lead configuration), Q2 (controlled
> recording) and Q3 (`sNo` logging) as BLOCKING, and states what each unblocks.
> The list below is kept here for context.

Ordered by how much they would change the work.

**1. Does `ecg_clean` carry one channel or two?**
If two, what is the structure — interleaved, a second array, a second key?

**2. If one channel, which lead is it — I, II, or III?**
We validated our detector against **modified Lead II** in the public reference
database, assuming the device streams Lead II. **If it actually streams Lead I,
that comparison is not like-for-like** — Lead I has a smaller R-wave in most
people, which would partly explain poorer detection here.

**3. If two leads exist at the device or app layer, where is the second one
dropped?** Recovering it may reduce the over-detection in §6 using data we
already hold.

**4. Is 89.7 Hz per channel, or shared across two channels (~45 Hz each)?**
This changes the noise-floor arithmetic in §5.

**5. Where exactly are the electrodes placed on the torso?**
A chest patch cannot reach the actual limbs, so these are torso-placed
equivalents. This affects how comparable the signal is to reference databases.

**6. What are the amplitude units, from device documentation?**
Not urgent — the rhythm maths is timing-only — but it would close an open item.

**7. Can `sNo`, `start_time`, `end_time`, `duration` and `vitals_status` be
written into captured files?**
They exist in the live stream. Capturing them turns replay and packet loss from
*estimates* into *counts*.

---

## 8. What is still needed, and cannot be substituted

**A controlled recording with beat-level ground truth.**

- One existing subject, wearing the **ProRhythm patch** and a **reference device
  that exports per-beat RR intervals** (Polar H10 or equivalent, ~1 ms
  resolution), **simultaneously**
- 5 minutes rest before recording
- **10 minutes seated, still, no talking**, electrode contact verified
- **2 minutes of deliberate movement at the end** — gives a labelled artefact
  segment for tuning the quality gate
- **Wall-clock time noted at both device start times**

Nothing we already hold substitutes for this. The device's own heart rate is a
smoothed average, not beat-level truth. Public data is a different device and a
different population. Until this recording exists, beat detection on this patch
cannot be corrected, and no result from the system on a real patch should be
treated as clinically meaningful.
