# Questions for the ProRhythm patch team

**Short answer to "do we need to clarify things before building?": YES — four
of these block work, and three of them cannot be answered by us at any cost.**

Ordered by how much each unblocks. Q1–Q4 are blocking. Q5–Q9 are useful.

> **Updated 2026-09-02.** The lead configuration is now **confirmed as Lead
> II** — thank you. Q1 has been narrowed to the half that remains open, which
> is whether a **second** lead can be made available; that is now our
> highest-value question. Q4 is new: the sample rate we receive changed from
> ~89.5 Hz to 133.83 Hz, and we need to understand why.

> Rhythm regularity indicator. Not a diagnosis. Not validated for clinical use.
> Cannot distinguish atrial fibrillation from other causes of irregularity.

---

## BLOCKING

### Q1. Is a second lead available in **any** mode or firmware configuration?

> **UPDATED 2026-09-02.** The original Q1 asked two things: *which* lead we
> receive, and *whether a second* is available. **The first is now answered —
> `ecg_clean` is Lead II, confirmed, on both the raw and clean paths.** That
> closes the like-for-like question against the reference database (MITDB's
> MLII), and retroactively validates our detector comparison.
>
> **The second half is not answered, and it is the more important half.** It is
> restated below as the whole of Q1.

**The question, precisely:** can the patch deliver a **second, physically
independent lead** — in the current firmware, in any other mode, in a
diagnostic or engineering configuration, or in a build you could produce? We
are not asking only about the mode we are running today.

Supporting details we still need if the answer is yes:

- How is it structured — interleaved in `ecg_clean`, a second array, a second key?
- Which lead is it (I or III)?
- Is the sample rate **per channel**, or shared across two channels?
- If two leads exist at the device or app layer but only one reaches us, where
  is the second dropped? Recovering it may need no firmware change at all.

**Why this is the highest-value question we have.**

Our beat detector finds **~4.4% too many beats** on patch data, and each false
beat splits one RR interval into two, manufacturing irregularity. Our defence
against this is a signal-quality gate built on **bSQI** — agreement between two
independent R-peak detection *algorithms*. On device data that gate collapses,
and the reason is structural:

> **Two algorithms on one waveform fail together on the same noise. Two
> physical leads do not.**

Both detectors see the same motion artefact and agree on the same wrong peak,
so bSQI stays high and the gate passes a window it should refuse. No change of
algorithm fixes this, because the failure is *correlation between the two
checks*, not the quality of either one. A second physically independent lead
breaks that correlation: a real heartbeat appears in both leads, a movement
artefact usually does not. That is the standard way multi-lead systems suppress
precisely the false-beat problem that is blocking us.

**And it is the only lever on this blocker that does not require Q2.**
Everything else device-side waits on the controlled reference recording. This
one could be answered from documentation today, and if a second lead is
recoverable it would attack the over-detection using **data we already hold**.

**A second reason it matters, from waveform work.** Morphology analysis is
lead-dependent by definition. Lead II is now confirmed, which unblocks that in
principle — but single-lead morphology cannot distinguish atrial from
ventricular ectopy, and a second lead is what would make that separation
approachable at all.

---

### Q2. Can we get one controlled recording with a beat-by-beat reference?

**This is the single blocking item for the whole project, and no amount of
engineering substitutes for it.**

**The ask:** one existing subject wears the **ProRhythm patch** and a **reference
device that exports per-beat RR intervals** (Polar H10 or equivalent, ~1 ms) at
the same time.

- 5 minutes rest before recording
- **10 minutes seated, still, no talking**, electrode contact verified
- **2 minutes of deliberate movement at the end** — gives a labelled artefact
  segment for tuning the quality gate
- **Wall-clock time noted at both device start times**

Under an hour of one person's time. No clinician needed.

**Why it blocks:** our beat detector finds **~4.4% too many beats** on patch
data. Each false beat splits one interval into two, so healthy adults score a
variability of **0.19** where genuinely healthy people in reference data score
**0.03** — six times higher, on a healthier group. We cannot fix this because
**we have no ground truth for any patch recording**, and without it we could not
tell a corrected detector from one overfitted to noise.

**Important:** the device's own reported heart rate is **not** a substitute. It
revealed the problem, but it is a smoothed average over ~12 s. It can show the
beat *count* is wrong; it cannot show where each beat truly is.

---

### Q3. Can `sNo` and session fields be written into captured files?

**What we know:** the live WebSocket packet carries `sNo` (sequence number),
`start_time`, `end_time`, `duration`, `vitals_status`. **None of these appear in
any captured CSV.**

**The ask:** log them per packet, plus samples-per-packet.

**Why it matters:** without `sNo` we *infer* replay from a ~13,100 ms backward
timestamp step, and *estimate* packet loss from clock-versus-count. With it,
both become **counts**. It will not help the 37 existing captures, but it
compounds for every future one — and it should be in place **before** the Q2
recording, so that recording captures it.

---

### Q4. The sample rate changed. Was the previous loss known, and what changed?

**NEW, 2026-09-02.** Every capture we hold from 2026-08-07 to 2026-08-20
delivers **~89.5 Hz**. The session recorded on **2026-09-02 delivers 133.83
Hz**, inside your stated 120–140 Hz band.

`89.50 / 133.83 = 0.6688` — **two thirds**, to within our measurement scatter.
The earlier captures were receiving **one sample in three fewer** than the
device produced.

We can tell you where it was *not* happening. Our clock-versus-count residual
(elapsed time × rate, minus samples counted) has a median of **−1.8 samples
over a 45-second window** across 1,210 windows. There was no timing
discrepancy to find. So the loss happened **upstream of packet timestamping** —
systematic 2:3 decimation somewhere in the device or app path, not Bluetooth
packet loss, which would have left a clear trace.

**What we need to know:**

1. Was the 2:3 reduction **intentional** — a bandwidth or power mode?
2. **What changed** between 2026-08-20 and 2026-09-02? A firmware update, an
   app update, a configuration change, a different capture path?
3. **Can the rate vary by mode, battery state, connection quality, or
   subject?** This is the part that changes our engineering.
4. Is 133.83 Hz **per channel**, or shared if a second channel is ever enabled?

**Why it matters, and why it is blocking rather than useful.**

The good news first: this did **not** corrupt our RR timing. We compute
intervals as `Δindex / fs_local`, measuring the rate per window rather than
assuming it, so 89.5 delivered samples still span one real second and the
arithmetic cancelled. Our regularity threshold is likewise unaffected — the
timing grid is a negligible term at the variability levels we work at.

What it cost is **resolution**: 11.15 ms sample spacing instead of 7.47 ms, so
a QRS complex spanned ~9 samples instead of ~13. Re-testing our detector at
both rates on reference data shows the extra resolution does not broadly
improve beat detection, but it **substantially rescues the cases where
over-detection has gone catastrophic** — which is exactly our device failure
mode.

It is blocking because of question 3. **If the rate can vary, then measuring it
per session is mandatory rather than defensive**, and any fixed-rate assumption
anywhere in a downstream system is a latent fault. We would also need to treat
our 37 historical captures as a permanently lower-resolution dataset, which we
already do — they carry 89.5 Hz of information and we will not pretend
otherwise.

**One caveat on our side:** we have **one** session at 133.83 Hz. We are not
treating it as the new normal until several more agree, and we would welcome
confirmation of what the device is specified to deliver.

---

## USEFUL, NOT BLOCKING

### Q5. What are the amplitude units?

Not mV — probably ADC counts. Every quality measure we use is scale-relative and
the rhythm maths is timing-only, so this has never caused a problem. But it is
an open item, and it would be needed for any absolute-amplitude work.

### Q6. Where exactly are the electrodes placed on the torso?

A chest patch cannot reach the actual limbs, so RA/LA/LL are torso-placed
equivalents. This affects how comparable the signal is to reference databases.

### Q7. What filtering does the firmware apply?

The field is named `ecg_clean`, and we measured that adding a second filter
chain leaves only **6–7% of the amplitude**. We therefore apply none. Knowing
the actual passband would let us confirm this is the right call rather than an
empirical one.

### Q8. Is the WebSocket message schema stable?

We now parse the real schema (`ecg_clean: [{e, t}, ...]`). An earlier client
assumed a different shape and, on a real packet, would have **connected
successfully, logged no error, and produced zero readings**. Our parser now
refuses unrecognised packets rather than failing silently — but we should be
told before the schema changes.

### Q9. Can we get a recording from someone with a known abnormal rhythm?

Every one of our 37 patch recordings is from a healthy adult — **a single
class**. A decision boundary cannot be derived from one class. Public data
supplies the other side today, which is the project's central methodological
dependency. Even one patch recording from a known irregular rhythm would be
independently valuable. This likely needs clinical involvement, so it is a
separate ask from Q2.

---

## Summary for the patch team

| # | Ask | Effort | Unblocks |
|---|---|---|---|
| **Q1** | Is a **second lead** available, in any mode? | An answer | The over-detection blocker — using data we already hold |
| **Q2** | One paired recording, ~1 hour | One person, one hour | **Everything device-side** |
| **Q3** | Log `sNo` in captures | One line in the capture path | Measured replay and packet loss |
| **Q4** | Why did the sample rate change, and can it vary? | An answer | Whether per-session rate measurement is mandatory |
| Q5–Q8 | Documentation answers | Low | Closes open items |
| Q9 | An abnormal-rhythm recording | Needs clinical involvement | Device-side boundary derivation |

**If only one is possible, it is Q2.** It has been the rate-limiting item for the
entire project.

**If only one *answer* is possible — no recording, no logging, just a reply from
someone who knows the hardware — it is Q1.** It costs you a sentence, and it is
the only route to the over-detection problem that does not require new data.
