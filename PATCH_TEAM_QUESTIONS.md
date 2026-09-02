# Questions for the ProRhythm patch team

**Short answer to "do we need to clarify things before building?": YES — three
of these block work, and two of them cannot be answered by us at any cost.**

Ordered by how much each unblocks. Q1–Q3 are blocking. Q4–Q8 are useful.

> Rhythm regularity indicator. Not a diagnosis. Not validated for clinical use.
> Cannot distinguish atrial fibrillation from other causes of irregularity.

---

## BLOCKING

### Q1. Which lead does `ecg_clean` carry — and is a second lead available?

**What we know:** the patch has **3 electrodes (RA, LA, LL)** producing a
**2-lead ECG**. Every file we hold has **one** amplitude channel, and the live
packet has **one** `ecg_clean` array.

**What we need:**
- Does `ecg_clean` carry one channel or two? If two, how are they structured —
  interleaved, second array, second key?
- If one, **which lead is it — I, II or III?**
- If a second lead exists at device or app level, where is it dropped?
- Is 89.7 Hz **per channel**, or shared across two (~45 Hz each)?

**Why it blocks:**

1. We validated our beat detector against **modified Lead II** in the reference
   database, assuming the patch streams Lead II. **If it streams Lead I, that
   comparison is not like-for-like** — Lead I has a smaller R-wave in most
   people, which would partly explain why detection performs worse on patch data.
2. Waveform-shape analysis is **lead-dependent by definition**. It cannot be
   applied to an unknown projection at all.
3. **A second lead would attack our main blocker directly.** Our quality check
   currently runs two *algorithms* on the *same* waveform, and it fails because
   both make the same mistake on the same noise. Two *physically independent
   leads* is a far stronger check — a real heartbeat appears in both, a movement
   artefact usually does not. That is the standard way multi-lead systems
   suppress exactly the false-beat problem we have.

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

## USEFUL, NOT BLOCKING

### Q4. What are the amplitude units?

Not mV — probably ADC counts. Every quality measure we use is scale-relative and
the rhythm maths is timing-only, so this has never caused a problem. But it is
an open item, and it would be needed for any absolute-amplitude work.

### Q5. Where exactly are the electrodes placed on the torso?

A chest patch cannot reach the actual limbs, so RA/LA/LL are torso-placed
equivalents. This affects how comparable the signal is to reference databases.

### Q6. What filtering does the firmware apply?

The field is named `ecg_clean`, and we measured that adding a second filter
chain leaves only **6–7% of the amplitude**. We therefore apply none. Knowing
the actual passband would let us confirm this is the right call rather than an
empirical one.

### Q7. Is the WebSocket message schema stable?

We now parse the real schema (`ecg_clean: [{e, t}, ...]`). An earlier client
assumed a different shape and, on a real packet, would have **connected
successfully, logged no error, and produced zero readings**. Our parser now
refuses unrecognised packets rather than failing silently — but we should be
told before the schema changes.

### Q8. Can we get a recording from someone with a known abnormal rhythm?

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
| **Q1** | Which lead(s) reach us | An answer | Waveform work; possibly the detector |
| **Q2** | One paired recording, ~1 hour | One person, one hour | **Everything device-side** |
| **Q3** | Log `sNo` in captures | One line in the capture path | Measured replay and packet loss |
| Q4–Q7 | Documentation answers | Low | Closes open items |
| Q8 | An abnormal-rhythm recording | Needs clinical involvement | Device-side boundary derivation |

**If only one is possible, it is Q2.** It has been the rate-limiting item for the
entire project.
