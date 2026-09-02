"""L5 - PVC morphology features, chosen for LEAD ROBUSTNESS.

The device lead configuration is OPEN (3 electrodes RA/LA/LL -> 2-lead, which
lead reaches us is unresolved). Any feature requiring a KNOWN lead can never
transfer, so every feature here is either scale-free, patient-relative, or a
duration - none depend on knowing the projection.

WHY PVC ONLY, NOT PAC. A PAC conducts through the normal pathway, so its QRS
is morphologically near-identical to a sinus beat. The discriminating feature
is the P wave: small, frequently buried in the preceding T wave, and
unreliable on a single lead. The prior attempt scored S-class F1 0.152 for
exactly this reason. PACs are handled by L4 from TIMING instead.

L5 IS PURE MORPHOLOGY. No RR-interval feature appears here, deliberately, so
that L5's contribution stays independent of L3/L4 and the combination layer
means something.

FEATURE JUSTIFICATION (lead-robustness stated per feature):

  qrs_width_norm     Duration above a fraction of the beat's own peak. A
                     DURATION, so it is invariant to lead polarity and gain.
                     The primary PVC discriminator: ventricular origin spreads
                     cell-to-cell rather than via His-Purkinje, so the complex
                     is wider.
  template_corr      Correlation against the patient's OWN median sinus beat.
                     Patient-relative and amplitude-normalised, so it measures
                     "unlike this person's normal beat" rather than any
                     absolute shape.
  template_corr_abs  As above on the rectified signal, so a polarity flip
                     between leads does not by itself look abnormal.
  pos_neg_ratio      Positive excursion over total excursion. A RATIO.
  slope_norm         Max |first difference| divided by peak amplitude. A RATIO.
  energy_conc        Fraction of beat energy inside the central QRS window.
                     Scale-free; a wide complex spreads its energy.
  skew, kurt         Shape statistics of the amplitude-normalised segment.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis

FEATURE_NAMES = ("qrs_width_norm", "template_corr", "template_corr_abs",
                 "pos_neg_ratio", "slope_norm", "energy_conc", "seg_skew",
                 "seg_kurt")

PRE_MS, POST_MS = 200.0, 300.0     # segment around the annotated R peak
QRS_HALF_MS = 60.0                 # central window for energy concentration


def _norm(x):
    x = np.asarray(x, float)
    x = x - np.median(x)
    s = np.max(np.abs(x))
    return x / s if s > 0 else x


def seg_len(fs) -> int:
    """Fixed segment length in samples, so all segments stack.

    Rounding PRE/POST independently gives lengths that differ by one sample
    depending on index parity - harmless at 360 Hz, fatal when building a
    median template at 89.7 Hz where the arrays must align.
    """
    return int(round((PRE_MS + POST_MS) / 1000.0 * fs))


def segment(sig, idx, fs):
    n = seg_len(fs)
    a = int(round(idx - PRE_MS / 1000.0 * fs))
    b = a + n
    if a < 0 or b > sig.size:
        return None
    return sig[a:b]


def build_template(sig, normal_idx, fs, max_beats=200):
    """Patient's own median sinus beat, amplitude-normalised."""
    segs = []
    for i in normal_idx[:max_beats]:
        s = segment(sig, i, fs)
        if s is not None:
            segs.append(_norm(s))
    if len(segs) < 5:
        return None
    return np.median(np.stack(segs), axis=0)


def extract(sig, idx, fs, template=None) -> dict:
    s = segment(sig, idx, fs)
    if s is None:
        return {k: np.nan for k in FEATURE_NAMES}
    z = _norm(s)
    n = z.size
    centre = int(PRE_MS / 1000.0 * fs)
    half = max(int(QRS_HALF_MS / 1000.0 * fs), 2)

    core = z[max(centre - half, 0):min(centre + half, n)]
    peak = np.max(np.abs(core)) if core.size else 0.0
    # width: samples in the core exceeding half the beat's own peak
    width = float(np.sum(np.abs(core) > 0.5 * peak)) / fs * 1000.0 if peak > 0 else np.nan

    d = np.diff(z)
    tot_e = float(np.sum(z ** 2))
    core_e = float(np.sum(core ** 2))

    if template is not None and template.size == n:
        c = float(np.corrcoef(z, template)[0, 1])
        ca = float(np.corrcoef(np.abs(z), np.abs(template))[0, 1])
    else:
        c = ca = np.nan

    pos = float(np.sum(z[z > 0])); neg = float(-np.sum(z[z < 0]))
    return {
        "qrs_width_norm": width,
        "template_corr": c,
        "template_corr_abs": ca,
        "pos_neg_ratio": pos / (pos + neg) if (pos + neg) > 0 else np.nan,
        "slope_norm": float(np.max(np.abs(d))) if d.size else np.nan,
        "energy_conc": core_e / tot_e if tot_e > 0 else np.nan,
        "seg_skew": float(skew(z)),
        "seg_kurt": float(kurtosis(z, fisher=False)),
    }
