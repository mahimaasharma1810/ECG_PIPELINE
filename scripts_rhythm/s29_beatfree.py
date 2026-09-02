"""Can rhythm be classified WITHOUT finding individual beats?

Motivation: our device failure is that motion artefact gets COUNTED AS A BEAT
(~4.4% excess). A method that never counts beats cannot make that mistake.

Three approaches compared on identical FIXED-TIME windows (30 s), so nothing
here depends on beat detection to define the window either:

  A. BEAT-BASED   detect peaks -> RR intervals -> RR CV        (what we do now)
  B. AUTOCORR     autocorrelation peak sharpness               (beat-free)
  C. SPECTRAL     concentration of power at the dominant rate  (beat-free)

Then the decisive test: inject DEVICE-LIKE MOTION ARTEFACT and see which
degrades least. Artefact is modelled on what was measured in the captures -
spikes reaching ~8x baseline amplitude.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, wfdb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from rhythm import SCOPE_STATEMENT
from rhythm.degrade import resample_to, DEVICE_FS_NATIVE
import neurokit2 as nk

WIN_S = 30.0
BEAT = set("NLRBAaJSVrFejnE/fQ?")
rng = np.random.default_rng(20260902)


def add_motion(x, fs, n_per_min=6.0, amp=8.0):
    """Device-like motion artefact: brief high-amplitude excursions."""
    y = x.copy()
    p95 = np.percentile(np.abs(x - np.median(x)), 95)
    n = int(n_per_min * len(x) / fs / 60)
    for _ in range(n):
        i = rng.integers(0, max(len(y) - int(fs), 1))
        w = int(rng.uniform(0.15, 0.5) * fs)
        t = np.linspace(0, np.pi, w)
        y[i:i+w] += amp * p95 * np.sin(t) * rng.choice([-1, 1])
    return y


def feat_beat(x, fs):
    """A: beat detection -> RR CV."""
    try:
        pk = np.asarray(nk.ecg_peaks(x, sampling_rate=fs)[1]["ECG_R_Peaks"], int)
    except Exception:
        return np.nan
    if pk.size < 5:
        return np.nan
    rr = np.diff(pk) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    return float(rr.std(ddof=1) / rr.mean()) if rr.size >= 4 else np.nan


def feat_autocorr(x, fs):
    """B: beat-free. Height of the autocorrelation peak at the dominant lag.

    A regular rhythm repeats, so the signal correlates strongly with itself one
    beat later. An irregular one does not. Returns 1 - peak, so HIGHER = more
    irregular, matching RR CV's direction.
    """
    z = x - np.mean(x)
    if np.std(z) == 0:
        return np.nan
    ac = np.correlate(z, z, mode="full")[len(z)-1:]
    ac /= ac[0]
    lo, hi = int(fs * 0.3), min(int(fs * 2.0), len(ac) - 1)   # 30-200 bpm
    if hi <= lo:
        return np.nan
    return float(1.0 - np.max(ac[lo:hi]))


def feat_spectral(x, fs):
    """C: beat-free. How concentrated power is at the dominant rate.

    A regular rhythm puts its energy in a sharp peak; an irregular one smears
    it. Returns 1 - concentration, so HIGHER = more irregular.
    """
    from scipy import signal as sps
    z = x - np.mean(x)
    if np.std(z) == 0:
        return np.nan
    f, p = sps.welch(z, fs=fs, nperseg=min(len(z), int(fs * 10)))
    m = (f >= 0.5) & (f <= 3.5)          # 30-210 bpm fundamental
    if not m.any() or p[m].sum() <= 0:
        return np.nan
    pm, fm = p[m], f[m]
    peak = fm[np.argmax(pm)]
    near = np.abs(fm - peak) <= 0.15
    return float(1.0 - pm[near].sum() / pm.sum())


def build(motion):
    rows = []
    for rec in sorted(p.stem for p in Path("data/raw/public/mitdb").glob("*.hea")):
        h = wfdb.rdheader(f"data/raw/public/mitdb/{rec}")
        if "MLII" not in h.sig_name:
            continue
        ch = h.sig_name.index("MLII")
        sig = wfdb.rdrecord(f"data/raw/public/mitdb/{rec}", channels=[ch]).p_signal[:, 0]
        ann = wfdb.rdann(f"data/raw/public/mitdb/{rec}", "atr")
        marks = [(s, (a or "").replace("\x00", "").strip())
                 for s, a in zip(ann.sample, ann.aux_note)
                 if (a or "").replace("\x00", "").strip().startswith("(")]
        if not marks:
            continue
        ms = np.array([m[0] for m in marks]); lb = np.array([m[1] for m in marks], dtype=object)
        x, fs = resample_to(sig, float(h.fs), DEVICE_FS_NATIVE)
        keep = np.arange(x.size) % 3 != 2          # device 2:3 decimation
        x, fs = x[keep], fs * 2 / 3
        if motion:
            x = add_motion(x, fs)
        n = int(WIN_S * fs)
        for i in range(0, x.size - n, n):
            t0 = int(i / fs * float(h.fs))
            j = np.searchsorted(ms, t0, side="right") - 1
            lab = lb[j] if j >= 0 else ""
            y = {"(AFIB": 1, "(N": 0}.get(lab)
            if y is None:
                continue
            seg = x[i:i+n]
            rows.append(dict(record=rec, y=y,
                             beat=feat_beat(seg, fs),
                             autocorr=feat_autocorr(seg, fs),
                             spectral=feat_spectral(seg, fs)))
    return pd.DataFrame(rows).dropna()


def auc(s, y):
    r = pd.Series(s).rank().to_numpy(); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)


if __name__ == "__main__":
    print("CAN WE SKIP THE BEAT FINDER?")
    print(f"MITDB, {WIN_S:.0f}s FIXED-TIME windows, device signal path "
          f"(133.83 Hz -> 2:3 decimation)\n")
    for motion, tag in ((False, "CLEAN"), (True, "WITH DEVICE-LIKE MOTION ARTEFACT")):
        d = build(motion)
        y = d.y.to_numpy()
        print(f"=== {tag} === ({len(d):,} windows, {y.sum()} irregular)")
        print(f"  {'method':<28} {'AUC':>8}")
        for c, nm in (("beat", "A. beat-based (RR CV)"),
                      ("autocorr", "B. autocorrelation  [beat-free]"),
                      ("spectral", "C. spectral         [beat-free]")):
            print(f"  {nm:<28} {auc(d[c].to_numpy(), y):>8.4f}")
        d.to_csv(f"reports_rhythm/s29_beatfree_{'motion' if motion else 'clean'}.csv",
                 index=False)
        print()
    print(SCOPE_STATEMENT)
