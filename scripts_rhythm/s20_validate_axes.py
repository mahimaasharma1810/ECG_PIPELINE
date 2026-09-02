"""Step 7 - held-out DATABASE validation of the combination layer.

The claim to test: seeing RATE and REGULARITY together catches windows that a
regularity-only view calls normal. Measured on databases the axes were not
derived from.
"""
import sys, warnings
from pathlib import Path

import numpy as np, pandas as pd, wfdb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.degrade import DEVICE_FS
from rhythm.features import window_features, BEATS_PER_WINDOW
from rhythm.verdict import decide, REGULAR
from rhythm.axes import assess_rate, assess_events, combine_axes
from rhythm.axes.rate import TACHYCARDIC, BRADYCARDIC

warnings.filterwarnings("ignore")
G = 1000.0 / DEVICE_FS
BS = set("NLRBAaJSVrFejnE/fQ?")


def clean(a): return (a or "").replace("\x00", "").strip()


def windows(dbpath, require_mlii=False):
    rows = []
    for r in sorted(p.stem for p in Path(dbpath).glob("*.hea")):
        h = wfdb.rdheader(f"{dbpath}/{r}")
        if require_mlii and "MLII" not in h.sig_name:
            continue
        ann = wfdb.rdann(f"{dbpath}/{r}", "atr")
        keep = np.array([s in BS for s in ann.symbol])
        bs = ann.sample[keep]; sym = np.array(ann.symbol, dtype=object)[keep]
        marks = [(s, clean(a)) for s, a in zip(ann.sample, ann.aux_note)
                 if clean(a).startswith("(")]
        if marks:
            ms = np.array([m[0] for m in marks]); lb = np.array([m[1] for m in marks], dtype=object)
            i = np.searchsorted(ms, bs, side="right") - 1
            rl = np.where(i >= 0, lb[np.clip(i, 0, None)], "")
        else:
            rl = np.array([""] * bs.size, dtype=object)
        t = np.round((bs / float(h.fs) * 1000.0) / G) * G
        for s in range(0, bs.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
            rr = np.diff(t[s:s + BEATS_PER_WINDOW])
            f = window_features(rr)
            if not f.usable:
                continue
            v = decide(f, True, "ok")
            ra = assess_rate(f, True); ev = assess_events(rr, f, True)
            c = combine_axes(ra, v, ev, clinical_permitted=True)
            w = sym[s:s + BEATS_PER_WINDOW]
            u = set(rl[s:s + BEATS_PER_WINDOW])
            rows.append(dict(record=r, rhythm=(u.pop() if len(u) == 1 else "mixed"),
                             hr=f.hr_bpm, rr_cv=f.rr_cv, verdict=v.verdict,
                             rate=ra.state, burden=ev.burden, pauses=ev.pause_count,
                             severity=c.severity, escalate=c.escalate,
                             true_ectopic=int(np.sum([x in ("S","A","a","J","V","E") for x in w]))))
    return pd.DataFrame(rows)


print("=" * 74)
print("HELD-OUT VALIDATION of the combination layer")
print("=" * 74)
for name, path, mlii in (("MITDB", "data/raw/public/mitdb", True),
                         ("SVDB", "data/raw/public/svdb", False)):
    x = windows(path, mlii)
    x.to_csv(f"reports_rhythm/s20_axes_{name.lower()}.csv", index=False)
    print(f"\n--- {name}: {len(x):,} windows, {x.record.nunique()} records ---")
    print(f"\n  severity distribution:")
    for k, v in x.severity.value_counts().items():
        print(f"    {k:<10} {v:>6,} ({v/len(x):>5.1%})")

    # THE CLAIM: regular-but-tachycardic windows a regularity-only view misses
    gap = x[(x.verdict == REGULAR) & (x.rate == TACHYCARDIC)]
    print(f"\n  REGULAR + TACHYCARDIC (the SVT gap):")
    print(f"    {len(gap):,} windows ({len(gap)/len(x):.1%})")
    print(f"    regularity-only view would call these NORMAL")
    print(f"    combination layer calls them: {dict(gap.severity.value_counts())}")
    if len(gap):
        print(f"    their HR: median {gap.hr.median():.0f} bpm, max {gap.hr.max():.0f} bpm")
        print(f"    their true ectopic beats: median {gap.true_ectopic.median():.0f}")

    reg_only_normal = int(((x.verdict == REGULAR)).sum())
    comb_normal = int((x.severity == "NORMAL").sum())
    print(f"\n  windows a REGULARITY-ONLY view would pass as normal: {reg_only_normal:,}")
    print(f"  windows the COMBINATION layer passes as NORMAL      : {comb_normal:,}")
    print(f"  -> {reg_only_normal - comb_normal:,} additional windows flagged "
          f"({(reg_only_normal-comb_normal)/max(reg_only_normal,1):.1%} of them)")

    esc = x[x.escalate]
    print(f"\n  escalating (CRITICAL): {len(esc):,} ({len(esc)/len(x):.2%}) - "
          f"rate extremes and pauses only")

print(f"\n{SCOPE_STATEMENT}")
