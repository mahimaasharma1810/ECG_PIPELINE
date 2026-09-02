"""Beat-timing validation against a controlled reference recording.

Consumes a paired recording: the ProRhythm patch and a reference device that
exports per-beat RR intervals (Polar H10 or equivalent, ~1 ms resolution),
worn simultaneously.

This is the measurement the whole device side is blocked on. Until it runs on
real paired data, device R-peak detection is unvalidated and no device-domain
threshold can be trusted.

Two problems it must solve:

  1. CLOCK ALIGNMENT. The two devices have independent clocks. Wall-clock start
     times get us close; the residual offset is found by maximising agreement
     between the two beat trains over a search range.

  2. WHAT "AGREEMENT" MEANS. Beat-level Se/PPV is necessary but not sufficient:
     a detector can find every beat and still misplace it enough to corrupt RR.
     The binding criterion is therefore the error in RR CV - the quantity the
     threshold actually consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np


# --- PASS CRITERIA ---------------------------------------------------------
# Beat detection must be accurate AND precise enough that its contribution to
# RR CV stays inside the uncertainty already budgeted by the refusal band
# (+/-0.027). If detection error exceeds that budget, the band is a fiction.
MATCH_TOL_MS = 100.0        # tighter than the 150 ms AAMI detection tolerance,
                            # because we care about timing, not just presence
MIN_SENSITIVITY = 0.95
MIN_PPV = 0.95
MAX_CV_ERROR_P95 = 0.027    # == refusal-band half width
MAX_RR_BIAS_MS = 10.0


@dataclass(frozen=True)
class TimingValidation:
    n_reference_beats: int
    n_device_beats: int
    offset_ms: float
    sensitivity: float
    ppv: float
    rr_bias_ms: float
    rr_loa_lower_ms: float      # Bland-Altman limits of agreement
    rr_loa_upper_ms: float
    cv_error_median: float
    cv_error_p95: float
    n_windows: int
    passed: bool
    failures: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return asdict(self)


def _match(a_ms: np.ndarray, b_ms: np.ndarray, tol: float):
    """Greedy nearest matching. Returns (matched_pairs, n_a_unmatched, n_b_unmatched)."""
    a = np.sort(np.asarray(a_ms, float)); b = np.sort(np.asarray(b_ms, float))
    used = np.zeros(b.size, bool); pairs = []
    j = 0
    for t in a:
        while j < b.size and b[j] < t - tol:
            j += 1
        k, best, bd = j, -1, tol
        while k < b.size and b[k] <= t + tol:
            if not used[k] and abs(b[k] - t) <= bd:
                bd, best = abs(b[k] - t), k
            k += 1
        if best >= 0:
            used[best] = True
            pairs.append((t, b[best]))
    return np.array(pairs), a.size - len(pairs), int((~used).sum())


def align(ref_beats_ms: np.ndarray, dev_beats_ms: np.ndarray,
          search_ms: float = 30000.0, step_ms: float = 20.0) -> float:
    """Find the clock offset that maximises beat-train agreement.

    Coarse scan then refine. Returns the offset to ADD to device times.
    """
    ref = np.sort(np.asarray(ref_beats_ms, float))
    dev = np.sort(np.asarray(dev_beats_ms, float))
    if ref.size < 10 or dev.size < 10:
        return 0.0
    base = ref[0] - dev[0]
    best_off, best_n = base, -1
    for coarse in np.arange(base - search_ms, base + search_ms + 1, 250.0):
        pairs, _, _ = _match(ref, dev + coarse, MATCH_TOL_MS)
        if len(pairs) > best_n:
            best_n, best_off = len(pairs), coarse
    for fine in np.arange(best_off - 250.0, best_off + 250.0 + 1, step_ms):
        pairs, _, _ = _match(ref, dev + fine, MATCH_TOL_MS)
        if len(pairs) > best_n:
            best_n, best_off = len(pairs), fine
    return float(best_off)


def validate(ref_beats_ms: np.ndarray, dev_beats_ms: np.ndarray,
             beats_per_window: int = 64) -> TimingValidation:
    """Full beat-timing validation of the device against the reference."""
    ref = np.sort(np.asarray(ref_beats_ms, float))
    dev = np.sort(np.asarray(dev_beats_ms, float))
    off = align(ref, dev)
    dev_a = dev + off

    pairs, n_fn, n_fp = _match(ref, dev_a, MATCH_TOL_MS)
    tp = len(pairs)
    se = tp / ref.size if ref.size else float("nan")
    ppv = tp / dev_a.size if dev_a.size else float("nan")

    # RR agreement on matched beats (Bland-Altman)
    if tp >= 3:
        rr_ref = np.diff(pairs[:, 0]); rr_dev = np.diff(pairs[:, 1])
        d = rr_dev - rr_ref
        bias = float(np.mean(d)); sd = float(np.std(d, ddof=1))
        loa = (bias - 1.96 * sd, bias + 1.96 * sd)
    else:
        bias, loa = float("nan"), (float("nan"), float("nan"))

    # CV error per window - the binding criterion
    cv_err, n_win = [], 0
    step = beats_per_window
    for s in range(0, ref.size - beats_per_window + 1, step):
        w = ref[s:s + beats_per_window]
        t0, t1 = w[0], w[-1]
        d_in = dev_a[(dev_a >= t0) & (dev_a <= t1)]
        if d_in.size < 10:
            continue
        rr_r = np.diff(w); rr_d = np.diff(d_in)
        if rr_r.size < 2 or rr_d.size < 2:
            continue
        cv_err.append(abs(rr_d.std(ddof=1) / rr_d.mean() - rr_r.std(ddof=1) / rr_r.mean()))
        n_win += 1
    cv_err = np.array(cv_err) if cv_err else np.array([np.nan])

    failures = []
    if not (se >= MIN_SENSITIVITY):
        failures.append(f"sensitivity {se:.4f} < {MIN_SENSITIVITY}")
    if not (ppv >= MIN_PPV):
        failures.append(f"PPV {ppv:.4f} < {MIN_PPV}")
    if not (abs(bias) <= MAX_RR_BIAS_MS):
        failures.append(f"RR bias {bias:+.1f} ms exceeds +/-{MAX_RR_BIAS_MS:.0f} ms")
    p95 = float(np.nanpercentile(cv_err, 95))
    if not (p95 <= MAX_CV_ERROR_P95):
        failures.append(f"CV error p95 {p95:.4f} > {MAX_CV_ERROR_P95} "
                        f"(the refusal-band budget)")

    return TimingValidation(
        n_reference_beats=int(ref.size), n_device_beats=int(dev.size),
        offset_ms=off, sensitivity=float(se), ppv=float(ppv),
        rr_bias_ms=bias, rr_loa_lower_ms=float(loa[0]), rr_loa_upper_ms=float(loa[1]),
        cv_error_median=float(np.nanmedian(cv_err)), cv_error_p95=p95,
        n_windows=n_win, passed=not failures, failures=tuple(failures))
