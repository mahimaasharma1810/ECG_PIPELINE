"""Real-time runner: WebSocket (live) or CSV replay (test), same engine.

    # live
    python scripts_rhythm/s17_realtime.py --ws ws://host:port/ecg

    # replay an existing capture at wall-clock speed (or faster)
    python scripts_rhythm/s17_realtime.py --replay prorithm_ecg/1789/2026-08-17_08.csv --speed 60

Clinical severity and the MedGemma report are governed by the device-domain
gate. Until beat timing is validated against a controlled reference recording,
the stream emits ENGINEERING output only: rhythm verdicts with evidence, no
severity, no MedGemma call.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.streaming import RhythmStream
from rhythm.pipeline import detect_peaks
from rhythm.domain_gate import read_gate
from rhythm.narrative import generate
from rhythm.axes.rate import PROVENANCE as RATE_PROVENANCE
from rhythm.axes.events import PROVENANCE as EVENTS_PROVENANCE
from rhythm.verdict import THRESHOLD_PROVENANCE
from rhythm.ws_schema import parse_packet, parse_vitals, session_meta, PacketStats, SchemaError

warnings.filterwarnings("ignore")


def replay_source(path, speed):
    """Yield (t_ms, amplitude) in file order, paced to wall clock / speed."""
    df = pd.read_csv(path, usecols=["timestamp_ms", "amplitude"])
    t = df.timestamp_ms.to_numpy(float); x = df.amplitude.to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(x); t, x = t[ok], x[ok]
    t0_stream, t0_wall = t[0], time.perf_counter()
    for ti, xi in zip(t, x):
        if speed > 0:
            due = (ti - t0_stream) / 1000.0 / speed
            lag = due - (time.perf_counter() - t0_wall)
            if lag > 0.002:
                time.sleep(lag)
        yield ti, xi


def ws_source(url, stats: PacketStats, meta_sink: dict):
    """Yield (t_ms, amplitude) from the ProRhythm WebSocket stream.

    Parses the REAL device schema (ecg_clean: [{e, t}, ...]) via
    rhythm.ws_schema. A schema mismatch raises rather than yielding nothing -
    the previous assumed schema would have produced a silently empty stream.
    """
    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("pip install websockets  (needed for --ws)")
    n_bad = 0
    with connect(url) as ws:
        for msg in ws:
            try:
                d = json.loads(msg)
            except Exception:
                continue
            if not meta_sink:
                meta_sink.update(session_meta(d))
                v = parse_vitals(d)
                if v:
                    meta_sink["first_vitals"] = v
            try:
                yield from parse_packet(d, stats)
            except SchemaError as e:
                n_bad += 1
                if n_bad <= 3:
                    print(f"  [SCHEMA] {e}")
                if n_bad == 20:
                    raise SystemExit(
                        "20 packets failed schema validation - the device schema "
                        "has changed. Refusing to continue on a stream we cannot "
                        "parse (a silently empty stream is worse than stopping).")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ws", help="ProRhythm WebSocket URL")
    g.add_argument("--replay", help="capture CSV to replay")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed multiplier; 0 = as fast as possible")
    ap.add_argument("--analyse-every", type=float, default=2.0,
                    help="seconds of wall clock between analysis passes")
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--out", default=None, help="write updates as JSONL")
    a = ap.parse_args()

    gate = read_gate()
    print(f"device-domain gate : {gate.state}")
    print(f"  rhythm verdict   : {'permitted' if gate.may_emit_rhythm_verdict else 'BLOCKED'}")
    print(f"  clinical severity: {'permitted' if gate.may_emit_clinical_severity else 'BLOCKED'}")
    print(f"  MedGemma report  : {'permitted' if gate.may_call_medgemma else 'BLOCKED'}")
    print(f"  threshold status : {THRESHOLD_PROVENANCE['status']}")
    print(f"  rate axis status : {RATE_PROVENANCE['status']} "
          f"({RATE_PROVENANCE['extreme_band_basis']})")
    print(f"  events axis      : {EVENTS_PROVENANCE['status']}")
    print(f"\n{gate.reason}\n" if not gate.may_call_medgemma else "")

    pstats, meta = PacketStats(), {}
    src = ws_source(a.ws, pstats, meta) if a.ws else replay_source(a.replay, a.speed)
    stream = RhythmStream()
    fh = open(a.out, "w") if a.out else None

    n_samp, n_upd, last = 0, 0, 0.0
    hdr = (f"{'stream t':>12} {'RATE':<13} {'REGULARITY':<20} {'EVENTS':<14} "
           f"{'severity':<10} {'HR':>6} {'RR CV':>7} {'bSQI':>6} {'lat ms':>7}")
    print(hdr); print("-" * len(hdr))
    t_start = time.perf_counter()
    for ts, amp in src:
        stream.push(ts, amp); n_samp += 1
        now = time.perf_counter()
        if now - last < a.analyse_every:
            continue
        last = now
        u = stream.analyse(detect_peaks)
        if u is None:
            continue
        n_upd += 1
        nar = None
        if gate.may_call_medgemma:
            r = generate(u.verdict.verdict, u.verdict.reason, u.verdict.evidence)
            nar = dict(text=r.text, source=r.source, accepted=r.accepted)
        ev = f"{u.events.burden[:4]}/{u.events.ectopic_couplets}c"
        if u.events.pause_count:
            ev += f"/{u.events.pause_count}p"
        print(f"{u.t_ms/1000.0:>12.1f} {u.rate.state:<13} {u.verdict.verdict:<20} "
              f"{ev:<14} {u.severity.severity:<10} {u.features.hr_bpm:>6.1f} "
              f"{u.features.rr_cv:>7.4f} {u.sqi.bsqi:>6.3f} {u.latency_ms:>7.1f}")
        if fh:
            fh.write(json.dumps(dict(
                t_ms=u.t_ms, verdict=u.verdict.verdict,
                actionable=u.verdict.actionable, action_state=u.verdict.action_state,
                reported_state=u.reported_state,
                rate=u.rate.as_dict(), events=u.events.as_dict(),
                severity=u.severity.severity, severity_reason=u.severity.reason,
                escalate=u.severity.escalate, findings=u.severity.findings,
                features=u.features.as_dict(), sqi=u.sqi.as_dict(),
                fs_local=u.fs_local, latency_ms=u.latency_ms,
                replayed_dropped=u.n_replayed_dropped,
                gate=gate.state, narrative=nar,
                scope_statement=SCOPE_STATEMENT), default=str) + "\n")
        if a.max_updates and n_upd >= a.max_updates:
            break

    el = time.perf_counter() - t_start
    if a.ws:
        d = pstats.as_dict()
        print(f"\npacket stats (from sNo, not inferred):")
        print(f"  packets            : {d['n_packets']:,}")
        print(f"  replayed (sNo seen): {d['n_seq_repeat']:,} "
              f"({d['replay_packet_fraction']:.1%})")
        print(f"  packet loss        : {d['n_packets_lost']:,} over "
              f"{d['n_seq_gap']:,} gaps ({d['packet_loss_fraction']:.1%})")
        print(f"  malformed samples  : {d['n_malformed']:,}")
        if meta:
            print(f"  session            : "
                  f"{ {k: meta[k] for k in list(meta)[:5]} }")
    print(f"\nsamples ingested   : {n_samp:,}")
    print(f"replayed rows dropped: {stream._n_replayed:,} "
          f"({stream._n_replayed/max(n_samp+stream._n_replayed,1):.1%} of arrivals)")
    print(f"updates emitted    : {n_upd}   wall clock {el:.1f}s")
    if fh: fh.close(); print(f"wrote {a.out}")
    print(f"\n{SCOPE_STATEMENT}")


if __name__ == "__main__":
    main()
