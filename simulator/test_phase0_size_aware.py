"""
Phase 0 acceptance test.

Before Phase 0 the simulator charged every transmission a fixed 5 ms of
airtime and dropped whole messages with one coin flip, so message length
changed nothing. This test fails if that is still true.

Run:  python simulator/test_phase0_size_aware.py
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.network_simulator import NetworkSimulator
from simulator.node import SimNode


def build(n: int, seed: int = 42, ttl: int = 8, **sim_kwargs):
    area = math.sqrt(n) * 80.0
    sim = NetworkSimulator(
        width=area, height=area, packet_loss_rate=0.03, seed=seed, **sim_kwargs
    )
    for i in range(n):
        sim.add_node(
            SimNode(
                node_id=1000 + i,
                x=sim.rng.uniform(20, area - 20),
                y=sim.rng.uniform(20, area - 20),
                radio_range=110.0,
                heartbeat_interval=20.0,
                is_gateway=(i >= n - 2),
            )
        )
    return sim


def run(n: int, text: str, seed: int = 42, **sim_kwargs):
    sim = build(n, seed=seed, **sim_kwargs)
    sim.run_until(5.0)
    sim.trigger_sos(1000, text=text)
    sim.run_until(20.0)
    m = sim.metrics
    return {
        "delivered": m.total_sos_delivered_to_gateway,
        "tx": m.total_transmissions,
        "frags": m.total_fragments_tx,
        "frag_loss": m.fragment_loss_drops,
        "collisions": m.collision_drops,
        "airtime_s": round(m.total_airtime_sec, 4),
        "bytes": m.payload_bytes_tx,
    }


SHORT = "Scale 200 S"          # 11 bytes of text
LONG = "X" * 240               # 240 bytes of text

# Fields that describe network BEHAVIOUR. `bytes` is a pure observability
# counter added in Phase 0, so it moves with message length even in legacy
# mode and must not be part of the legacy-equivalence check.
BEHAVIOUR_KEYS = ("delivered", "tx", "frags", "frag_loss", "collisions", "airtime_s")


def behaviour(result: dict) -> dict:
    return {k: result[k] for k in BEHAVIOUR_KEYS}


def main() -> int:
    print("=" * 72)
    print("PHASE 0 ACCEPTANCE TEST — does message size change the outcome?")
    print("=" * 72)

    print("\n[1] Legacy mode (all Phase 0 features off) — should be IDENTICAL")
    legacy = dict(size_aware_airtime=False, model_fragmentation=False, model_collisions=False)
    a = run(200, SHORT, **legacy)
    b = run(200, LONG, **legacy)
    print(f"    short : {a}")
    print(f"    long  : {b}")
    legacy_same = behaviour(a) == behaviour(b)
    print(f"    -> behaviour identical: {legacy_same}  (expected True)")

    print("\n[2] Phase 0 mode (size-aware) — should DIFFER")
    a = run(200, SHORT)
    b = run(200, LONG)
    print(f"    short : {a}")
    print(f"    long  : {b}")
    phase0_differs = behaviour(a) != behaviour(b)
    print(f"    -> behaviour differs: {phase0_differs}  (expected True)")

    print("\n[3] Fragment counts for one SOS")
    for label, txt in (("short", SHORT), ("long", LONG)):
        sim = build(10)
        node = sim.nodes[1000]
        sos = node.create_sos(0.0, text=txt)
        nbytes = sim.message_size_bytes(sos)
        nfrags = sim.fragment_count(nbytes)
        air = sim.airtime_sec(nbytes, nfrags)
        print(f"    {label:>5}: {nbytes:>4} B -> {nfrags:>2} frames, {air*1000:.2f} ms airtime")

    print("\n[4] Delivery across seeds, 100 nodes (reachable within TTL=8)")
    for label, txt in (("short", SHORT), ("long", LONG)):
        delivered = sum(run(100, txt, seed=s)["delivered"] for s in range(20))
        print(f"    {label:>5}: delivered in {delivered}/20 seeds")

    print("\n" + "=" * 72)
    ok = legacy_same and phase0_differs
    print("RESULT:", "PASS — Phase 0 is working" if ok else "FAIL — size still has no effect")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
