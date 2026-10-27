"""
Phase 0.4 — separate TTL-limited failure from size-limited failure.

The 200-node row of the original EVALUATION.md reported SOS delivered = 0.
That is a topology/TTL result, not a payload result: the nearest gateway is
about 17 hops away and TTL is 8, so the message dies before it arrives, no
matter how small it is.

This sweep makes the two causes visible side by side. Read it as:
  - TTL too low  -> both short and long fail, and raising TTL fixes both
  - TTL adequate -> short beats long, and that gap is what compression buys

Run:  python simulator/ttl_vs_size_experiment.py [seeds]
"""

from __future__ import annotations
import math
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.network_simulator import NetworkSimulator
from simulator.node import SimNode

SHORT = "Scale 200 S"
LONG = "X" * 240


def build(n: int, seed: int):
    area = math.sqrt(n) * 80.0
    sim = NetworkSimulator(width=area, height=area, packet_loss_rate=0.03, seed=seed)
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


def hops_to_gateway(sim, src_id: int):
    """Minimum hops from src to the nearest gateway over the radio-range graph."""
    nodes = sim.nodes
    ids = list(nodes)
    adj = {nid: [] for nid in ids}
    for a in range(len(ids)):
        na = nodes[ids[a]]
        for b in range(a + 1, len(ids)):
            nb = nodes[ids[b]]
            if na.distance_to(nb) <= na.radio_range:
                adj[ids[a]].append(ids[b])
                adj[ids[b]].append(ids[a])
    dist = {src_id: 0}
    q = deque([src_id])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    reach = [dist[g] for g, nd in nodes.items() if nd.is_gateway and g in dist]
    return min(reach) if reach else None


def run(n: int, text: str, ttl: int, seed: int) -> int:
    sim = build(n, seed)
    sim.run_until(5.0)
    sim.trigger_sos(1000, text=text, ttl=ttl)
    sim.run_until(20.0)
    return sim.metrics.total_sos_delivered_to_gateway


def main() -> int:
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    node_counts = (50, 100, 200)
    ttls = (8, 12, 16, 20)

    print("=" * 74)
    print(f"TTL vs MESSAGE SIZE  —  {seeds} seeds per cell, 3% per-frame loss")
    print("=" * 74)

    for n in node_counts:
        needed = [hops_to_gateway(build(n, s), 1000) for s in range(seeds)]
        clean = [h for h in needed if h is not None]
        med = sorted(clean)[len(clean) // 2] if clean else None
        print(f"\nN = {n}   (hops to nearest gateway: median {med}, "
              f"range {min(clean)}-{max(clean)})" if clean else f"\nN = {n}")
        print(f"  {'TTL':>4} {'short':>8} {'long':>8} {'gap':>8}   verdict")
        print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*8}   {'-'*28}")

        for ttl in ttls:
            s_ok = sum(run(n, SHORT, ttl, seed) for seed in range(seeds))
            l_ok = sum(run(n, LONG, ttl, seed) for seed in range(seeds))
            gap = s_ok - l_ok

            if s_ok == 0 and l_ok == 0:
                verdict = "TTL-limited (size irrelevant)"
            elif gap > 0:
                verdict = "size matters -> compression helps"
            elif s_ok == seeds and l_ok == seeds:
                verdict = "saturated, no headroom"
            else:
                verdict = "no size effect at this TTL"

            print(f"  {ttl:>4} {s_ok:>4}/{seeds:<3} {l_ok:>4}/{seeds:<3} {gap:>+8}   {verdict}")

    print("\n" + "=" * 74)
    print("Report BOTH a TTL-limited row and an adequate-TTL row in the paper.")
    print("Claiming compression fixes a TTL-limited case would be wrong.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
