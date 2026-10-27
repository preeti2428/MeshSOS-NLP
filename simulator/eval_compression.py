"""
Phase 5 — what semantic compression does to mesh delivery.

Three conditions, same topology, same seeds, same loss:

  A  baseline    full SosMessage carrying the original text
  B  compressed  CompactSosMessage, 25 bytes, always
  C  realistic   compressed when the encoder is confident, full text when
                 it is not - the configuration you would actually ship

C is the honest one. B is the ceiling. Reporting B alone would overstate
the result, because it assumes the encoder is never unsure.

Run:  python simulator/eval_compression.py [seeds]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.rules import RuleEncoder
from nlp.sample_messages import SAMPLES
from protocol.mesh_protocol import SosCategory
from simulator.network_simulator import NetworkSimulator
from simulator.node import SimNode

SHARED_KEY = b"meshsos-gateway-key"
NODE_COUNTS = (50, 100, 200)
TTL = 8


def build(n: int, seed: int) -> NetworkSimulator:
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


def run_one(n: int, seed: int, condition: str, text: str, encoder: RuleEncoder) -> Dict:
    sim = build(n, seed)
    sim.run_until(5.0)

    result = encoder.encode(text)
    used_compact = False

    if condition == "A":
        sim.trigger_sos(1000, text=text, category=SosCategory.OTHER, ttl=TTL)
    elif condition == "B":
        sim.trigger_compact_sos(1000, result.envelope, ttl=TTL, shared_key=SHARED_KEY)
        used_compact = True
    elif condition == "C":
        if result.should_compress:
            sim.trigger_compact_sos(1000, result.envelope, ttl=TTL, shared_key=SHARED_KEY)
            used_compact = True
        else:
            sim.trigger_sos(1000, text=text, category=SosCategory.OTHER, ttl=TTL)
    else:
        raise ValueError(condition)

    sim.run_until(20.0)
    m = sim.metrics
    return {
        "delivered": m.total_sos_delivered_to_gateway,
        "tx": m.total_transmissions,
        "frames": m.total_fragments_tx,
        "airtime": m.total_airtime_sec,
        "compact": used_compact,
    }


def main() -> int:
    """
    Usage: python simulator/eval_compression.py [trials] [node_count ...]

    `trials` pairs (seed, message) round-robin rather than running every
    message against every seed, which keeps the run tractable while still
    varying topology and text together.
    """
    args = sys.argv[1:]
    trials = int(args[0]) if args else 60
    wanted = [int(a) for a in args[1:]] or list(NODE_COUNTS)

    encoder = RuleEncoder()
    texts = [t for t, _ in SAMPLES]

    out_path = Path(__file__).resolve().parent / "compression_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    print("=" * 78)
    print("SEMANTIC COMPRESSION vs MESH DELIVERY")
    print(f"{trials} trials per cell (seed x message round-robin), "
          f"TTL={TTL}, 3% per-frame loss")
    print("=" * 78)

    for n in wanted:
        print(f"\nN = {n} nodes")
        print(f"  {'cond':<4} {'delivery':>10} {'frames/SOS':>11} {'airtime(s)':>11} {'compact':>9}")
        print(f"  {'-'*4} {'-'*10} {'-'*11} {'-'*11} {'-'*9}")
        cell = {}

        for cond in ("A", "B", "C"):
            delivered = 0
            frames = 0.0
            airtime = 0.0
            compact_used = 0

            for i in range(trials):
                seed = i
                text = texts[i % len(texts)]
                r = run_one(n, seed, cond, text, encoder)
                delivered += r["delivered"]
                frames += r["frames"]
                airtime += r["airtime"]
                compact_used += int(r["compact"])

            cell[cond] = {
                "rate": delivered / trials,
                "frames": frames / trials,
                "airtime": airtime / trials,
                "compact_pct": 100.0 * compact_used / trials,
                "trials": trials,
            }
            print(f"  {cond:<4} {cell[cond]['rate']*100:>9.1f}% {cell[cond]['frames']:>11.0f} "
                  f"{cell[cond]['airtime']:>11.3f} {cell[cond]['compact_pct']:>8.0f}%")

        a, b, c = cell["A"], cell["B"], cell["C"]
        if a["rate"] > 0:
            cell["delivery_gain_B"] = 100 * (b["rate"] / a["rate"] - 1)
            cell["delivery_gain_C"] = 100 * (c["rate"] / a["rate"] - 1)
            print(f"  -> delivery  B vs A: {cell['delivery_gain_B']:+.1f}%   "
                  f"C vs A: {cell['delivery_gain_C']:+.1f}%")
        cell["airtime_saved_C"] = 100 * (1 - c["airtime"] / a["airtime"])
        cell["frames_saved_C"] = 100 * (1 - c["frames"] / a["frames"])
        print(f"  -> airtime saved C vs A: {cell['airtime_saved_C']:.1f}%   "
              f"frames saved: {cell['frames_saved_C']:.1f}%")

        results[str(n)] = cell
        out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print("A = full text   B = always compressed   C = compressed when confident")
    print("Quote C as the headline. B is the ceiling, not the shipping configuration.")
    print(f"saved -> {out_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
