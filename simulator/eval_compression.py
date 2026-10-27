"""
Phase 5 - what semantic compression does to mesh delivery.

Three conditions, run on the SAME topology, seed and message:

  A  baseline    full SosMessage carrying the original text
  B  compressed  CompactSosMessage, 25 bytes, always
  C  realistic   compressed when the encoder is confident, full text when
                 it is not - the configuration you would actually ship

C is the honest one. B is the ceiling. Reporting B alone would overstate
the result, because it assumes the encoder is never unsure.

STATISTICS
----------
Every trial is one Bernoulli outcome: the SOS reached a gateway or it did
not. An earlier version of this script reported only the mean of 60 such
trials, which cannot distinguish a real 4% improvement from noise. Two
things are now reported instead.

  Wilson score intervals on each arm's delivery rate. Wilson rather than
  the normal approximation because delivery rates here run near 0.8 and
  samples are moderate, where the normal interval is visibly wrong and
  can even extend past 1.

  Exact McNemar tests between arms. The three conditions see the same
  seed and the same text in every trial, so the design is PAIRED and the
  only informative trials are those where two arms disagree. McNemar uses
  exactly those, and is both correct and more powerful here than a
  two-sample proportion test, which would throw the pairing away.

Both are computed from the standard library, so the script has no new
dependency.

Run:  python simulator/eval_compression.py [trials] [node_count ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.rules import RuleEncoder
from nlp.sample_messages import SAMPLES
from protocol.mesh_protocol import SosCategory
from simulator.network_simulator import NetworkSimulator
from simulator.node import SimNode

SHARED_KEY = b"meshsos-gateway-key"
NODE_COUNTS = (50, 100, 200)
TTL = 8
Z = 1.959963985  # 95%


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = Z) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_tail(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 0.5), computed exactly."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)


def mcnemar(x: Sequence[int], y: Sequence[int]) -> Dict:
    """
    Exact McNemar test on paired binary outcomes.

    b = trials where x succeeded and y did not
    c = trials where y succeeded and x did not
    Concordant trials carry no information about which arm is better.
    """
    b = sum(1 for i, j in zip(x, y) if i and not j)
    c = sum(1 for i, j in zip(x, y) if j and not i)
    n_disc = b + c
    p = min(1.0, 2.0 * binom_tail(min(b, c), n_disc)) if n_disc else 1.0
    return {"b": b, "c": c, "discordant": n_disc, "p_value": p}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    return "ns "


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

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
        "delivered": 1 if m.total_sos_delivered_to_gateway else 0,
        "frames": m.total_fragments_tx,
        "airtime": m.total_airtime_sec,
        "compact": used_compact,
    }


def main() -> int:
    args = sys.argv[1:]
    trials = int(args[0]) if args else 500
    wanted = [int(a) for a in args[1:]] or list(NODE_COUNTS)

    encoder = RuleEncoder()
    texts = [t for t, _ in SAMPLES]

    out_path = Path(__file__).resolve().parent / "compression_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    print("=" * 78)
    print("SEMANTIC COMPRESSION vs MESH DELIVERY")
    print(f"{trials} paired trials per cell, TTL={TTL}, 3% per-frame loss")
    print("95% Wilson intervals; arms compared with an exact McNemar test")
    print("=" * 78)

    for n in wanted:
        print(f"\nN = {n} nodes")

        # Every condition sees the same seed and the same text in a trial,
        # which is what makes the McNemar comparison below legitimate.
        outcomes: Dict[str, List[int]] = {"A": [], "B": [], "C": []}
        frames: Dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}
        airtime: Dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}
        compact_used = 0

        for i in range(trials):
            seed = i
            text = texts[i % len(texts)]
            for cond in ("A", "B", "C"):
                r = run_one(n, seed, cond, text, encoder)
                outcomes[cond].append(r["delivered"])
                frames[cond] += r["frames"]
                airtime[cond] += r["airtime"]
                if cond == "C":
                    compact_used += int(r["compact"])

        cell: Dict = {"trials": trials}
        print(f"  {'cond':<5}{'delivery':>9}{'95% CI':>18}{'frames/SOS':>12}{'airtime(s)':>12}")
        print(f"  {'-'*5}{'-'*9}{'-'*18}{'-'*12}{'-'*12}")
        for cond in ("A", "B", "C"):
            k = sum(outcomes[cond])
            lo, hi = wilson(k, trials)
            cell[cond] = {
                "rate": k / trials,
                "ci_low": lo,
                "ci_high": hi,
                "delivered": k,
                "frames": frames[cond] / trials,
                "airtime": airtime[cond] / trials,
                "outcomes": outcomes[cond],
            }
            print(f"  {cond:<5}{100*k/trials:>8.1f}%"
                  f"{f'[{100*lo:.1f}, {100*hi:.1f}]':>18}"
                  f"{frames[cond]/trials:>12.0f}{airtime[cond]/trials:>12.3f}")

        cell["C"]["compact_pct"] = 100.0 * compact_used / trials
        print(f"  C compressed {cell['C']['compact_pct']:.0f}% of messages; "
              f"the rest fell back to full text")

        print(f"\n  paired comparisons (exact McNemar)")
        print(f"  {'pair':<8}{'wins':>14}{'p':>12}")
        for lo_c, hi_c in (("A", "B"), ("A", "C"), ("B", "C")):
            t = mcnemar(outcomes[lo_c], outcomes[hi_c])
            cell[f"mcnemar_{lo_c}{hi_c}"] = t
            print(f"  {lo_c} vs {hi_c:<4}{f'{t[chr(98)]} / {t[chr(99)]}':>14}"
                  f"{t['p_value']:>11.4f} {stars(t['p_value'])}")

        a, b, c = cell["A"], cell["B"], cell["C"]
        if a["rate"] > 0:
            cell["delivery_gain_B"] = 100 * (b["rate"] / a["rate"] - 1)
            cell["delivery_gain_C"] = 100 * (c["rate"] / a["rate"] - 1)
            print(f"\n  relative delivery  B vs A {cell['delivery_gain_B']:+.1f}%   "
                  f"C vs A {cell['delivery_gain_C']:+.1f}%")
        cell["airtime_saved_C"] = 100 * (1 - c["airtime"] / a["airtime"])
        cell["frames_saved_C"] = 100 * (1 - c["frames"] / a["frames"])
        print(f"  airtime saved C vs A {cell['airtime_saved_C']:.1f}%   "
              f"frames saved {cell['frames_saved_C']:.1f}%")

        results[str(n)] = cell
        out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print("A = full text   B = always compressed   C = compressed when confident")
    print("Quote C as the headline. B is the ceiling, not the shipping")
    print("configuration. 'wins' is b / c: trials where the left arm")
    print("delivered and the right did not, and the reverse. Only those")
    print("trials carry information; the rest agree.")
    print(f"saved -> {out_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
