"""
Score the encoders on the stratified test set.

Prints per-field accuracy per stratum, so the result says where each
encoder breaks rather than giving one number that hides it.

The unintelligible stratum is scored differently: there is nothing to get
right, so what matters is whether the encoder refuses to compress.

Run:  python nlp/eval_encoders.py [per_stratum] [--model]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.rules import RuleEncoder
from nlp.testset import STRATA, generate

FIELDS = ("category", "casualty_count", "mobility", "hazard_flags", "severity")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    per = int(args[0]) if args else 2000
    want_model = "--model" in sys.argv

    samples = generate(per)
    rules = RuleEncoder()

    print("=" * 86)
    print(f"ENCODER EVALUATION  —  {len(samples):,} generated messages, "
          f"{per:,} per stratum")
    print("Generated text with exact labels. A coverage diagnostic, not a benchmark score.")
    print("=" * 86)

    t0 = time.perf_counter()
    preds = [rules.encode(s.text) for s in samples]
    elapsed_ms = (time.perf_counter() - t0) / len(samples) * 1000

    report = {}
    print(f"\n{'stratum':<16} {'categ':>7} {'count':>7} {'mobil':>7} {'hazard':>7} "
          f"{'sever':>7} {'ALL':>7} {'fallback':>9}")
    print("-" * 86)

    for st in STRATA:
        idx = [i for i, s in enumerate(samples) if s.stratum == st]
        n = len(idx)
        fallback = sum(1 for i in idx if not preds[i].should_compress)

        if st == "unintelligible":
            # Nothing to recover; the only right answer is to refuse.
            print(f"{st:<16} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} "
                  f"{100*fallback/n:>8.1f}%")
            report[st] = {"n": n, "correct_refusals_pct": round(100 * fallback / n, 1)}
            continue

        row = {}
        for f in FIELDS:
            c = sum(1 for i in idx
                    if getattr(preds[i].envelope, f) == getattr(samples[i].gold, f))
            row[f] = 100 * c / n
        all_c = sum(1 for i in idx
                    if all(getattr(preds[i].envelope, f) == getattr(samples[i].gold, f)
                           for f in FIELDS))
        row["all_fields"] = 100 * all_c / n
        row["fallback"] = 100 * fallback / n

        print(f"{st:<16} {row['category']:>6.1f}% {row['casualty_count']:>6.1f}% "
              f"{row['mobility']:>6.1f}% {row['hazard_flags']:>6.1f}% "
              f"{row['severity']:>6.1f}% {row['all_fields']:>6.1f}% "
              f"{row['fallback']:>8.1f}%")
        report[st] = {"n": n, **{k: round(v, 1) for k, v in row.items()}}

    print("-" * 86)
    print(f"rule encoder: {elapsed_ms:.3f} ms per message, 0 KB model")

    print("\nReading the table:")
    print("  known_vocab    the ceiling - wording the lexicon was built for")
    print("  synonyms       the out-of-vocabulary gap, isolated. The number to improve.")
    print("  spelling       how far the lexicon survives real typing")
    print("  unintelligible fallback % is the score: refusing is correct here")

    out = Path(__file__).resolve().parent.parent / "models" / "stratified_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"per_stratum": per, "total": len(samples),
         "rule_ms_per_message": round(elapsed_ms, 4), "strata": report,
         "note": "Generated text with labels true by construction. Diagnostic of "
                 "lexicon coverage, not a benchmark result."},
        indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
