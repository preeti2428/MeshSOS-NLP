"""
The cascade: rules first, the learned model only where the rules are unsure.

The two encoders fail in opposite places. On the stratified set the rules
reach 79.8% on wording their lexicon was built for and collapse to 8.3%
on synonyms they have never seen; the model manages 39.1% there, five
times better, while losing on familiar wording. Routing between them
needs no new training, because the signal that decides it -- the rules'
own confidence -- already exists and already decides whether to compress.

    confident rules  -> take the rules' answer
    unsure rules     -> ask the model

This file exists so the idea can be MEASURED ON REAL TEXT before anyone
ports it to Kotlin. The 67.7% quoted for the cascade was computed on
generated messages, where the model had seen the generator. Real
messages are the test that matters.

Run:  python nlp/cascade.py [dev|test|all]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nlp.envelope import Category, Mobility, SemanticEnvelope, Severity  # noqa: E402
from nlp.rules import RuleEncoder                                        # noqa: E402

MODELS = ROOT / "models" / "encoder_fp32.pkl"


class Cascade:
    """Rules, with the model as a second opinion where the rules are unsure."""

    def __init__(self, threshold: float | None = None):
        self.rules = RuleEncoder()
        self.threshold = (RuleEncoder.DEFAULT_THRESHOLD
                          if threshold is None else threshold)
        if not MODELS.exists():
            raise SystemExit("run nlp/train_model.py first")
        self.models = pickle.loads(MODELS.read_bytes())

    def model_envelope(self, text: str) -> SemanticEnvelope:
        flags = 0
        for b in range(6):
            m = self.models.get(f"hazard_bit_{b}")
            if m is not None and int(m.predict([text])[0]):
                flags |= (1 << b)
        return SemanticEnvelope(
            category=Category(int(self.models["category"].predict([text])[0])),
            casualty_count=int(self.models["casualty_count"].predict([text])[0]),
            mobility=Mobility(int(self.models["mobility"].predict([text])[0])),
            hazard_flags=flags,
            severity=Severity(int(self.models["severity"].predict([text])[0])),
        )

    def encode(self, text: str):
        """Returns (envelope, which_stage, rules_confidence)."""
        r = self.rules.encode(text)
        if r.confidence >= self.threshold:
            return r.envelope, "rules", r.confidence
        return self.model_envelope(text), "model", r.confidence


def main() -> int:
    split = next((a for a in sys.argv[1:] if a in ("dev", "test", "all")), "test")

    import importlib.util
    spec = importlib.util.spec_from_file_location("dg", ROOT / "nlp" / "derived_gold.py")
    dg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dg)

    rows = [r for r in dg.load()
            if r["gold_cat"] not in (None, "SKIP")
            and (split == "all" or r["split"] == split)]
    non_sos = [r for r in dg.load()
               if r["gold_cat"] == "SKIP" and (split == "all" or r["split"] == split)]

    casc = Cascade()
    rules = casc.rules

    print("=" * 70)
    print("CASCADE vs RULES ALONE, ON REAL HELD-OUT MESSAGES")
    print("=" * 70)
    print(f"  split {split}: {len(rows)} messages with a specific human aid label")

    r_hit = c_hit = routed = 0
    r_hit_unsure = c_hit_unsure = 0
    for r in rows:
        res = rules.encode(r["text"])
        rules_pred = Category(res.envelope.category).name
        r_hit += rules_pred == r["gold_cat"]

        env, stage, _ = casc.encode(r["text"])
        casc_pred = Category(env.category).name
        c_hit += casc_pred == r["gold_cat"]

        if stage == "model":
            routed += 1
            r_hit_unsure += rules_pred == r["gold_cat"]
            c_hit_unsure += casc_pred == r["gold_cat"]

    n = len(rows)
    print(f"\n  {'':<28}{'correct':>10}")
    print(f"  {'rules alone':<28}{100*r_hit/n:>9.1f}%")
    print(f"  {'cascade':<28}{100*c_hit/n:>9.1f}%")
    print(f"  {'difference':<28}{100*(c_hit-r_hit)/n:>+9.1f} points")

    print(f"\n  routed to the model: {routed} of {n} ({100*routed/n:.0f}%)")
    if routed:
        print(f"  on those messages only:")
        print(f"    {'rules would have got':<28}{100*r_hit_unsure/routed:>9.1f}%")
        print(f"    {'the model got':<28}{100*c_hit_unsure/routed:>9.1f}%")

    # The safety property: the model has no confidence signal of its own,
    # so anything routed to it is compressed. That is the risk.
    if non_sos:
        rules_fb = sum(1 for r in non_sos
                       if not rules.encode(r["text"]).should_compress)
        print(f"\n  non-emergencies ({len(non_sos)}):")
        print(f"    rules fall back on {100*rules_fb/len(non_sos):.1f}%")
        print(f"    a cascade compresses ALL of them, because the model has no")
        print(f"    confidence of its own and the rules already said 'unsure'.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
