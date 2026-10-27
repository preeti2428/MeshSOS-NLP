"""
Train a learned encoder and compare it, honestly, against the rules.

Training data is template-generated (nlp/synth.py) because the real corpora
- HumAID, CrisisBench - were not reachable from the build environment. The
model therefore learns a narrower world than real reports contain.

Two test sets, and the second is the one that counts:

  held-out synthetic   same generator, unseen samples. Flatters the model.
  hand-written         nlp/sample_messages.py, never templated. This is the
                       out-of-distribution test, and the number to quote.

Run:  python nlp/train_model.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.envelope import Category, Mobility, SemanticEnvelope, Severity
from nlp.rules import RuleEncoder
from nlp.sample_messages import SAMPLES
from nlp.synth import generate

MODELS = Path(__file__).resolve().parent.parent / "models"
FIELDS = ("category", "casualty_count", "mobility", "severity")
HAZARD_BITS = 6


def build_model() -> Pipeline:
    # Character n-grams, so romanised-Hindi spelling variants that no
    # word list anticipates still land near each other.
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  max_features=4000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=4.0)),
    ])


def field_value(env: SemanticEnvelope, field: str) -> int:
    v = getattr(env, field)
    return int(v)


def main() -> int:
    train = generate(6000, seed=11)
    dev = generate(1200, seed=99)

    X_tr = [t for t, _ in train]
    X_dev = [t for t, _ in dev]
    X_hand = [t for t, _ in SAMPLES]

    models = {}
    print("training ...")
    t0 = time.perf_counter()
    for f in FIELDS:
        m = build_model()
        m.fit(X_tr, [field_value(e, f) for _, e in train])
        models[f] = m
    # One binary model per hazard bit - hazards co-occur.
    for b in range(HAZARD_BITS):
        m = build_model()
        y = [(e.hazard_flags >> b) & 1 for _, e in train]
        if len(set(y)) < 2:
            continue
        m.fit(X_tr, y)
        models[f"hazard_bit_{b}"] = m
    train_s = time.perf_counter() - t0
    print(f"trained {len(models)} heads on {len(X_tr)} samples in {train_s:.1f}s\n")

    def predict(texts):
        out = []
        preds = {f: models[f].predict(texts) for f in FIELDS}
        hz = {b: models[f"hazard_bit_{b}"].predict(texts)
              for b in range(HAZARD_BITS) if f"hazard_bit_{b}" in models}
        for i in range(len(texts)):
            flags = 0
            for b, p in hz.items():
                if p[i]:
                    flags |= (1 << b)
            out.append(SemanticEnvelope(
                category=Category(int(preds["category"][i])),
                casualty_count=int(preds["casualty_count"][i]),
                mobility=Mobility(int(preds["mobility"][i])),
                hazard_flags=flags,
                severity=Severity(int(preds["severity"][i])),
            ))
        return out

    rules = RuleEncoder()

    def score(name, texts, golds):
        model_env = predict(texts)
        rule_env = [rules.encode(t).envelope for t in texts]
        print(f"--- {name}  (n={len(texts)}) ---")
        print(f"  {'field':<16} {'model':>8} {'rules':>8}")
        rows = {}
        for f in FIELDS + ("hazard_flags",):
            mc = sum(getattr(m, f) == getattr(g, f) for m, g in zip(model_env, golds))
            rc = sum(getattr(r, f) == getattr(g, f) for r, g in zip(rule_env, golds))
            rows[f] = (mc / len(texts), rc / len(texts))
            print(f"  {f:<16} {100*mc/len(texts):>7.1f}% {100*rc/len(texts):>7.1f}%")
        m_all = sum(all(getattr(m, f) == getattr(g, f) for f in FIELDS + ("hazard_flags",))
                    for m, g in zip(model_env, golds))
        r_all = sum(all(getattr(r, f) == getattr(g, f) for f in FIELDS + ("hazard_flags",))
                    for r, g in zip(rule_env, golds))
        rows["all_fields"] = (m_all / len(texts), r_all / len(texts))
        print(f"  {'ALL FIELDS':<16} {100*m_all/len(texts):>7.1f}% {100*r_all/len(texts):>7.1f}%\n")
        return rows

    synth_rows = score("held-out synthetic (same generator - flatters the model)",
                       X_dev, [e for _, e in dev])
    hand_rows = score("hand-written, out of distribution - THE NUMBER TO QUOTE",
                      X_hand, [e for _, e in SAMPLES])

    # --- size and latency -------------------------------------------------
    MODELS.mkdir(exist_ok=True)
    raw = MODELS / "encoder_fp32.pkl"
    raw.write_bytes(pickle.dumps(models))

    # int8 quantisation of the coefficients, the part that dominates size.
    q = {}
    for name, pipe in models.items():
        clf = pipe.named_steps["clf"]
        c = clf.coef_
        scale = np.abs(c).max() / 127.0 or 1.0
        q[name] = {
            "coef_int8": np.round(c / scale).astype(np.int8),
            "scale": float(scale),
            "intercept": clf.intercept_.astype(np.float32),
            "classes": clf.classes_,
            "vocab": pipe.named_steps["tfidf"].vocabulary_,
            "idf": pipe.named_steps["tfidf"].idf_.astype(np.float32),
        }
    quant = MODELS / "encoder_int8.pkl"
    quant.write_bytes(pickle.dumps(q))

    t0 = time.perf_counter()
    for _ in range(20):
        predict(X_hand)
    model_ms = (time.perf_counter() - t0) / (20 * len(X_hand)) * 1000

    t0 = time.perf_counter()
    for _ in range(200):
        for t in X_hand:
            rules.encode(t)
    rules_ms = (time.perf_counter() - t0) / (200 * len(X_hand)) * 1000

    print("--- cost ---")
    print(f"  model fp32   {raw.stat().st_size/1024:>8.0f} KB   {model_ms:.3f} ms / message")
    print(f"  model int8   {quant.stat().st_size/1024:>8.0f} KB")
    print(f"  rules             0 KB   {rules_ms:.3f} ms / message")

    (MODELS / "training_report.json").write_text(json.dumps({
        "train_samples": len(X_tr),
        "train_seconds": round(train_s, 1),
        "synthetic_heldout": {k: [round(a, 4), round(b, 4)] for k, (a, b) in synth_rows.items()},
        "hand_written": {k: [round(a, 4), round(b, 4)] for k, (a, b) in hand_rows.items()},
        "model_fp32_kb": round(raw.stat().st_size / 1024, 1),
        "model_int8_kb": round(quant.stat().st_size / 1024, 1),
        "model_ms_per_message": round(model_ms, 4),
        "rules_ms_per_message": round(rules_ms, 4),
        "note": "values are [model, rules]. Training data is synthetic; the "
                "hand-written set is the out-of-distribution test.",
    }, indent=2, default=str))
    print(f"\nsaved -> {MODELS/'training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
