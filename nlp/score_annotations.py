"""
Score two annotators, then evaluate the encoders on real text.

    python nlp/score_annotations.py A.csv B.csv            # agreement + disagreements
    python nlp/score_annotations.py gold.csv --evaluate    # encoders on the gold set

Cohen's kappa is reported per field, not as one number, because the fields
fail differently: counts are usually easy to agree on and severity is not.
A field below 0.6 means the guide is unclear for that field - fix
docs/ANNOTATION_GUIDE.md and relabel it, rather than averaging the problem
away.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIELDS = ("category", "casualty_count", "mobility", "hazard_flags", "severity")


def read(path: str) -> Dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        if (r.get("notes") or "").strip().upper().startswith("SKIP"):
            continue
        out[r["id"]] = r
    return out


def norm(field: str, value: str) -> str:
    v = (value or "").strip().upper()
    if field == "hazard_flags":
        # Order must not count as disagreement.
        return ",".join(sorted(p.strip() for p in v.split(",") if p.strip()))
    if field == "casualty_count":
        return v or "0"
    return v


def kappa(a: List[str], b: List[str]) -> float:
    """Cohen's kappa. Returns 1.0 when both annotators used a single value."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def agreement(path_a: str, path_b: str) -> int:
    A, B = read(path_a), read(path_b)
    shared = sorted(set(A) & set(B), key=lambda x: int(x) if x.isdigit() else 0)
    if not shared:
        print("No rows in common. Are both files the same sample?")
        return 1

    only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
    print("=" * 64)
    print(f"AGREEMENT  —  {len(shared)} rows labelled by both")
    if only_a or only_b:
        print(f"  (skipped or missing: {only_a} only in A, {only_b} only in B)")
    print("=" * 64)
    print(f"\n{'field':<18} {'agree':>8} {'kappa':>8}   reading")
    print("-" * 64)

    disagreements = []
    for f in FIELDS:
        va = [norm(f, A[i].get(f, "")) for i in shared]
        vb = [norm(f, B[i].get(f, "")) for i in shared]
        k = kappa(va, vb)
        pct = 100 * sum(x == y for x, y in zip(va, vb)) / len(shared)
        verdict = ("strong" if k >= 0.8 else "acceptable" if k >= 0.6
                   else "TOO LOW - fix the guide")
        print(f"{f:<18} {pct:>7.1f}% {k:>8.2f}   {verdict}")
        for i, x, y in zip(shared, va, vb):
            if x != y:
                disagreements.append((i, f, x, y, A[i].get("tweet_text", "")[:90]))

    print("-" * 64)
    print("kappa >= 0.8 strong · 0.6-0.8 acceptable · below 0.6 means the guide"
          "\nis ambiguous for that field, not that someone was careless")

    out = Path(path_a).parent / "disagreements.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "field", "annotator_A", "annotator_B", "text", "resolved"])
        w.writerows([[*d, ""] for d in disagreements])
    print(f"\n{len(disagreements)} disagreements -> {out}")
    print("Work through them together, fill `resolved`, then build gold.csv and run")
    print(f"  python {Path(__file__).name} gold.csv --evaluate")
    return 0


def evaluate(gold_path: str) -> int:
    from nlp.envelope import Category, Hazard, Mobility, SemanticEnvelope, Severity
    from nlp.rules import RuleEncoder

    HZ = {h.name: int(h) for h in Hazard}
    rows = read(gold_path)
    if not rows:
        print("No labelled rows found.")
        return 1

    golds, texts = [], []
    for r in rows.values():
        try:
            flags = 0
            for p in norm("hazard_flags", r.get("hazard_flags", "")).split(","):
                if p:
                    flags |= HZ[p]
            golds.append(SemanticEnvelope(
                category=Category[norm("category", r["category"]) or "OTHER"],
                casualty_count=int(norm("casualty_count", r.get("casualty_count", "0"))),
                mobility=Mobility[norm("mobility", r.get("mobility", "")) or "UNKNOWN"],
                hazard_flags=flags,
                severity=Severity[norm("severity", r.get("severity", "")) or "UNKNOWN"],
            ))
            texts.append(r["tweet_text"])
        except (KeyError, ValueError) as e:
            print(f"  row {r.get('id')}: skipped ({e})")

    enc = RuleEncoder()
    preds = [enc.encode(t) for t in texts]
    n = len(texts)

    print("=" * 64)
    print(f"RULE ENCODER ON REAL TEXT  —  {n} human-labelled messages")
    print("=" * 64)
    for f in FIELDS:
        c = sum(getattr(p.envelope, f) == getattr(g, f) for p, g in zip(preds, golds))
        print(f"  {f:<18} {100*c/n:>6.1f}%")
    allf = sum(all(getattr(p.envelope, f) == getattr(g, f) for f in FIELDS)
               for p, g in zip(preds, golds))
    fb = sum(1 for p in preds if not p.should_compress)
    print(f"  {'ALL FIELDS':<18} {100*allf/n:>6.1f}%")
    print(f"  {'fell back':<18} {100*fb/n:>6.1f}%")
    print("\nThis is the number the paper should quote. Everything measured on")
    print("generated text was a rehearsal for it.")
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--evaluate" in sys.argv:
        if len(args) != 1:
            print(__doc__)
            return 1
        return evaluate(args[0])
    if len(args) != 2:
        print(__doc__)
        return 1
    return agreement(args[0], args[1])


if __name__ == "__main__":
    raise SystemExit(main())
