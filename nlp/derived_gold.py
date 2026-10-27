"""
Evaluate the encoder against gold labels DERIVED from the corpus's own
human annotations.

The honest problem with every encoder number in this project is that the
gold labels were written by us. The honest fix is two independent
annotators, and that is still the plan. But the Figure Eight corpus
already carries 36 binary labels per message, applied by that project's
annotators years before this one existed. Those are human judgements, made
by people with no stake in this encoder.

This script defines a deterministic, published mapping from those labels
to our envelope fields, and evaluates the encoder on all 10,402 direct
messages at once.

WHAT THIS IS AND IS NOT
-----------------------
It IS: a large evaluation on real messages whose labels no one on this
project chose, with the mapping stated in full so a reader can disagree
with it line by line.

It is NOT a substitute for annotation. Three things keep it honest:

  Coverage. Only `category` and four of six hazard bits can be derived.
  Figure Eight has no label that counts people, none for mobility and
  none for injury severity, so casualty_count, mobility and severity are
  NOT evaluated here at all. They still need annotators.

  Granularity. Their scheme is coarser than ours. `search_and_rescue`
  does not distinguish someone pinned under a slab from someone
  reported missing; we map it to TRAPPED and accept the error.

  The mapping is ours. The labels are human, but the translation is a
  design choice, and a self-serving translation would flatter the
  encoder. It was written once from the Figure Eight label definitions,
  before running the encoder against it, and is not tuned afterwards.
  It needs validating against ~50 human-checked messages, which is 45
  minutes of one person's time rather than six hours of two people's.

Run:  python nlp/derived_gold.py
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

csv.field_size_limit(10 ** 7)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"

from nlp.envelope import Category, Hazard          # noqa: E402
from nlp.rules import RuleEncoder                  # noqa: E402


# --------------------------------------------------------------------------
# The mapping. Each rule cites the Figure Eight labels it reads.
# Ordered by our own urgency priority, first match wins -- the same rule
# docs/ANNOTATION_GUIDE.md gives human annotators.
# --------------------------------------------------------------------------
CATEGORY_RULES = [
    ("COLLAPSE",        ("buildings",)),
    ("MEDICAL",         ("medical_help", "hospitals")),
    ("FIRE",            ("fire",)),
    ("FLOOD",           ("floods",)),
    ("EVACUATION",      ("refugees",)),
    ("SUPPLIES_NEEDED", ("food", "water", "shelter", "clothing")),
]

# Figure Eight's `request` and `aid_related` are far broader than "this is
# an emergency report". Reading the messages they cover, they include "I
# need information about the earthquake", "are they predicting another
# one?" and complaints about how distribution was organised -- all of
# which our own guide calls SKIP. An earlier version of this mapping sent
# them to RESCUE_NEEDED and so scored the encoder against a gold label a
# human annotator would not have given.
#
# There is no label in their scheme that separates an SOS from an enquiry,
# so these messages are EXCLUDED rather than guessed. Excluding them
# shrinks the evaluation and keeps it honest; RESCUE_NEEDED therefore
# cannot be evaluated here at all.
AMBIGUOUS_ONLY = ("request", "aid_related", "other_aid", "direct_report",
                  "weather_related", "earthquake", "infrastructure_related",
                  "search_and_rescue", "medical_products")

# Two labels were dropped from the rules above after reading the messages
# they cover. Both were judged against docs/ANNOTATION_GUIDE.md, not
# against anything the encoder predicted.
#
#   medical_products covers donation and hygiene lists -- "PLEASE DONATE
#   SOAP SHAMPOO TOOTHBRUSHES", "paper products, pet food, diapers" --
#   which our guide does not call MEDICAL.
#
#   search_and_rescue does not mean "cannot get out". It is applied to
#   "my house is destroyed, I left with the clothes on my body. I'm out
#   on the street", where the writer explicitly got out. Our TRAPPED
#   requires that people cannot, so TRAPPED has no counterpart here and
#   is not evaluated.

# Only four of our six hazard bits have any counterpart in their scheme.
HAZARD_RULES = {
    "WATER":      ("floods", "storm"),
    "FIRE":       ("fire",),
    "STRUCTURAL": ("buildings",),
    "ELECTRICAL": ("electricity",),
}
UNDERIVABLE_HAZARDS = ("GAS", "CHEMICAL")
UNDERIVABLE_FIELDS = ("casualty_count", "mobility", "severity")


def parse_labels(blob: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in blob.split(";"):
        if "-" in part:
            key, _, val = part.rpartition("-")
            try:
                out[key.strip()] = int(val)
            except ValueError:
                pass
    return out


def derive(fe: dict[str, int]) -> tuple[str | None, set[str]]:
    """
    Figure Eight's labels -> (our category, our hazard bits).

    Returns None for the category when their labels cannot distinguish an
    emergency report from an enquiry about one. Those messages are dropped
    from the category evaluation rather than assigned a guess.
    """
    hazards = {h for h, keys in HAZARD_RULES.items()
               if any(fe.get(k, 0) == 1 for k in keys)}

    if fe.get("related", 0) != 1:
        return "SKIP", hazards                  # cleanly not an emergency

    for name, keys in CATEGORY_RULES:
        if any(fe.get(k, 0) == 1 for k in keys):
            return name, hazards                # a specific aid category

    return None, hazards                        # related, but only vaguely


def load() -> list[dict]:
    mp, cp = RAW / "disaster_messages.csv", RAW / "disaster_categories.csv"
    if not (mp.exists() and cp.exists()):
        sys.exit("missing data/raw/*.csv -- run nlp/build_annotation_set.py first")

    labels = {r["id"]: parse_labels(r["categories"])
              for r in csv.DictReader(open(cp, encoding="utf-8"))}

    rows, seen = [], set()
    for r in csv.DictReader(open(mp, encoding="utf-8")):
        if r["genre"] != "direct":
            continue
        t = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", r["message"] or "")).strip()
        if len(t) < 15 or re.match(r"^\s*NOTES?\s*:", t, re.I):
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        fe = labels.get(r["id"])
        if not fe:
            continue
        cat, haz = derive(fe)
        rows.append({"text": t, "gold_cat": cat, "gold_haz": haz})
    return rows


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    rows = load()
    enc = RuleEncoder()

    print("=" * 76)
    print("ENCODER vs GOLD LABELS DERIVED FROM FIGURE EIGHT'S HUMAN ANNOTATIONS")
    print("=" * 76)
    print(f"  {len(rows)} real messages. Labels are human; the mapping to our")
    print(f"  fields is ours and is printed below so it can be argued with.")
    print(f"\n  not evaluated here (no counterpart in their scheme):")
    print(f"    fields  {', '.join(UNDERIVABLE_FIELDS)}, and the TRAPPED category")
    print(f"    hazards {', '.join(UNDERIVABLE_HAZARDS)}")

    emergencies     = [r for r in rows if r["gold_cat"] not in (None, "SKIP")]
    non_emergencies = [r for r in rows if r["gold_cat"] == "SKIP"]
    ambiguous       = [r for r in rows if r["gold_cat"] is None]
    print(f"\n  {len(ambiguous)} messages excluded: their labels cannot tell an")
    print(f"  emergency report from an enquiry about one. RESCUE_NEEDED has no")
    print(f"  counterpart in their scheme and is not evaluated.")

    # ---- category, on messages their annotators called real ----
    per_cat: dict[str, list[int]] = {}
    correct = compressed = 0
    for r in emergencies:
        res = enc.encode(r["text"])
        pred = Category(res.envelope.category).name
        hit = int(pred == r["gold_cat"])
        correct += hit
        compressed += int(res.should_compress)
        per_cat.setdefault(r["gold_cat"], []).append(hit)

    n = len(emergencies)
    print(f"\n  CATEGORY, on the {n} messages Figure Eight marked related")
    print(f"  {'gold category':<20}{'n':>7}{'encoder correct':>18}")
    print("  " + "-" * 45)
    for name, _ in CATEGORY_RULES:
        hits = per_cat.get(name)
        if not hits:
            continue
        print(f"  {name:<20}{len(hits):>7}{100*sum(hits)/len(hits):>17.1f}%")
    print("  " + "-" * 45)
    print(f"  {'overall':<20}{n:>7}{100*correct/n:>17.1f}%")
    print(f"  compressed (did not fall back): {100*compressed/n:.1f}%")

    # ---- the safety property, on messages their annotators called unrelated ----
    if non_emergencies:
        fell_back = sum(1 for r in non_emergencies
                        if not enc.encode(r["text"]).should_compress)
        m = len(non_emergencies)
        print(f"\n  NON-EMERGENCIES, {m} messages Figure Eight marked unrelated")
        print(f"  encoder fell back to full text: {100*fell_back/m:.1f}%")
        print(f"  (it should refuse these; compressing them invents a report)")

    # ---- hazard bits ----
    print(f"\n  HAZARD BITS, the four that can be derived")
    print(f"  {'bit':<14}{'gold n':>8}{'precision':>11}{'recall':>9}{'F1':>8}")
    print("  " + "-" * 50)
    for bit in HAZARD_RULES:
        flag = int(Hazard[bit])
        tp = fp = fn = 0
        for r in rows:
            pred = bool(enc.encode(r["text"]).envelope.hazard_flags & flag)
            gold = bit in r["gold_haz"]
            tp += pred and gold
            fp += pred and not gold
            fn += gold and not pred
        p, rc, f1 = prf(tp, fp, fn)
        print(f"  {bit:<14}{tp+fn:>8}{p:>11.2f}{rc:>9.2f}{f1:>8.2f}")

    print(f"\n  THE MAPPING")
    for name, keys in CATEGORY_RULES:
        print(f"    {name:<18} <- {' or '.join(keys)}")
    print(f"    {'SKIP':<18} <- not related")
    print(f"    {'(excluded)':<18} <- related but only {', '.join(AMBIGUOUS_ONLY[:3])}, ...")
    for bit, keys in HAZARD_RULES.items():
        print(f"    hazard {bit:<11} <- {' or '.join(keys)}")

    print("\n  First match wins, in our own urgency order, which is the same")
    print("  rule docs/ANNOTATION_GUIDE.md gives human annotators.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
