"""
The encoder on elicited code-mixed Hindi.

547 messages from 56 volunteers, each answering ten scenarios described
in plain English that deliberately avoid every word the encoder matches
on, so the vocabulary and spelling are the volunteers' own. See
data/hindi/README.md.

LABELS. Each message starts with the labels its scenario was designed to
produce. Those are a starting point, not ground truth: a volunteer who
answered the "neighbour's house collapsed" scenario with "I'm no so
close with my neighbours anyway" has a seed label of COLLAPSE and a real
label of SKIP. Numbers here are therefore a FLOOR -- every mislabelled
joke counts against the encoder. The annotator pass over
hindi_annotator_A/B.csv replaces them.

CONTAMINATION. The Google Form asked its questions in Hinglish, and
seven of the ten reintroduced words that are in the encoder's own
lexicon -- S4 showed "aag" and "dhuan", S6 showed "girr gaya", S2
"saans", S10 "khoon". Volunteers echoed them. Matching a word the
volunteer was just shown proves nothing about the encoder, so the
scenarios are reported in two groups and only the clean group is a fair
estimate. This was a flaw in the form, not in the volunteers' answers;
data/hindi/GOOGLE_FORM.md now carries questions that avoid every
lexicon term, for a future round.

SPLIT. By RESPONDENT, not by message. One person's spelling habits run
through all ten of their answers, so splitting by message would leak
"fase" from dev into test inside the same writer's idiolect. Lexicon
work is done against dev; test is reported.

Run:  python nlp/eval_hindi.py [dev|test|all]
"""

from __future__ import annotations

import collections
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HIN = ROOT / "data" / "hindi"

from nlp.envelope import Category          # noqa: E402
from nlp.rules import RuleEncoder          # noqa: E402

# Scenarios whose question text contained no term from any lexicon.
# Checked against nlp.rules; see the module docstring.
CLEAN_SCENARIOS = {"FLOOD", "SUPPLIES_NEEDED", "EVACUATION"}


def load(split: str = "all") -> list[dict]:
    msgs = list(csv.DictReader(open(HIN / "hindi_messages.csv", encoding="utf-8")))
    seed = {r["id"]: r for r in
            csv.DictReader(open(HIN / "hindi_seed_labels.csv", encoding="utf-8"))}
    out = []
    for m in msgs:
        half = ("dev" if hashlib.sha1(m["respondent"].encode()).digest()[0] % 2
                else "test")
        if split != "all" and half != split:
            continue
        out.append({"id": m["id"], "text": m["message"],
                    "gold": seed[m["id"]]["category"],
                    "respondent": m["respondent"], "split": half})
    return out


def main() -> int:
    split = next((a for a in sys.argv[1:] if a in ("dev", "test", "all")), "all")
    rows = load(split)
    enc = RuleEncoder()

    people = len({r["respondent"] for r in rows})
    print("=" * 68)
    print("ENCODER ON ELICITED CODE-MIXED HINDI")
    print("=" * 68)
    print(f"  split {split}: {len(rows)} messages from {people} volunteers")
    print(f"  labels are scenario seeds, not annotator-corrected -- a floor")

    per = collections.defaultdict(lambda: [0, 0, 0])
    tot = [0, 0, 0]
    for r in rows:
        if r["gold"] == "SKIP":
            continue
        res = enc.encode(r["text"])
        pred = Category(res.envelope.category).name
        hit = int(pred == r["gold"])
        per[r["gold"]][0] += 1
        per[r["gold"]][1] += hit
        per[r["gold"]][2] += int(res.should_compress)
        tot[0] += 1; tot[1] += hit; tot[2] += int(res.should_compress)

    print(f"\n  {'scenario label':<18}{'n':>5}{'correct':>10}{'compressed':>12}")
    print("  " + "-" * 45)
    for g in sorted(per, key=lambda k: -per[k][0]):
        n, c, z = per[g]
        print(f"  {g:<18}{n:>5}{100*c/n:>9.1f}%{100*z/n:>11.1f}%")
    print("  " + "-" * 45)
    print(f"  {'overall':<18}{tot[0]:>5}{100*tot[1]/tot[0]:>9.1f}%{100*tot[2]/tot[0]:>11.1f}%")

    clean = [(g, v) for g, v in per.items() if g in CLEAN_SCENARIOS]
    dirty = [(g, v) for g, v in per.items() if g not in CLEAN_SCENARIOS]
    if clean and dirty:
        cn = sum(v[0] for _, v in clean); cc = sum(v[1] for _, v in clean)
        dn = sum(v[0] for _, v in dirty); dc = sum(v[1] for _, v in dirty)
        print(f"\n  scenarios whose wording leaked no lexicon term "
              f"(FLOOD, SUPPLIES, EVACUATION)")
        print(f"    {cn} messages, {100*cc/cn:.1f}% correct   <-- the fair estimate")
        print(f"  scenarios whose wording showed the volunteer a lexicon term")
        print(f"    {dn} messages, {100*dc/dn:.1f}% correct")
        print(f"  The gap is the form's fault, not the encoder's merit.")

    sel_n = sel_hit = 0
    for r in rows:
        if r["gold"] == "SKIP":
            continue
        res = enc.encode(r["text"])
        if res.should_compress:
            sel_n += 1
            sel_hit += Category(res.envelope.category).name == r["gold"]
    if sel_n:
        print(f"  OF WHAT IT COMPRESSED, correct: {100*sel_hit/sel_n:.1f}% ({sel_hit}/{sel_n})")

    skips = [r for r in rows if r["gold"] == "SKIP"]
    if skips:
        fb = sum(1 for r in skips if not enc.encode(r["text"]).should_compress)
        print(f"\n  S9 'I am safe, is help coming': {len(skips)} messages, "
              f"fell back {100*fb/len(skips):.1f}%")
        print(f"  (these state no emergency; compressing one invents a report)")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
