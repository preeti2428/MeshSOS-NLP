"""
Turn Google Form responses into a code-mixed Hindi evaluation set.

No public disaster corpus contains Hindi. Every `original` field in the
Figure Eight corpus is Latin-script Creole or French (see
docs/CORPUS_CHOICE.md), so the code-mixed claim cannot be tested on
existing data and the messages are elicited from volunteers instead.

Each volunteer answers ten scenarios described in plain English. The
descriptions deliberately avoid the words the encoder matches on, so the
volunteer supplies their own vocabulary and spelling. Each scenario
carries the labels it was designed to produce; those are written out as a
STARTING POINT only, and an annotator corrects them against what the
volunteer actually wrote. A message that does not match its scenario is
kept with corrected labels, not discarded.

Input :  data/hindi/form_responses.csv   (Google Forms CSV export)
Output:  data/hindi/hindi_messages.csv     id, scenario, message
         data/hindi/hindi_annotator_A.csv  blank sheet for correction
         data/hindi/hindi_annotator_B.csv  blank sheet for correction
         data/hindi/hindi_seed_labels.csv  scenario labels, pre-correction

Run:  python nlp/build_hindi_set.py
"""

from __future__ import annotations

import csv
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIN = ROOT / "data" / "hindi"
RESPONSES = HIN / "form_responses.csv"

SEED = 20261027
MIN_CHARS = 4

LABEL_COLUMNS = ["category", "casualty_count", "mobility",
                 "hazard_flags", "severity", "notes"]

# Scenario -> the labels it is designed to produce. Order matches the ten
# questions in data/hindi/GOOGLE_FORM.md.
SEED_LABELS = {
    "S1":  dict(category="TRAPPED",         casualty_count="3", mobility="UNRESPONSIVE", hazard_flags="STRUCTURAL",     severity="SERIOUS", notes=""),
    "S2":  dict(category="MEDICAL",         casualty_count="1", mobility="AMBULATORY",   hazard_flags="",               severity="SERIOUS", notes=""),
    "S3":  dict(category="FLOOD",           casualty_count="4", mobility="TRAPPED",      hazard_flags="WATER",          severity="UNKNOWN", notes=""),
    "S4":  dict(category="FIRE",            casualty_count="2", mobility="AMBULATORY",   hazard_flags="FIRE",           severity="UNKNOWN", notes=""),
    "S5":  dict(category="SUPPLIES_NEEDED", casualty_count="6", mobility="AMBULATORY",   hazard_flags="",               severity="UNKNOWN", notes=""),
    "S6":  dict(category="COLLAPSE",        casualty_count="7", mobility="TRAPPED",      hazard_flags="STRUCTURAL",     severity="UNKNOWN", notes=""),
    "S7":  dict(category="EVACUATION",      casualty_count="7", mobility="AMBULATORY",   hazard_flags="",               severity="UNKNOWN", notes=""),
    "S8":  dict(category="OTHER",           casualty_count="1", mobility="AMBULATORY",   hazard_flags="GAS,ELECTRICAL", severity="UNKNOWN", notes=""),
    "S9":  dict(category="SKIP",            casualty_count="",  mobility="",             hazard_flags="",               severity="",        notes="INFO_REQUEST"),
    "S10": dict(category="MEDICAL",         casualty_count="1", mobility="AMBULATORY",   hazard_flags="",               severity="SERIOUS", notes=""),
}
SCENARIOS = list(SEED_LABELS)

# Volunteers were told not to include personal details; strip anything that
# slipped through rather than trusting the instruction.
PHONE = re.compile(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def scrub(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = PHONE.sub("[phone]", text)
    text = EMAIL.sub("[email]", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    if not RESPONSES.exists():
        sys.exit(
            f"missing {RESPONSES.relative_to(ROOT)}\n"
            "Export the Google Form responses as CSV and save them there.\n"
            "See data/hindi/GOOGLE_FORM.md."
        )

    with open(RESPONSES, encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        sys.exit("the responses file has no answers in it yet")

    header, answers = rows[0], rows[1:]

    # Google Forms writes a Timestamp column first, then one column per
    # question in order. Anything beyond the ten questions is ignored.
    start = 1 if header and "timestamp" in header[0].strip().lower() else 0
    n_q = min(len(SCENARIOS), len(header) - start)
    if n_q < len(SCENARIOS):
        print(f"  warning: found {n_q} question columns, expected {len(SCENARIOS)}")

    records: list[dict] = []
    seen: set[str] = set()
    dropped = Counter()

    for r_i, row in enumerate(answers):
        for q_i in range(n_q):
            col = start + q_i
            raw = row[col] if col < len(row) else ""
            text = scrub(raw)
            if len(text) < MIN_CHARS:
                dropped["blank or too short"] += 1
                continue
            key = text.lower()
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            records.append({
                "scenario": SCENARIOS[q_i],
                "message": text,
                "respondent": r_i + 1,
            })

    if not records:
        sys.exit("no usable messages found")

    random.Random(SEED).shuffle(records)
    for i, rec in enumerate(records, start=1):
        rec["id"] = f"h{i:03d}"

    HIN.mkdir(parents=True, exist_ok=True)

    with open(HIN / "hindi_messages.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "scenario", "respondent", "message"])
        for r in records:
            w.writerow([r["id"], r["scenario"], r["respondent"], r["message"]])

    # Correction sheets: the scenario is hidden so the annotator reads only
    # what the volunteer actually wrote.
    for who in ("A", "B"):
        with open(HIN / f"hindi_annotator_{who}.csv", "w",
                  encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "message"] + LABEL_COLUMNS)
            for r in records:
                w.writerow([r["id"], r["message"]] + [""] * len(LABEL_COLUMNS))

    with open(HIN / "hindi_seed_labels.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "scenario"] + LABEL_COLUMNS)
        for r in records:
            lab = SEED_LABELS[r["scenario"]]
            w.writerow([r["id"], r["scenario"]] + [lab[c] for c in LABEL_COLUMNS])

    print("=" * 66)
    print("HINDI ELICITATION SET")
    print("=" * 66)
    print(f"  respondents          {len(answers)}")
    print(f"  usable messages      {len(records)}")
    for reason, n in dropped.most_common():
        print(f"  dropped {n:>4}         {reason}")
    print("\n  per scenario:")
    per = Counter(r["scenario"] for r in records)
    for s in SCENARIOS:
        print(f"    {s:<4} {per.get(s, 0):>4}")
    lens = sorted(len(r["message"]) for r in records)
    print(f"\n  length: median {lens[len(lens) // 2]} chars, "
          f"p90 {lens[int(len(lens) * 0.9)]}, max {lens[-1]}")
    print("\n  next: two people correct hindi_annotator_A.csv and _B.csv")
    print("        against docs/ANNOTATION_GUIDE.md, then")
    print("        python nlp/score_annotations.py \\")
    print("            data/hindi/hindi_annotator_A.csv \\")
    print("            data/hindi/hindi_annotator_B.csv")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
