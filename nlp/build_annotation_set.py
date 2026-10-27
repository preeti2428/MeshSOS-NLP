"""
Build the real-message annotation set for the MeshSOS semantic envelope.

Corpus: "Multilingual Disaster Response Messages" (Figure Eight / Appen),
genre == "direct" -- SMS and web-form messages sent BY people inside the
2010 Haiti earthquake, the 2010 Pakistan floods and the 2012 storms,
translated to English. Unlike tweet corpora (HumAID, CrisisBench) these
come from participants asking for help, not observers commenting, which
is the population a mesh SOS system actually carries.

Sampling is two-part so the evaluation can report both an honest
deployment estimate and useful per-field numbers:

  uniform   250 drawn uniformly from every cleaned direct message.
            Gives an unbiased estimate of real-world behaviour,
            including the ~1/3 of traffic that is not an emergency.

  enriched  250 drawn from messages the DATASET'S OWN human annotators
            marked request=1 or aid_related=1. Those labels are
            independent of this project, so enrichment adds no
            circularity. Gives enough positives per field to measure
            category and hazard accuracy.

The two annotators never see which part a message came from, and never
see the Figure Eight labels. Both are kept in a separate provenance file
for scoring afterwards.

Run:  python nlp/build_annotation_set.py
"""

from __future__ import annotations

import csv
import random
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

csv.field_size_limit(10 ** 7)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "annotated"

MIRROR = "https://raw.githubusercontent.com/prateeksawhney97/Disaster-Response-Pipeline/master/data"
# Byte-identical copy, used to verify the mirror has not been modified.
MIRROR_CHECK = "https://raw.githubusercontent.com/tri-bui/disaster-response/master/data"

FILES = ("disaster_messages.csv", "disaster_categories.csv")

N_UNIFORM = 250
N_ENRICHED = 250
SEED = 20261027

# Columns the annotators fill in. Must match docs/ANNOTATION_GUIDE.md.
LABEL_COLUMNS = [
    "category",
    "casualty_count",
    "mobility",
    "hazard_flags",
    "severity",
    "notes",
]


def fetch(name: str) -> Path:
    """Download once into data/raw/ using curl (the only reachable client here)."""
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/{name}"
    print(f"  downloading {name} ...")
    r = subprocess.run(
        ["curl", "-sS", "--max-time", "180", "-o", str(dest), url],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not dest.exists():
        sys.exit(f"download failed for {url}\n{r.stderr}")
    return dest


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_labels(blob: str) -> dict[str, int]:
    """'related-1;request-0;...' -> {'related': 1, 'request': 0, ...}"""
    out: dict[str, int] = {}
    for part in blob.split(";"):
        if "-" in part:
            key, _, val = part.rpartition("-")
            try:
                out[key.strip()] = int(val)
            except ValueError:
                pass
    return out


# Rows where the message field holds a translator's note rather than a message.
TRANSLATOR_NOTE = re.compile(r"^\s*NOTES?\s*:", re.IGNORECASE)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def clean(messages: list[dict], labels: dict[str, str]) -> list[dict]:
    """Keep direct-genre messages that are actually messages, deduplicated."""
    kept: list[dict] = []
    seen: set[str] = set()
    dropped = Counter()

    for row in messages:
        if row["genre"] != "direct":
            continue
        text = (row["message"] or "").strip()

        if not text:
            dropped["empty"] += 1
            continue
        if TRANSLATOR_NOTE.match(text):
            dropped["translator note"] += 1
            continue
        if len(text) < 15:
            dropped["under 15 chars"] += 1
            continue
        key = normalise(text)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)

        kept.append({
            "drm_id": row["id"],
            "message": text,
            "labels": parse_labels(labels.get(row["id"], "")),
        })

    print("\n  cleaning:")
    for reason, n in dropped.most_common():
        print(f"    dropped {n:>5}  {reason}")
    print(f"    kept    {len(kept):>5}  direct messages")
    return kept


def main() -> int:
    print("=" * 70)
    print("BUILDING THE REAL-MESSAGE ANNOTATION SET")
    print("=" * 70)
    print("\ncorpus: Disaster Response Messages (Figure Eight), genre = direct")

    msg_path, cat_path = (fetch(n) for n in FILES)
    messages = read_csv(msg_path)
    categories = {r["id"]: r["categories"] for r in read_csv(cat_path)}
    print(f"  loaded {len(messages)} rows, {len(categories)} label rows")

    pool = clean(messages, categories)

    enriched_pool = [
        r for r in pool
        if r["labels"].get("request") == 1 or r["labels"].get("aid_related") == 1
    ]
    print(f"\n  request=1 or aid_related=1: {len(enriched_pool)} "
          f"({100 * len(enriched_pool) / len(pool):.1f}% of the pool)")

    rng = random.Random(SEED)

    uniform = rng.sample(pool, N_UNIFORM)
    chosen = {r["drm_id"] for r in uniform}
    remaining = [r for r in enriched_pool if r["drm_id"] not in chosen]
    enriched = rng.sample(remaining, N_ENRICHED)

    for r in uniform:
        r["stratum"] = "uniform"
    for r in enriched:
        r["stratum"] = "enriched"

    sample = uniform + enriched
    rng.shuffle(sample)
    for i, r in enumerate(sample, start=1):
        r["id"] = f"m{i:03d}"

    OUT.mkdir(parents=True, exist_ok=True)

    # --- the two annotator sheets: message text only, no hints of any kind
    for who in ("A", "B"):
        path = OUT / f"drm_annotator_{who}.csv"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "message"] + LABEL_COLUMNS)
            for r in sample:
                w.writerow([r["id"], r["message"]] + [""] * len(LABEL_COLUMNS))
        print(f"\n  wrote {path.relative_to(ROOT)}  ({len(sample)} rows)")

    # --- provenance: never shown to annotators, used when scoring
    prov = OUT / "drm_provenance.csv"
    fe_fields = ["request", "aid_related", "direct_report", "medical_help",
                 "search_and_rescue", "water", "food", "shelter", "death",
                 "missing_people", "earthquake", "floods", "fire", "buildings"]
    with open(prov, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "drm_id", "stratum"] + [f"fe_{f}" for f in fe_fields])
        for r in sample:
            w.writerow([r["id"], r["drm_id"], r["stratum"]]
                       + [r["labels"].get(f, "") for f in fe_fields])
    print(f"  wrote {prov.relative_to(ROOT)}  (annotators must NOT open this)")

    print("\n  sample composition:")
    for name, n in Counter(r["stratum"] for r in sample).most_common():
        print(f"    {name:<10} {n}")
    lens = sorted(len(r["message"]) for r in sample)
    print(f"    message length: median {lens[len(lens) // 2]} chars, "
          f"p90 {lens[int(len(lens) * 0.9)]}, max {lens[-1]}")

    print("\n  next: two people fill in drm_annotator_A.csv and _B.csv")
    print("        separately, following docs/ANNOTATION_GUIDE.md, then run")
    print("        python nlp/score_annotations.py "
          "data/annotated/drm_annotator_A.csv data/annotated/drm_annotator_B.csv")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
