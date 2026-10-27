"""
Compressing the fallback path with a shared dictionary.

When the encoder is not confident, MeshSOS refuses to compress and sends
the original text inside a full SosMessage. That is the safe behaviour --
nothing is guessed -- but it is expensive: the message runs to 160-190
bytes and costs six or seven advertising frames where a compact SOS costs
one.

Section "Why not general-purpose compression" in the paper shows that
gzip, bzip2 and LZMA all ENLARGE messages this short, because their
headers cost more than they save. A shared dictionary is the exception:
both ends ship the same dictionary, so zlib can reference common
emergency phrasing without transmitting it.

This script measures how much that is worth, honestly:

  The dictionary is built from a TRAINING split of the corpus and
  evaluated on a HELD-OUT split. Building it from the same messages it
  is scored on would be straightforward cheating and would roughly
  halve the apparent compressed size.

  Sizes are reported as whole SOS messages and as advertising FRAMES,
  because frames are what the radio actually charges for. Bytes saved
  inside a frame that still has to be sent are worth nothing.

  The semantic envelope is shown alongside, not as a competitor but as
  the reminder that this path only exists for the messages the encoder
  declined to compress.

Run:  python nlp/fallback_compression.py
"""

from __future__ import annotations

import csv
import random
import re
import sys
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

csv.field_size_limit(10 ** 7)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "data" / "raw" / "disaster_messages.csv"

SEED = 20261027
TRAIN_FRACTION = 0.7
DICT_BYTES = 8192          # ships inside the APK; 8 KB is nothing there
ZLIB_LEVEL = 9

# Everything in a full SosMessage that is not the text, measured from the
# protocol definition: header 13, lat/lon/accuracy 12, category 1,
# battery 1, text length 1, pubkey 32, signature 64.
SOS_FIXED_BYTES = 13 + 12 + 1 + 1 + 1 + 32 + 64
ADV_USABLE_PAYLOAD_BYTES = 27
COMPACT_SOS_BYTES = 25


def frames(n_bytes: int) -> int:
    return max(1, -(-n_bytes // ADV_USABLE_PAYLOAD_BYTES))


def load_direct() -> list[str]:
    if not CORPUS.exists():
        sys.exit(
            f"missing {CORPUS.relative_to(ROOT)}\n"
            "Run nlp/build_annotation_set.py first, which downloads it."
        )
    out, seen = [], set()
    with open(CORPUS, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["genre"] != "direct":
                continue
            t = unicodedata.normalize("NFKC", (row["message"] or "")).strip()
            t = re.sub(r"\s+", " ", t)
            if len(t) < 15 or re.match(r"^\s*NOTES?\s*:", t, re.I):
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
    return out


def build_dictionary(train: list[str], budget: int = DICT_BYTES) -> bytes:
    """
    A zlib dictionary is just a byte string; the compressor can reference
    any substring of it. Later bytes are cheaper to reference than earlier
    ones, so the most valuable material goes at the END.

    We score candidate phrases by (frequency x length), which approximates
    the bytes each one saves, and lay them down worst-first.
    """
    counts: Counter[str] = Counter()
    for msg in train:
        low = msg.lower()
        words = low.split()
        for n in (1, 2, 3, 4):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                if 4 <= len(phrase) <= 40:
                    counts[phrase] += 1

    scored = [
        (cnt * len(phrase), phrase)
        for phrase, cnt in counts.items()
        if cnt >= 3
    ]
    scored.sort()  # ascending: least valuable first, most valuable last

    chunks: list[str] = []
    total = 0
    for _, phrase in scored:
        piece = phrase + " "
        b = len(piece.encode("utf-8"))
        if total + b > budget:
            continue
        chunks.append(piece)
        total += b
    return "".join(chunks).encode("utf-8")[-budget:]


def compress(text: str, dictionary: bytes | None) -> int:
    """Compressed size in bytes, excluding the zlib stream header, which a
    fixed-format protocol field does not need to carry."""
    if dictionary is None:
        c = zlib.compressobj(ZLIB_LEVEL, zlib.DEFLATED, -15)
    else:
        c = zlib.compressobj(ZLIB_LEVEL, zlib.DEFLATED, -15, zdict=dictionary)
    raw = c.compress(text.encode("utf-8")) + c.flush()
    return len(raw)


def verify_roundtrip(texts: list[str], dictionary: bytes) -> None:
    """A compression result nobody decompressed is not a result."""
    for t in texts:
        c = zlib.compressobj(ZLIB_LEVEL, zlib.DEFLATED, -15, zdict=dictionary)
        blob = c.compress(t.encode("utf-8")) + c.flush()
        d = zlib.decompressobj(-15, zdict=dictionary)
        back = (d.decompress(blob) + d.flush()).decode("utf-8")
        if back != t:
            raise AssertionError(f"round trip failed for: {t[:60]!r}")


def summarise(name: str, text_sizes: list[int]) -> dict:
    n = len(text_sizes)
    totals = [SOS_FIXED_BYTES + t for t in text_sizes]
    fr = [frames(t) for t in totals]
    return {
        "name": name,
        "text_mean": sum(text_sizes) / n,
        "text_max": max(text_sizes),
        "msg_mean": sum(totals) / n,
        "frames_mean": sum(fr) / n,
        "frames_max": max(fr),
        "one_frame_pct": 100.0 * sum(1 for f in fr if f == 1) / n,
    }


def main() -> int:
    msgs = load_direct()
    rng = random.Random(SEED)
    rng.shuffle(msgs)
    split = int(len(msgs) * TRAIN_FRACTION)
    train, test = msgs[:split], msgs[split:]

    print("=" * 74)
    print("COMPRESSING THE FALLBACK PATH")
    print("=" * 74)
    print(f"  corpus      {len(msgs)} real direct messages")
    print(f"  train       {len(train)} (dictionary built from these only)")
    print(f"  held-out    {len(test)} (every number below measured on these)")

    dictionary = build_dictionary(train)
    print(f"  dictionary  {len(dictionary)} bytes, ships inside the APK")

    verify_roundtrip(test[:500], dictionary)
    print("  round trip  verified on 500 held-out messages")

    rows = [
        summarise("raw UTF-8", [len(t.encode("utf-8")) for t in test]),
        summarise("zlib, no dictionary", [compress(t, None) for t in test]),
        summarise("zlib + shared dictionary", [compress(t, dictionary) for t in test]),
    ]

    print(f"\n  {'method':<26}{'text B':>9}{'msg B':>9}{'frames':>9}"
          f"{'max fr':>8}{'1 frame':>10}")
    print("  " + "-" * 71)
    for r in rows:
        print(f"  {r['name']:<26}{r['text_mean']:>9.1f}{r['msg_mean']:>9.1f}"
              f"{r['frames_mean']:>9.2f}{r['frames_max']:>8}{r['one_frame_pct']:>9.0f}%")
    print(f"  {'compact SOS (envelope)':<26}{3.0:>9.1f}{COMPACT_SOS_BYTES:>9.1f}"
          f"{1.00:>9.2f}{1:>8}{100:>9.0f}%")

    raw, dict_c = rows[0], rows[2]
    saved_frames = 100 * (1 - dict_c["frames_mean"] / raw["frames_mean"])
    print(f"\n  A shared dictionary cuts the text to "
          f"{100*dict_c['text_mean']/raw['text_mean']:.0f}% of raw and the "
          f"FRAME count by {saved_frames:.0f}%")
    print(f"  ({raw['frames_mean']:.2f} -> {dict_c['frames_mean']:.2f} frames "
          f"per fallback message).")

    print(f"\n  But the floor is the crypto, not the text. The fixed part of a")
    print(f"  full SosMessage is {SOS_FIXED_BYTES} B ({frames(SOS_FIXED_BYTES)} frames) before a single")
    print(f"  character of text, of which 96 B is the signature and public key.")
    print(f"  Even compressing the text to nothing leaves {frames(SOS_FIXED_BYTES)} frames against the")
    print(f"  compact path's 1. Dictionary compression is worth having on this")
    print(f"  path and it does not remove the reason the compact path exists.")

    # ------------------------------------------------------------------
    # Why the dictionary must be built on a different split.
    # ------------------------------------------------------------------
    from nlp.sample_messages import SAMPLES

    hand = [t for t, _ in SAMPLES]
    raw_hand = sum(len(t.encode("utf-8")) for t in hand) / len(hand)
    d_verbatim = (" ".join(hand)).encode("utf-8")[-DICT_BYTES:]

    variants = [
        ("no dictionary", None),
        ("dictionary from the corpus train split", dictionary),
        ("dictionary containing these messages", d_verbatim),
    ]
    print(f"\n  How sensitive this number is to contamination")
    print(f"  ({len(hand)} hand-written sample messages, raw mean {raw_hand:.1f} B)")
    print("  " + "-" * 71)
    for name, d in variants:
        sizes = [compress(t, d) for t in hand]
        print(f"  {name:<42}{sum(sizes)/len(sizes):>8.1f} B{max(sizes):>8} B max")
    print("\n  The last row is not a result. If the evaluation text is inside the")
    print("  dictionary, each message is a literal substring of it and zlib emits")
    print("  little more than a back-reference. Any shared-dictionary figure in")
    print("  the single digits should be read as a split that leaked. The row to")
    print("  quote is the middle one, and the corpus-wide held-out figure above.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
