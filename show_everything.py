#!/usr/bin/env python3
"""
One command for a viva or a demo: runs the whole evidence chain live.

    python show_everything.py           every section, pausing between
    python show_everything.py --fast    skip the slow ones
    python show_everything.py 3         just section 3

Nothing here is pre-computed. Every number on screen is produced by the
code in this repository while the examiner watches, which is the point.
See docs/VIVA.md for what to say alongside each section.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# (title, what to say it proves, command, slow?)
SECTIONS = [
    ("The pipeline, one message end to end",
     "Free text becomes 3 bytes, then a 25-byte frame, then readable "
     "text again at the gateway.",
     [PY, "demo.py"], False),

    ("Training, live",
     "6,000 synthetic messages. The model wins on its own generated "
     "split and loses badly on hand-written text, which is why the "
     "phone ships rules and no model file.",
     [PY, "nlp/train_model.py"], False),

    ("Why the tweet corpus was rejected",
     "HumAID is 12.3% first-person appeals against this corpus's 41.6%. "
     "Tweets are written by observers; we carry messages from "
     "participants.",
     [PY, "nlp/compare_corpora.py"], False),

    ("Real messages, held-out half, human labels",
     "1,524 real messages from the 2010 Haiti earthquake, labelled by "
     "Figure Eight's annotators years before this project. The row that "
     "matters is what it got right among the messages it chose to "
     "compress.",
     [PY, "nlp/derived_gold.py", "test"], False),

    ("Code-mixed Hindi, 547 elicited messages",
     "Read the two groups: seven of our ten form questions accidentally "
     "showed volunteers words from our own lexicon. Only the clean "
     "three are a fair estimate.",
     [PY, "nlp/eval_hindi.py", "test"], False),

    ("Failure by type, 4,000 generated messages",
     "Where the encoder breaks: unseen synonyms. Note that it refuses "
     "100% of unintelligible input rather than guessing.",
     [PY, "nlp/eval_encoders.py", "800"], True),

    ("The frame budget",
     "Compressing the text cannot reach one frame, because 124 bytes of "
     "header and cryptography come before any text at all.",
     [PY, "nlp/fallback_compression.py"], True),

    ("Size-aware mesh simulation",
     "The original simulator charged every message the same airtime, so "
     "payload size could not affect anything. This fails if that is "
     "still true.",
     [PY, "simulator/test_phase0_size_aware.py"], True),

    ("The test suite",
     "37 tests, including one that fails the build if a field pushes "
     "the message past 27 bytes, and one that fails if the Python and "
     "Kotlin encoders drift apart.",
     [PY, "-m", "pytest", "tests/", "-q",
      "--ignore=tests/test_backend_pipeline.py"], False),
]


def run(i: int, title: str, says: str, cmd: list[str]) -> bool:
    bar = "=" * 72
    print(f"\n{bar}\n  {i}. {title.upper()}\n{bar}")
    print(f"  {says}\n")
    print(f"  $ {' '.join(c.replace(str(PY), 'python') for c in cmd)}\n")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"\n  [section {i} exited {r.returncode}]")
    return r.returncode == 0


def main() -> int:
    args = sys.argv[1:]
    fast = "--fast" in args
    picked = [int(a) for a in args if a.isdigit()]

    chosen = [
        (i, t, s, c) for i, (t, s, c, slow) in enumerate(SECTIONS, 1)
        if (not picked or i in picked) and not (fast and slow)
    ]

    print("\n  MeshSOS — everything, run live")
    print(f"  {len(chosen)} sections. Press Enter between them, Ctrl-C to stop.")

    failed = []
    for n, (i, t, s, c) in enumerate(chosen):
        if n:
            try:
                input("\n  [Enter for the next section] ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if not run(i, t, s, c):
            failed.append(i)

    print("\n" + "=" * 72)
    if failed:
        print(f"  sections that did not exit cleanly: {failed}")
    else:
        print("  Every section ran. Nothing above was pre-computed.")
    print("  What to say alongside each one: docs/VIVA.md")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
