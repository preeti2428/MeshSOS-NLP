"""
The Python and Kotlin encoders must stay identical.

The app ships its own copy of the encoder in Kotlin. Twice during
development a change landed in nlp/rules.py and not in
SemanticEncoder.kt, which does not fail any build: the phone simply
behaves like an older version, and the only symptom is field results
that disagree with every number in the paper.

This test parses the lexicons and confidence constants out of both
files and fails when they drift. It is deliberately a text comparison
rather than a behavioural one, because there is no JVM in this test
environment -- it cannot prove the two produce the same output, only
that they were given the same vocabulary and the same arithmetic.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY_SRC = ROOT / "nlp" / "rules.py"
KT_SRC = ROOT / "android" / "app" / "src" / "main" / "java" / "org" / "meshsos" / "nlp" / "SemanticEncoder.kt"

CATEGORIES = ("TRAPPED", "COLLAPSE", "MEDICAL", "FLOOD", "FIRE",
              "SUPPLIES_NEEDED", "EVACUATION", "RESCUE_NEEDED")


def strip_comments(src: str, kotlin: bool) -> str:
    """Comments may quote example text; that must not count as a term."""
    if kotlin:
        src = re.sub(r"//[^\n]*", "", src)
        return re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"#[^\n]*", "", src)


@pytest.fixture(scope="module")
def sources() -> tuple[str, str]:
    assert PY_SRC.exists() and KT_SRC.exists(), "encoder sources missing"
    return (strip_comments(PY_SRC.read_text(encoding="utf-8"), kotlin=False),
            strip_comments(KT_SRC.read_text(encoding="utf-8"), kotlin=True))


def py_terms(src: str, cat: str) -> set[str]:
    m = re.search(rf"Category\.{cat}: \((.*?)\n    \),", src, re.S)
    assert m, f"python lexicon for {cat} not found"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def kt_terms(src: str, cat: str) -> set[str]:
    m = re.search(rf"Category\.{cat} to listOf\((.*?)\),\s*\n\s*(?:Category\.|\))",
                  src, re.S)
    assert m, f"kotlin lexicon for {cat} not found"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


@pytest.mark.parametrize("category", CATEGORIES)
def test_category_lexicons_match(sources, category):
    py_src, kt_src = sources
    p, k = py_terms(py_src, category), kt_terms(kt_src, category)
    assert p == k, (
        f"{category} lexicon differs between nlp/rules.py and SemanticEncoder.kt\n"
        f"  only in python: {sorted(p - k)}\n"
        f"  only in kotlin: {sorted(k - p)}"
    )


def test_confidence_constants_match(sources):
    """The numbers in the confidence function, in both files."""
    py_src, kt_src = sources
    for label, needle in [
        ("category base", "0.50"),
        ("per supporting field", "0.10"),
        ("no category, per field", "0.12"),
        ("unrecognised text penalty", "0.6"),
        ("uncertainty penalty", "0.8"),
        ("very short penalty", "0.7"),
        ("compression threshold", "0.45"),
    ]:
        in_py = needle in py_src
        in_kt = needle in kt_src or f"{needle}f" in kt_src
        assert in_py and in_kt, (
            f"confidence constant for {label} ({needle}) "
            f"present in python={in_py}, kotlin={in_kt}"
        )


def test_shelter_is_a_supply_not_an_evacuation(sources):
    """
    Regression. The annotation guide defines SUPPLIES_NEEDED as "food,
    water, medicine, shelter needed", but "shelter" sat under the
    higher-priority EVACUATION, so "TENT, SHELTER, FOOD AND OTHER THINGS"
    was read as an evacuation.
    """
    py_src, kt_src = sources
    for name, src, terms in (("python", py_src, py_terms),
                             ("kotlin", kt_src, kt_terms)):
        assert "shelter" in terms(src, "SUPPLIES_NEEDED"), f"{name}: missing"
        assert "shelter" not in terms(src, "EVACUATION"), f"{name}: still in EVACUATION"
