"""
Rule-based semantic encoder.

Turns a free-text emergency report into a 3-byte SemanticEnvelope, with a
confidence score the caller uses to decide whether to trust the compression
or fall back to sending the original text.

This is deliberately the simplest encoder that could work. It is the
baseline every learned model must beat: it needs no training data, no model
file, no accelerator, and it runs in microseconds on any handset. Report its
numbers honestly before claiming a neural model is necessary.

Handles English and romanised Hindi / Hindi-English code-mixed input, which
is what the deployment context actually produces.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from nlp.envelope import (
    CASUALTY_MANY,
    Category,
    Hazard,
    Mobility,
    SemanticEnvelope,
    Severity,
)

# --- lexicons --------------------------------------------------------------
# Each term maps to a field value. Romanised Hindi sits beside English
# because code-mixed input is the norm, not an edge case.

CATEGORY_TERMS: Dict[Category, Tuple[str, ...]] = {
    Category.TRAPPED: (
        "trapped", "stuck", "pinned", "buried", "under rubble", "can't get out",
        "cannot get out", "fansa", "phansa", "fanse", "phanse", "dabba", "dabe",
        "malbe ke niche", "malbe ke neeche", "atka", "atke",
    ),
    Category.COLLAPSE: (
        "collapse", "collapsed", "building fell", "wall fell", "roof fell",
        "rubble", "debris", "girr gaya", "gir gayi", "malba", "malbe",
        "chhat gir", "deewar gir", "imarat gir",
    ),
    Category.MEDICAL: (
        "injured", "injury", "bleeding", "unconscious", "heart attack", "fracture",
        "broken leg", "broken arm", "medical", "doctor", "ambulance", "stretcher",
        "ghayal", "chot", "khoon", "behosh", "dawai", "bimar", "saans",
    ),
    Category.FLOOD: (
        "flood", "flooded", "water rising", "drowning", "submerged", "waterlogged",
        "baadh", "badh", "paani bhar", "pani bhar", "doob", "dub rahe",
    ),
    Category.FIRE: (
        "fire", "burning", "smoke", "flames", "blaze",
        "aag", "jal raha", "jal rahi", "dhuan", "dhuaan",
    ),
    Category.SUPPLIES_NEEDED: (
        "need water", "need food", "no water", "no food", "supplies", "medicine",
        "blanket", "khana", "paani chahiye", "pani chahiye", "bhookh", "dawa chahiye",
    ),
    Category.EVACUATION: (
        "evacuate", "evacuation", "need to leave", "get out", "shelter",
        "nikalna", "nikalo", "bahar nikal", "shelter chahiye",
    ),
    Category.RESCUE_NEEDED: (
        "rescue", "help us", "save us", "send help", "need help",
        "bachao", "madad", "help chahiye", "koi hai",
    ),
}

# Category priority when several match. More specific and more
# life-threatening wins: being trapped outranks a general call for help.
CATEGORY_PRIORITY: Tuple[Category, ...] = (
    Category.TRAPPED,
    Category.COLLAPSE,
    Category.MEDICAL,
    Category.FIRE,
    Category.FLOOD,
    Category.EVACUATION,
    Category.SUPPLIES_NEEDED,
    Category.RESCUE_NEEDED,
    Category.OTHER,
)

MOBILITY_TERMS: Dict[Mobility, Tuple[str, ...]] = {
    Mobility.UNRESPONSIVE: (
        "unconscious", "not responding", "unresponsive", "no pulse", "not breathing",
        "behosh", "jawab nahi", "saans nahi", "hosh nahi",
    ),
    Mobility.TRAPPED: (
        "trapped", "stuck", "pinned", "buried", "can't move", "cannot move",
        "can't get out", "cannot get out", "immobile",
        "fansa", "phansa", "fanse", "phanse", "hil nahi", "nikal nahi",
        "dabba", "dabe", "atka",
    ),
    Mobility.AMBULATORY: (
        "can walk", "walking", "we are walking", "able to move", "mobile",
        "chal sakte", "chal raha", "chal rahe", "theek hai", "safe hai",
    ),
}

HAZARD_TERMS: Dict[Hazard, Tuple[str, ...]] = {
    Hazard.FIRE: ("fire", "burning", "flames", "smoke", "aag", "jal raha", "dhuan", "dhuaan"),
    Hazard.WATER: ("flood", "flooded", "drowning", "submerged", "rising water",
                   "water rising", "waterlogged", "under water",
                   "baadh", "badh", "paani bhar", "pani bhar", "doob", "dub rahe"),
    Hazard.GAS: ("gas", "gas leak", "lpg", "cylinder", "fumes", "gas leak ho", "gais"),
    Hazard.STRUCTURAL: ("collapse", "collapsed", "rubble", "debris", "unstable",
                        "cracks", "wall", "roof", "ceiling",
                        "malba", "malbe", "chhat", "deewar", "darar"),
    Hazard.ELECTRICAL: ("live wire", "electric", "electrocut", "short circuit", "current",
                        "bijli", "taar", "karent"),
    Hazard.CHEMICAL: ("chemical", "toxic", "acid", "ammonia", "fertilizer", "pesticide",
                      "zeharila", "rasayan"),
}

SEVERITY_SERIOUS = (
    "severe", "serious", "critical", "heavy bleeding", "unconscious", "not breathing",
    "dying", "life threatening", "amputat", "deep wound", "head injury",
    "gambhir", "bahut chot", "khoon beh", "khoon bah", "marne", "behosh", "saans nahi",
)
SEVERITY_MINOR = (
    "minor", "small cut", "scratch", "bruise", "mild", "slight", "stable",
    "halka", "halki", "mamuli", "thodi chot", "theek hun", "theek hoon",
)
SEVERITY_NONE = (
    "no injury", "no injuries", "nobody hurt", "no one hurt", "all safe", "unhurt",
    "koi chot nahi", "sab theek", "sab safe", "koi ghayal nahi",
)

# Number words, English and romanised Hindi.
NUMBER_WORDS: Dict[str, int] = {
    "one": 1, "a": 1, "an": 1, "single": 1, "ek": 1,
    "two": 2, "couple": 2, "both": 2, "do": 2, "don": 2,
    "three": 3, "teen": 3, "tin": 3,
    "four": 4, "char": 4, "chaar": 4,
    "five": 5, "paanch": 5, "panch": 5,
    "six": 6, "chhe": 6, "che": 6, "chah": 6,
    "seven": 7, "saat": 7, "sat": 7,
    "eight": 8, "aath": 8, "ath": 8,
    "nine": 9, "nau": 9,
    "ten": 10, "das": 10, "dus": 10,
    "dozen": 12, "many": CASUALTY_MANY, "several": 3, "lots": CASUALTY_MANY,
    "kai": 3, "bahut": CASUALTY_MANY, "sab": CASUALTY_MANY,
}

# Words that mean the count refers to people.
PERSON_WORDS = (
    "people", "person", "persons", "men", "women", "children", "child", "kids",
    "kid", "family", "families", "adults", "boys", "girls", "students", "workers",
    "injured", "trapped", "casualt", "victims", "survivors",
    "log", "logo", "admi", "aadmi", "bacche", "baccha", "bache", "aurat", "aurte",
    "parivar", "ladke", "ladki", "jan",
)

# Markers that the writer is unsure. These reduce confidence.
UNCERTAINTY_TERMS = (
    "maybe", "not sure", "i think", "possibly", "unclear", "unknown", "approximately",
    "around", "about", "?", "shayad", "pata nahi", "lagta hai", "ho sakta",
)

# Negation markers that invert a nearby match.
NEGATION_TERMS = ("no ", "not ", "without ", "nobody ", "none ", "koi nahi")

# Romanised Hindi negates after the word it negates ("ghayal nahi"), so these
# are looked for in a short window following the matched term.
NEGATION_TERMS_TRAILING = ("nahi", "nhi", "mat ", "na hai")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _match_terms(text: str, terms: Tuple[str, ...]) -> List[str]:
    return [t for t in terms if t in text]


def _negated_near(text: str, term: str, before: int = 18, after: int = 14) -> bool:
    """
    True if every occurrence of `term` is negated.

    English negates before the term ("no injuries"); romanised Hindi negates
    after it ("ghayal nahi"). Both windows are checked, and the trailing one
    is kept short so a "nahi" later in the sentence does not reach back and
    cancel an unrelated word.
    """
    idx = text.find(term)
    if idx == -1:
        return False
    while idx != -1:
        prefix = text[max(0, idx - before):idx]
        suffix = text[idx + len(term): idx + len(term) + after]
        negated = (any(neg in prefix for neg in NEGATION_TERMS)
                   or any(neg in suffix for neg in NEGATION_TERMS_TRAILING))
        if not negated:
            return False  # at least one un-negated occurrence
        idx = text.find(term, idx + 1)
    return True


@dataclass
class EncodeResult:
    envelope: SemanticEnvelope
    confidence: float          # 0.0-1.0, before quantisation into the envelope
    matched_terms: List[str]
    unrepresented_ratio: float  # how much of the text the encoder did not use
    reasons: List[str]          # why confidence landed where it did

    @property
    def should_compress(self) -> bool:
        """Convenience for the default threshold."""
        return self.confidence >= RuleEncoder.DEFAULT_THRESHOLD


class RuleEncoder:
    """Keyword and pattern encoder. No training, no model file."""

    DEFAULT_THRESHOLD = 0.45

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    # --- field extraction ------------------------------------------------

    def _category(self, text: str) -> Tuple[Category, List[str]]:
        hits: Dict[Category, List[str]] = {}
        for cat, terms in CATEGORY_TERMS.items():
            matched = [t for t in _match_terms(text, terms) if not _negated_near(text, t)]
            if matched:
                hits[cat] = matched
        for cat in CATEGORY_PRIORITY:
            if cat in hits:
                return cat, hits[cat]
        return Category.OTHER, []

    def _casualty_count(self, text: str) -> Tuple[int, List[str]]:
        """
        Find a number that refers to people. Looks for a digit or number word
        within a short window before a person word, so "4 people trapped"
        counts and "4 floors" does not.
        """
        best: Optional[int] = None
        matched: List[str] = []

        for m in re.finditer(r"\b(\d{1,3})\b", text):
            tail = text[m.end():m.end() + 28]
            if any(p in tail for p in PERSON_WORDS):
                n = int(m.group(1))
                best = max(best or 0, n)
                matched.append(m.group(0))

        if best is None:
            for word, value in NUMBER_WORDS.items():
                for m in re.finditer(rf"\b{re.escape(word)}\b", text):
                    tail = text[m.end():m.end() + 28]
                    if any(p in tail for p in PERSON_WORDS):
                        best = max(best or 0, value)
                        matched.append(word)
                        break

        if best is None:
            return 0, []
        return min(best, CASUALTY_MANY), matched

    def _mobility(self, text: str) -> Tuple[Mobility, List[str]]:
        for mob in (Mobility.UNRESPONSIVE, Mobility.TRAPPED, Mobility.AMBULATORY):
            matched = [t for t in _match_terms(text, MOBILITY_TERMS[mob])
                       if not _negated_near(text, t)]
            if matched:
                return mob, matched
        return Mobility.UNKNOWN, []

    def _hazards(self, text: str) -> Tuple[int, List[str]]:
        flags = 0
        matched: List[str] = []
        for hz, terms in HAZARD_TERMS.items():
            hit = [t for t in _match_terms(text, terms) if not _negated_near(text, t)]
            if hit:
                flags |= int(hz)
                matched.extend(hit)
        return flags, matched

    def _severity(self, text: str) -> Tuple[Severity, List[str]]:
        for term in SEVERITY_NONE:
            if term in text:
                return Severity.NONE, [term]
        hit = [t for t in _match_terms(text, SEVERITY_SERIOUS) if not _negated_near(text, t)]
        if hit:
            return Severity.SERIOUS, hit
        hit = [t for t in _match_terms(text, SEVERITY_MINOR) if not _negated_near(text, t)]
        if hit:
            return Severity.MINOR, hit
        return Severity.UNKNOWN, []

    # --- confidence -------------------------------------------------------

    def _confidence(
        self,
        text: str,
        category: Category,
        casualty: int,
        mobility: Mobility,
        hazards: int,
        severity: Severity,
        matched: List[str],
    ) -> Tuple[float, float, List[str]]:
        reasons: List[str] = []

        extras = sum([
            casualty > 0,
            mobility != Mobility.UNKNOWN,
            hazards != 0,
            severity != Severity.UNKNOWN,
        ])

        # Confidence asks "did I understand what this message says", not
        # "did I fill all five slots". The earlier version scored
        # filled/5, which treats a complete short report as four fifths
        # missing. On 10,398 real disaster messages that refused all but
        # 0.2% of them: 92.8% of real messages let the encoder recover one
        # field or none, because people write one thing at a time --
        # "we need water", "my house fell". Those are not partial reports.
        #
        # So the category carries the score: naming the emergency is the
        # part a rescuer acts on, and corroborating detail raises
        # confidence from there rather than being required for it.
        if category != Category.OTHER:
            score = 0.50 + 0.10 * extras
            reasons.append(f"category recognised, {extras} supporting field(s)")
        else:
            # No category matched. Detail alone can still describe
            # something real ("3 people, one unconscious"), but without
            # knowing what happened it stays below the threshold unless
            # nearly everything else is present.
            score = 0.12 * extras
            reasons.append(f"no category matched, {extras} other field(s)")

        # How much of the text did the lexicons actually account for?
        tokens = [t for t in re.findall(r"[a-z0-9']+", text) if len(t) > 2]
        covered: Set[str] = set()
        for term in matched:
            covered.update(re.findall(r"[a-z0-9']+", term))
        unrepresented = 0.0
        if tokens:
            unrepresented = 1.0 - (len({t for t in tokens if t in covered}) / len(set(tokens)))

        # Text carrying a lot the encoder could not read is a reason to doubt
        # the compression, not to trust the fields it did find.
        if unrepresented > 0.85 and len(tokens) > 8:
            score *= 0.6
            reasons.append("most of the message was not recognised")

        if any(u in text for u in UNCERTAINTY_TERMS):
            score *= 0.8
            reasons.append("writer signalled uncertainty")

        if len(tokens) <= 2:
            score *= 0.7
            reasons.append("message too short to judge")

        return max(0.0, min(1.0, score)), unrepresented, reasons

    # --- public API -------------------------------------------------------

    def encode(self, text: str) -> EncodeResult:
        norm = _normalise(text or "")

        category, c_terms = self._category(norm)
        casualty, n_terms = self._casualty_count(norm)
        mobility, m_terms = self._mobility(norm)
        hazards, h_terms = self._hazards(norm)
        severity, s_terms = self._severity(norm)

        matched = c_terms + n_terms + m_terms + h_terms + s_terms
        conf, unrep, reasons = self._confidence(
            norm, category, casualty, mobility, hazards, severity, matched
        )

        envelope = SemanticEnvelope(
            category=category,
            casualty_count=casualty,
            mobility=mobility,
            hazard_flags=hazards,
            severity=severity,
            confidence=round(conf * 7),
        )
        return EncodeResult(
            envelope=envelope,
            confidence=conf,
            matched_terms=matched,
            unrepresented_ratio=unrep,
            reasons=reasons,
        )
