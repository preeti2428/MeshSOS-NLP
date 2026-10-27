"""
Stratified test set, thousands of messages with ground truth by construction.

A flat synthetic set of 10,000 messages tells you almost nothing: score it
once and you get a number whose meaning depends entirely on how the
generator happened to phrase things. Split by *how far the wording sits
from what the encoder knows* and the same 10,000 messages tell you where
the encoder breaks and why.

Five strata:

  S1 known_vocab    words the rule lexicon contains. The ceiling.
  S2 synonyms       same meaning, words the lexicon does NOT contain
                    ("jal" for "paani", "jalplavan" for "baadh"). This is
                    the out-of-vocabulary problem, isolated.
  S3 spelling       known words, misspelled the way people actually type
                    ("paani"/"pani"/"panee", "fansa"/"phansa"/"fnsa").
  S4 code_mixed     Hindi and English interleaved inside one sentence.
  S5 unintelligible messages carrying no recoverable emergency content.
                    Correct behaviour here is to FALL BACK, not to score.

Still generated text, not real human reports: the labels are exact because
the generator chose them, and real reports are messier than any generator.
Treat these numbers as a diagnostic of coverage, not as a benchmark score.
A real evaluation needs HumAID / CrisisBench with annotated fields.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from nlp.envelope import Category, Hazard, Mobility, SemanticEnvelope, Severity

STRATA = ("known_vocab", "synonyms", "spelling", "code_mixed", "unintelligible")


@dataclass(frozen=True)
class Sample:
    text: str
    gold: SemanticEnvelope
    stratum: str


# --- S1: wording the rule lexicon contains --------------------------------
KNOWN: Dict[Category, List[str]] = {
    Category.TRAPPED: ["we are trapped", "stuck under the rubble", "main fansa hun",
                       "malbe ke neeche dabe hain", "cannot get out"],
    Category.COLLAPSE: ["the building collapsed", "roof fell", "chhat gir gayi",
                        "deewar gir gayi", "rubble everywhere"],
    Category.MEDICAL: ["someone is injured", "bleeding badly", "ghayal hai",
                       "chot lagi hai", "need ambulance"],
    Category.FLOOD: ["flood here", "water rising", "paani bhar raha hai",
                     "baadh aa gayi", "street is flooded"],
    Category.FIRE: ["fire in the building", "smoke everywhere", "aag lag gayi",
                    "dhuaan bhar gaya", "flames on this floor"],
    Category.SUPPLIES_NEEDED: ["need water", "no food", "khana nahi hai",
                               "paani chahiye", "need medicine"],
    Category.EVACUATION: ["we must evacuate", "need to leave", "bahar nikalna hai",
                          "going to the shelter"],
    Category.RESCUE_NEEDED: ["need help", "send help", "madad bhejo", "bachao"],
}

# --- S2: same meaning, words the lexicon does NOT contain -----------------
# This is the OOV set. Every phrase here is deliberately absent from
# nlp/rules.py, so S2 measures exactly what a keyword list cannot reach.
SYNONYMS: Dict[Category, List[str]] = {
    Category.TRAPPED: ["we are hemmed in", "wedged beneath the slab",
                       "main yahin kaid hun", "nikasi ka raasta band hai",
                       "immured under concrete"],
    Category.COLLAPSE: ["the structure gave way", "masonry came down",
                        "bhavan dhah gaya", "chhajja toot gaya",
                        "the storey pancaked"],
    Category.MEDICAL: ["haemorrhaging", "needs resuscitation", "rakt bah raha hai",
                       "upchaar chahiye", "a casualty needs attention"],
    Category.FLOOD: ["inundated", "deluge here", "jalplavan ho gaya",
                     "nadi ka jal ghus gaya", "the ground floor is awash"],
    Category.FIRE: ["conflagration", "the place is ablaze", "agnikand hua hai",
                    "lapte uth rahi hain", "combustion on this level"],
    Category.SUPPLIES_NEEDED: ["provisions exhausted", "rations depleted",
                               "rasad khatam", "jal aapurti band"],
    Category.EVACUATION: ["we require relocation", "vacating the premises",
                          "sthanantaran karna hai", "palayan karna padega"],
    Category.RESCUE_NEEDED: ["requesting assistance", "sahayata chahiye",
                             "raahat bhejiye"],
}

# --- S3: known words, as people actually type them ------------------------
MISSPELL: Dict[str, List[str]] = {
    "paani bhar raha hai": ["pani bhr rha hai", "panee bhar raha h", "paani bharr raha hai"],
    "main fansa hun": ["mai fnsa hu", "main phansa hoon", "mai fansaa hun"],
    "aag lag gayi": ["ag lg gyi", "aag lagg gayi", "aaag lag gai"],
    "chhat gir gayi": ["chat gir gyi", "chhatt girr gayi", "cht gir gai"],
    "ghayal hai": ["ghyal hai", "ghayaal h", "gayal hai"],
    "madad bhejo": ["mdd bhejo", "madat bhejo", "madad bhjo"],
    "we are trapped": ["we r trapped", "we are trappd", "we're trapd"],
    "need help": ["ned help", "need halp", "nd help"],
}

# --- counts, mobility, severity, hazards ----------------------------------
COUNT_KNOWN = {n: [f"{n} people" if n > 1 else "1 person", f"{n} log"] for n in range(1, 8)}
COUNT_KNOWN[1] += ["one person", "ek aadmi"]
COUNT_KNOWN[2] += ["two people", "do log"]
COUNT_KNOWN[3] += ["three people", "teen log"]
COUNT_KNOWN[7] += ["seven people", "saat log"]

COUNT_SYN = {n: [f"a party of {n} persons" if n > 1 else "a lone individual",
                 f"{n} vyakti"] for n in range(1, 8)}

MOB_KNOWN = {
    Mobility.TRAPPED: ["cannot move", "hil nahi sakte"],
    Mobility.UNRESPONSIVE: ["unconscious", "behosh hai", "not responding"],
    Mobility.AMBULATORY: ["can walk", "chal sakte hain"],
}
MOB_SYN = {
    Mobility.TRAPPED: ["immobilised", "sthir hain", "pinned fast"],
    Mobility.UNRESPONSIVE: ["non-responsive", "achet avastha me", "comatose"],
    Mobility.AMBULATORY: ["mobile and upright", "paidal ja sakte hain"],
}

SEV_KNOWN = {
    Severity.SERIOUS: ["bleeding heavily", "critical", "gambhir hai"],
    Severity.MINOR: ["minor cuts", "halki chot"],
    Severity.NONE: ["nobody hurt", "koi ghayal nahi", "all safe"],
}
SEV_SYN = {
    Severity.SERIOUS: ["grievously wounded", "sankatmay sthiti"],
    Severity.MINOR: ["superficial abrasions", "saamanya kharonch"],
    Severity.NONE: ["all unscathed", "sabhi surakshit"],
}

HZ_KNOWN = {
    Hazard.FIRE: ["fire nearby", "aag bhi hai"],
    Hazard.WATER: ["water rising", "paani bhar raha hai"],
    Hazard.GAS: ["gas leak", "gas leak ho raha"],
    Hazard.STRUCTURAL: ["rubble shifting", "malba gir raha"],
    Hazard.ELECTRICAL: ["live wire", "bijli ka taar"],
    Hazard.CHEMICAL: ["chemical spill", "zeharila dhuaan"],
}
HZ_SYN = {
    Hazard.FIRE: ["blaze adjacent", "agni paas hai"],
    Hazard.WATER: ["inundation spreading", "jal star badh raha"],
    Hazard.GAS: ["vapour escaping", "gais ka risaav"],
    Hazard.STRUCTURAL: ["masonry unstable", "dhancha kamzor"],
    Hazard.ELECTRICAL: ["exposed conductor", "vidyut taar khula"],
    Hazard.CHEMICAL: ["noxious discharge", "rasayanik risaav"],
}

NOISE = ["", "", "please hurry", "jaldi", "second floor", "near the market",
         "battery low", "sending again"]

GIBBERISH = [
    "idk what is going on here", "the thing near the place is bad",
    "kuch to hua hai yahan", "pata nahi kya ho raha",
    "situation deteriorating, advise", "EMERGENCY DISTRESS CALL FROM DEVICE",
    "tell ravi we moved", "check on us when you can",
    "asdf test test", "hello hello anyone there",
    "sab kuch ajeeb hai", "something is wrong",
]


def _join(rng: random.Random, parts: List[str]) -> str:
    parts = [p for p in parts if p]
    rng.shuffle(parts)
    return rng.choice([", ", ", ", " and ", " - "]).join(parts)


# A category phrase carries its own hazard: "fire in the building" states a
# fire hazard whether or not a separate hazard phrase was added. Leaving
# these out of the gold label marked the encoder wrong for being right.
CATEGORY_IMPLIES_HAZARD: Dict[Category, int] = {
    Category.FIRE: int(Hazard.FIRE),
    Category.FLOOD: int(Hazard.WATER),
    Category.COLLAPSE: int(Hazard.STRUCTURAL),
    Category.TRAPPED: int(Hazard.STRUCTURAL),
}

# Several concepts can appear in one message. The spec resolves that by
# priority - being trapped outranks needing supplies - so the gold label
# follows the same order. This grades detection, not tie-breaking: the
# encoder still has to find the words before the order matters.
CATEGORY_PRIORITY: Tuple[Category, ...] = (
    Category.TRAPPED, Category.COLLAPSE, Category.MEDICAL, Category.FIRE,
    Category.FLOOD, Category.EVACUATION, Category.SUPPLIES_NEEDED,
    Category.RESCUE_NEEDED, Category.OTHER,
)

# Which category a hazard phrase also implies, if any.
HAZARD_IMPLIES_CATEGORY: Dict[int, Category] = {
    int(Hazard.FIRE): Category.FIRE,
    int(Hazard.WATER): Category.FLOOD,
    int(Hazard.STRUCTURAL): Category.COLLAPSE,
}


def _resolve_category(present: List[Category]) -> Category:
    for c in CATEGORY_PRIORITY:
        if c in present:
            return c
    return Category.OTHER


def _build(rng: random.Random, stratum: str) -> Sample:
    if stratum == "unintelligible":
        text = rng.choice(GIBBERISH)
        if rng.random() < 0.3:
            text += " " + rng.choice(GIBBERISH)
        return Sample(text, SemanticEnvelope(), stratum)

    syn = stratum == "synonyms"
    cat_bank = SYNONYMS if syn else KNOWN
    category = rng.choice([c for c in cat_bank if cat_bank[c]])
    parts = [rng.choice(cat_bank[category])]

    count = 0
    if rng.random() < 0.7:
        count = rng.randint(1, 7)
        bank = COUNT_SYN if syn else COUNT_KNOWN
        parts.append(rng.choice(bank[count]))

    mobility = Mobility.UNKNOWN
    if rng.random() < 0.55:
        mobility = rng.choice(list(MOB_KNOWN))
        parts.append(rng.choice((MOB_SYN if syn else MOB_KNOWN)[mobility]))

    severity = Severity.UNKNOWN
    if rng.random() < 0.5:
        severity = rng.choice(list(SEV_KNOWN))
        parts.append(rng.choice((SEV_SYN if syn else SEV_KNOWN)[severity]))

    # Concepts actually present in the text, starting with the chosen one.
    present = [category]
    hazard_flags = CATEGORY_IMPLIES_HAZARD.get(category, 0)

    for hz in rng.sample(list(HZ_KNOWN), rng.choice([0, 0, 1, 1, 2])):
        hazard_flags |= int(hz)
        parts.append(rng.choice((HZ_SYN if syn else HZ_KNOWN)[hz]))
        implied = HAZARD_IMPLIES_CATEGORY.get(int(hz))
        if implied is not None:
            present.append(implied)

    # A message saying both "no food" and "water rising" is a flood message.
    category = _resolve_category(present)

    parts.append(rng.choice(NOISE))
    text = _join(rng, parts)

    if stratum == "spelling":
        for correct, variants in MISSPELL.items():
            if correct in text:
                text = text.replace(correct, rng.choice(variants))
    elif stratum == "code_mixed":
        # Force both scripts into one sentence.
        text += rng.choice([" hum yahan hain", " please come fast",
                            " koi hai kya", " we need help now"])

    return Sample(text, SemanticEnvelope(
        category=category, casualty_count=count, mobility=mobility,
        hazard_flags=hazard_flags, severity=severity,
    ), stratum)


def generate(per_stratum: int = 2000, seed: int = 7) -> List[Sample]:
    rng = random.Random(seed)
    return [_build(rng, s) for s in STRATA for _ in range(per_stratum)]


if __name__ == "__main__":
    for s in STRATA:
        ex = _build(random.Random(hash(s) & 0xFFFF), s)
        print(f"[{s}]\n  {ex.text}\n  -> {ex.gold.category.name}, n={ex.gold.casualty_count}\n")
