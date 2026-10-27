"""
Template-generated training corpus.

**Synthetic.** Every message here was produced by filling slots in a
template whose field values are known by construction, so the labels are
ground truth rather than something the rule encoder produced. That matters:
training on rule output would only teach a model to imitate the rules.

It is still synthetic, and a model trained on it has seen a narrower world
than real reports contain. The honest test is `nlp/sample_messages.py`,
which is hand-written and never templated - results there are what the
paper should quote.

Real training data means HumAID / CrisisBench with the envelope fields
annotated by people. That is Phase 2 and is not done.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from nlp.envelope import Category, Hazard, Mobility, SemanticEnvelope, Severity

# Surface forms per concept. Deliberately wider than the rule lexicon, and
# written as whole phrases rather than the keywords the rules look for.
CATEGORY_PHRASES = {
    Category.TRAPPED: [
        "we are stuck under the slab", "can't get out from under here",
        "pinned beneath the beam", "buried in the debris", "trapped inside",
        "malbe ke neeche dabe hain", "andar fanse hue hain", "nikal nahi pa rahe",
        "chhat ke neeche dab gaye", "bahar nahi aa pa rahe",
    ],
    Category.COLLAPSE: [
        "the building came down", "roof has caved in", "wall gave way",
        "whole structure fell", "slab collapsed on us",
        "imarat gir gayi", "chhat beth gayi", "deewar dhah gayi", "poora dhancha gir gaya",
    ],
    Category.MEDICAL: [
        "someone is badly hurt", "we need a medic", "bleeding will not stop",
        "needs a stretcher urgently", "injured and needs help",
        "ghayal hai koi", "chot lagi hai buri tarah", "doctor chahiye turant",
        "khoon ruk nahi raha", "ambulance bhejo",
    ],
    Category.FLOOD: [
        "water is coming up fast", "the street is under water",
        "current is too strong to cross", "we are on the roof as it rises",
        "paani tezi se chadh raha hai", "poora mohalla doob gaya",
        "paani ghar me ghus gaya", "baadh aa gayi hai",
    ],
    Category.FIRE: [
        "there are flames on this floor", "thick smoke everywhere",
        "the whole floor is alight", "cannot breathe for the smoke",
        "aag phail rahi hai", "dhuaan bhar gaya hai", "poori manzil jal rahi hai",
    ],
    Category.SUPPLIES_NEEDED: [
        "we have run out of drinking water", "no food since yesterday",
        "out of medicine here", "need blankets for the night",
        "peene ka paani khatam", "khana nahi hai kal se", "dawai chahiye",
    ],
    Category.EVACUATION: [
        "we have to get everyone out", "moving to the shelter now",
        "need to leave this block", "which way out is safe",
        "sab ko nikalna hai", "shelter tak jaana hai", "yahan se hatna padega",
    ],
    Category.RESCUE_NEEDED: [
        "please send someone", "anyone there", "we need help here",
        "koi hai", "madad bhejo", "jaldi koi aao",
    ],
}

MOBILITY_PHRASES = {
    Mobility.TRAPPED: ["cannot move at all", "stuck fast", "legs are pinned",
                       "hil nahi sakte", "bilkul fanse hain"],
    Mobility.UNRESPONSIVE: ["one is not waking up", "not responding to us",
                            "has stopped responding", "behosh pada hai",
                            "jawab nahi de raha", "hosh me nahi hai"],
    Mobility.AMBULATORY: ["we can still walk", "everyone is on their feet",
                          "able to move around", "chal sakte hain", "sab chal rahe hain"],
}

SEVERITY_PHRASES = {
    Severity.SERIOUS: ["bleeding heavily", "in a very bad way", "critical condition",
                       "halat gambhir hai", "bahut zyada chot", "khoon beh raha hai"],
    Severity.MINOR: ["just a few scratches", "nothing serious", "small cuts only",
                     "halki chot hai", "mamuli kharonch"],
    Severity.NONE: ["nobody is hurt", "everyone is unharmed", "no injuries at all",
                    "koi ghayal nahi", "sab theek hain"],
}

HAZARD_PHRASES = {
    Hazard.FIRE: ["there is fire nearby", "smoke is filling up", "aag bhi lagi hai"],
    Hazard.WATER: ["water keeps rising", "it is flooded here", "paani bhar raha hai"],
    Hazard.GAS: ["smells like a gas leak", "gas cylinder is leaking", "gas leak ho raha hai"],
    Hazard.STRUCTURAL: ["more of the wall is cracking", "rubble keeps shifting",
                        "malba aur gir raha hai", "deewar me darar hai"],
    Hazard.ELECTRICAL: ["a live wire is down", "there is current in the water",
                        "bijli ka taar gira hai"],
    Hazard.CHEMICAL: ["some chemical has spilled", "fumes are toxic", "zeharila dhuaan hai"],
}

COUNT_PHRASES = {
    1: ["one person", "a single person", "one of us", "ek aadmi", "ek log"],
    2: ["two people", "a pair of us", "do log", "do aadmi", "2 log"],
    3: ["three people", "three of us", "teen log", "3 log", "teen aadmi"],
    4: ["four people", "four of us", "char log", "4 log", "chaar aadmi"],
    5: ["five people", "paanch log", "5 log"],
    6: ["six people", "chhe log", "6 log"],
    7: ["seven or more people", "a whole group of people", "bahut saare log", "7 log"],
}

NOISE = [
    "", "", "", "please hurry", "jaldi karo", "it is dark here",
    "phone battery is low", "we are near the market", "second floor",
    "sending from the stairwell", "network nahi mil raha",
]


def _sentence(rng: random.Random, parts: List[str]) -> str:
    parts = [p for p in parts if p]
    rng.shuffle(parts)
    sep = rng.choice([", ", ", ", " and ", " - ", ". "])
    return sep.join(parts)


def generate(n: int, seed: int = 0) -> List[Tuple[str, SemanticEnvelope]]:
    """n (text, ground-truth envelope) pairs."""
    rng = random.Random(seed)
    out: List[Tuple[str, SemanticEnvelope]] = []

    for _ in range(n):
        category = rng.choice(list(CATEGORY_PHRASES))
        parts = [rng.choice(CATEGORY_PHRASES[category])]

        count = 0
        if rng.random() < 0.72:
            count = rng.choice(list(COUNT_PHRASES))
            parts.append(rng.choice(COUNT_PHRASES[count]))

        mobility = Mobility.UNKNOWN
        if rng.random() < 0.55:
            mobility = rng.choice(list(MOBILITY_PHRASES))
            parts.append(rng.choice(MOBILITY_PHRASES[mobility]))

        severity = Severity.UNKNOWN
        if rng.random() < 0.55:
            severity = rng.choice(list(SEVERITY_PHRASES))
            parts.append(rng.choice(SEVERITY_PHRASES[severity]))

        hazard_flags = 0
        for hz in rng.sample(list(HAZARD_PHRASES), rng.choice([0, 0, 1, 1, 2])):
            hazard_flags |= int(hz)
            parts.append(rng.choice(HAZARD_PHRASES[hz]))

        parts.append(rng.choice(NOISE))

        text = _sentence(rng, parts)
        if rng.random() < 0.3:
            text = text.upper() if rng.random() < 0.25 else text.capitalize()

        out.append((text, SemanticEnvelope(
            category=category,
            casualty_count=count,
            mobility=mobility,
            hazard_flags=hazard_flags,
            severity=severity,
        )))
    return out


if __name__ == "__main__":
    for t, e in generate(6, seed=1):
        print(f"{t}\n   -> {e.to_dict()}\n")
