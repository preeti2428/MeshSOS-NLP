"""
Hand-written emergency messages for development and evaluation.

NOT a benchmark dataset. These were written by the authors to exercise the
encoder, and results on them must be reported as such. A real evaluation
needs HumAID / CrisisBench plus human annotation of the envelope fields;
that work is Phase 2 and is not done yet.

Each entry carries the envelope a human annotator would assign, so the
encoder can be scored field by field rather than on a single joint label.
"""

from __future__ import annotations

from typing import List, Tuple

from nlp.envelope import Category, Hazard, Mobility, SemanticEnvelope, Severity

# (text, expected envelope). Confidence is not annotated - it is the
# encoder's own output, not ground truth.
SAMPLES: List[Tuple[str, SemanticEnvelope]] = [
    # --- English, clearly encodable ---
    ("Collapsed stairwell, 4 people trapped, one bleeding badly",
     SemanticEnvelope(Category.TRAPPED, 4, Mobility.TRAPPED, int(Hazard.STRUCTURAL), Severity.SERIOUS)),
    ("Building collapsed, 2 people buried under rubble",
     SemanticEnvelope(Category.TRAPPED, 2, Mobility.TRAPPED, int(Hazard.STRUCTURAL), Severity.UNKNOWN)),
    ("Fire on second floor, smoke everywhere, 3 people cannot get out",
     SemanticEnvelope(Category.TRAPPED, 3, Mobility.TRAPPED, int(Hazard.FIRE), Severity.UNKNOWN)),
    ("Water rising fast, 5 people on the roof, all safe for now",
     SemanticEnvelope(Category.FLOOD, 5, Mobility.UNKNOWN, int(Hazard.WATER), Severity.NONE)),
    ("One person unconscious, not breathing, need ambulance",
     SemanticEnvelope(Category.MEDICAL, 1, Mobility.UNRESPONSIVE, 0, Severity.SERIOUS)),
    ("Gas leak in the kitchen, everyone evacuated, nobody hurt",
     SemanticEnvelope(Category.EVACUATION, 0, Mobility.UNKNOWN, int(Hazard.GAS), Severity.NONE)),
    ("Live wire down across the road, 2 people injured",
     SemanticEnvelope(Category.MEDICAL, 2, Mobility.UNKNOWN, int(Hazard.ELECTRICAL), Severity.UNKNOWN)),
    ("Need water and food, 7 people here, all safe",
     SemanticEnvelope(Category.SUPPLIES_NEEDED, 7, Mobility.UNKNOWN, 0, Severity.NONE)),
    ("Minor cuts only, we can walk, heading to the shelter",
     SemanticEnvelope(Category.EVACUATION, 0, Mobility.AMBULATORY, 0, Severity.MINOR)),
    ("Roof collapsed, 6 people trapped inside, heavy bleeding",
     SemanticEnvelope(Category.TRAPPED, 6, Mobility.TRAPPED, int(Hazard.STRUCTURAL), Severity.SERIOUS)),

    # --- Romanised Hindi / code-mixed ---
    ("main malbe ke neeche fansa hun, 3 log hain",
     SemanticEnvelope(Category.TRAPPED, 3, Mobility.TRAPPED, int(Hazard.STRUCTURAL), Severity.UNKNOWN)),
    ("2 bacche fanse hue hain, ek behosh hai",
     SemanticEnvelope(Category.TRAPPED, 2, Mobility.UNRESPONSIVE, 0, Severity.SERIOUS)),
    ("aag lag gayi hai, sab log bahar nikal gaye, koi ghayal nahi",
     SemanticEnvelope(Category.FIRE, 7, Mobility.UNKNOWN, int(Hazard.FIRE), Severity.NONE)),
    ("paani bhar raha hai tezi se, 4 log chhat pe hain",
     SemanticEnvelope(Category.FLOOD, 4, Mobility.UNKNOWN, int(Hazard.WATER), Severity.UNKNOWN)),
    ("chhat gir gayi, do aadmi dabe hue hain",
     SemanticEnvelope(Category.TRAPPED, 2, Mobility.TRAPPED, int(Hazard.STRUCTURAL), Severity.UNKNOWN)),
    ("ek aadmi ko bahut chot lagi hai, khoon beh raha hai",
     SemanticEnvelope(Category.MEDICAL, 1, Mobility.UNKNOWN, 0, Severity.SERIOUS)),
    ("hum 5 log safe hain, khana paani chahiye",
     SemanticEnvelope(Category.SUPPLIES_NEEDED, 5, Mobility.UNKNOWN, 0, Severity.UNKNOWN)),
    ("gas leak ho raha hai, jaldi aao",
     SemanticEnvelope(Category.RESCUE_NEEDED, 0, Mobility.UNKNOWN, int(Hazard.GAS), Severity.UNKNOWN)),
    ("bijli ka taar gira hai, koi haath mat lagana",
     SemanticEnvelope(Category.OTHER, 0, Mobility.UNKNOWN, int(Hazard.ELECTRICAL), Severity.UNKNOWN)),
    ("madad chahiye, hum teen log hain, hil nahi sakte",
     SemanticEnvelope(Category.TRAPPED, 3, Mobility.TRAPPED, 0, Severity.UNKNOWN)),

    # --- deliberately hard: these SHOULD get low confidence and fall back ---
    ("idk something happened here maybe check",
     SemanticEnvelope(Category.OTHER, 0, Mobility.UNKNOWN, 0, Severity.UNKNOWN)),
    ("EMERGENCY DISTRESS CALL FROM DEVICE",
     SemanticEnvelope(Category.OTHER, 0, Mobility.UNKNOWN, 0, Severity.UNKNOWN)),
    ("the thing near the place is bad now please",
     SemanticEnvelope(Category.OTHER, 0, Mobility.UNKNOWN, 0, Severity.UNKNOWN)),
    ("tell my brother ravi at 42 nehru road that we moved to the school",
     SemanticEnvelope(Category.OTHER, 0, Mobility.UNKNOWN, 0, Severity.UNKNOWN)),
    ("situation deteriorating rapidly, advise immediately",
     SemanticEnvelope(Category.OTHER, 0, Mobility.UNKNOWN, 0, Severity.UNKNOWN)),
]


def texts() -> List[str]:
    return [t for t, _ in SAMPLES]
