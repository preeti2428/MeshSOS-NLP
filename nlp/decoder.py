"""
Envelope -> readable text, for the rescuer dashboard.

The 3 bytes that crossed the mesh are expanded back into a sentence a
responder can act on. Nothing is invented here: every clause comes from a
field that was actually set, and anything the encoder was unsure about is
shown as unknown rather than guessed.
"""

from __future__ import annotations

from typing import List

from nlp.envelope import (
    CASUALTY_MANY,
    Category,
    HAZARD_NAMES,
    Mobility,
    SemanticEnvelope,
    Severity,
)

CATEGORY_TEXT = {
    Category.OTHER: "Emergency",
    Category.MEDICAL: "Medical emergency",
    Category.TRAPPED: "People trapped",
    Category.FLOOD: "Flooding",
    Category.COLLAPSE: "Structural collapse",
    Category.FIRE: "Fire",
    Category.RESCUE_NEEDED: "Rescue requested",
    Category.SUPPLIES_NEEDED: "Supplies needed",
    Category.EVACUATION: "Evacuation needed",
}

MOBILITY_TEXT = {
    Mobility.AMBULATORY: "can move unaided",
    Mobility.TRAPPED: "cannot move",
    Mobility.UNRESPONSIVE: "unresponsive",
}

SEVERITY_TEXT = {
    Severity.NONE: "no injuries reported",
    Severity.MINOR: "minor injuries",
    Severity.SERIOUS: "serious injuries",
}

# Triage weight per category, replacing the hardcoded dict that used to live
# in inference/sos_hotspots.py. Driven by what the message said, not by which
# button someone managed to press.
CATEGORY_URGENCY = {
    Category.TRAPPED: 4.0,
    Category.COLLAPSE: 4.0,
    Category.MEDICAL: 3.0,
    Category.FIRE: 3.5,
    Category.FLOOD: 3.0,
    Category.EVACUATION: 2.5,
    Category.RESCUE_NEEDED: 2.0,
    Category.SUPPLIES_NEEDED: 1.5,
    Category.OTHER: 1.0,
}


def decode_to_text(env: SemanticEnvelope) -> str:
    """One line a responder can read off a map pin."""
    parts: List[str] = [CATEGORY_TEXT.get(env.category, "Emergency")]

    if env.casualty_count > 0:
        parts.append(env.casualty_text())

    if env.mobility in MOBILITY_TEXT:
        parts.append(MOBILITY_TEXT[env.mobility])

    if env.severity in SEVERITY_TEXT:
        parts.append(SEVERITY_TEXT[env.severity])

    hazards = env.hazards()
    if hazards:
        names = [HAZARD_NAMES[h] for h in hazards]
        if len(names) == 1:
            parts.append(f"hazard: {names[0]}")
        else:
            parts.append("hazards: " + ", ".join(names[:-1]) + f" and {names[-1]}")

    line = "; ".join(parts)

    conf = env.confidence_fraction()
    if conf < 0.45:
        line += "  [LOW CONFIDENCE — treat as unverified]"
    elif conf < 0.7:
        line += "  [partial]"

    return line


def urgency_score(env: SemanticEnvelope) -> float:
    """
    Triage priority from the message content.

    Replaces the fixed per-category weights in inference/sos_hotspots.py,
    which scored every app-generated SOS as OTHER = 1.0 because the category
    was hardcoded at the point of sending.
    """
    score = CATEGORY_URGENCY.get(env.category, 1.0)

    if env.mobility == Mobility.UNRESPONSIVE:
        score += 3.0
    elif env.mobility == Mobility.TRAPPED:
        score += 2.0

    if env.severity == Severity.SERIOUS:
        score += 2.5
    elif env.severity == Severity.MINOR:
        score += 0.5

    # More people at risk raises priority, but sub-linearly: one unresponsive
    # casualty should not be outranked by seven people who can walk.
    if env.casualty_count >= CASUALTY_MANY:
        score += 2.0
    elif env.casualty_count > 0:
        score += 0.4 * env.casualty_count

    score += 0.5 * len(env.hazards())

    # An uncertain report is still dispatched, only ranked a little lower.
    return round(score * (0.6 + 0.4 * env.confidence_fraction()), 3)
