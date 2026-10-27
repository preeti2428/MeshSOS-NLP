"""
MeshSOS semantic envelope.

A free-text emergency report compressed into 3 bytes, so that a complete
authenticated SOS fits inside one legacy BLE advertising frame.

Layout (24 bits, big-endian):

    bit  23..20   category        (4)  what kind of emergency
    bit  19..17   casualty_count  (3)  0-6 people, 7 = "7 or more"
    bit  16..15   mobility        (2)  can walk / trapped / unresponsive
    bit  14..9    hazard_flags    (6)  fire water gas structural electrical chemical
    bit   8..7    severity        (2)  none / minor / serious
    bit   6..4    confidence      (3)  encoder's own certainty, 0-7
    bit   3..0    reserved        (4)  future use, must be 0

Design note: `confidence` travels with the data on purpose. A rescuer
dashboard must be able to show how much of a report was inferred rather
than stated, and the relay layer uses it to decide whether to fall back to
sending the original text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List

ENVELOPE_BYTES = 3


class Category(IntEnum):
    OTHER = 0
    MEDICAL = 1
    TRAPPED = 2
    FLOOD = 3
    COLLAPSE = 4
    FIRE = 5
    RESCUE_NEEDED = 6
    SUPPLIES_NEEDED = 7
    EVACUATION = 8


class Mobility(IntEnum):
    UNKNOWN = 0
    AMBULATORY = 1      # can move unaided
    TRAPPED = 2         # cannot move, pinned or enclosed
    UNRESPONSIVE = 3    # not responding


class Severity(IntEnum):
    UNKNOWN = 0
    NONE = 1
    MINOR = 2
    SERIOUS = 3


class Hazard(IntEnum):
    FIRE = 1
    WATER = 2
    GAS = 4
    STRUCTURAL = 8
    ELECTRICAL = 16
    CHEMICAL = 32


HAZARD_NAMES = {
    Hazard.FIRE: "fire",
    Hazard.WATER: "water",
    Hazard.GAS: "gas",
    Hazard.STRUCTURAL: "structural collapse",
    Hazard.ELECTRICAL: "live electricity",
    Hazard.CHEMICAL: "chemicals",
}

CASUALTY_MANY = 7  # the saturating bucket: "7 or more"


@dataclass(frozen=True)
class SemanticEnvelope:
    category: Category = Category.OTHER
    casualty_count: int = 0        # 0-7, where 7 means "7 or more"
    mobility: Mobility = Mobility.UNKNOWN
    hazard_flags: int = 0          # OR of Hazard values, 0-63
    severity: Severity = Severity.UNKNOWN
    confidence: int = 0            # 0-7

    def __post_init__(self) -> None:
        if not 0 <= int(self.category) <= 15:
            raise ValueError(f"category out of range: {self.category}")
        if not 0 <= self.casualty_count <= 7:
            raise ValueError(f"casualty_count must be 0-7, got {self.casualty_count}")
        if not 0 <= int(self.mobility) <= 3:
            raise ValueError(f"mobility out of range: {self.mobility}")
        if not 0 <= self.hazard_flags <= 63:
            raise ValueError(f"hazard_flags must be 0-63, got {self.hazard_flags}")
        if not 0 <= int(self.severity) <= 3:
            raise ValueError(f"severity out of range: {self.severity}")
        if not 0 <= self.confidence <= 7:
            raise ValueError(f"confidence must be 0-7, got {self.confidence}")

    # --- wire format -----------------------------------------------------

    def pack(self) -> bytes:
        value = (
            (int(self.category) & 0xF) << 20
            | (self.casualty_count & 0x7) << 17
            | (int(self.mobility) & 0x3) << 15
            | (self.hazard_flags & 0x3F) << 9
            | (int(self.severity) & 0x3) << 7
            | (self.confidence & 0x7) << 4
        )
        return value.to_bytes(ENVELOPE_BYTES, "big")

    @classmethod
    def unpack(cls, data: bytes) -> "SemanticEnvelope":
        if len(data) != ENVELOPE_BYTES:
            raise ValueError(f"envelope must be exactly {ENVELOPE_BYTES} bytes, got {len(data)}")
        value = int.from_bytes(data, "big")
        return cls(
            category=Category((value >> 20) & 0xF),
            casualty_count=(value >> 17) & 0x7,
            mobility=Mobility((value >> 15) & 0x3),
            hazard_flags=(value >> 9) & 0x3F,
            severity=Severity((value >> 7) & 0x3),
            confidence=(value >> 4) & 0x7,
        )

    # --- helpers ---------------------------------------------------------

    def hazards(self) -> List[Hazard]:
        return [h for h in Hazard if self.hazard_flags & int(h)]

    def casualty_text(self) -> str:
        if self.casualty_count == 0:
            return "unknown number of people"
        if self.casualty_count == CASUALTY_MANY:
            return "7 or more people"
        if self.casualty_count == 1:
            return "1 person"
        return f"{self.casualty_count} people"

    def confidence_fraction(self) -> float:
        """Confidence as 0.0-1.0, for thresholding."""
        return self.confidence / 7.0

    def to_dict(self) -> dict:
        return {
            "category": self.category.name,
            "casualty_count": self.casualty_count,
            "mobility": self.mobility.name,
            "hazards": [h.name for h in self.hazards()],
            "severity": self.severity.name,
            "confidence": round(self.confidence_fraction(), 3),
        }
