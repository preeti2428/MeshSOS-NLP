"""
Invariants for the semantic compression path.

The byte-budget test is the important one: it fails the build the moment a
new field pushes the compact SOS past what a legacy BLE advertising frame
can carry. That is the whole premise of the design, so it is asserted, not
assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.envelope import (
    Category,
    Hazard,
    Mobility,
    SemanticEnvelope,
    Severity,
)
from nlp.decoder import decode_to_text, urgency_score
from nlp.rules import RuleEncoder
from protocol.compact_sos import (
    ADV_USABLE_PAYLOAD_BYTES,
    COMPACT_SOS_SIZE,
    CompactSosMessage,
    latlon_to_grid,
)
from protocol.mesh_protocol import Header, MessageType

KEY = b"test-gateway-key"


def _msg(envelope: SemanticEnvelope, ttl: int = 8) -> CompactSosMessage:
    return CompactSosMessage(
        header=Header(1, MessageType.SOS, 1000, 7, ttl, 1700000000),
        grid_x=13, grid_y=9, battery=42, envelope=envelope,
    )


# --- the budget -----------------------------------------------------------

def test_compact_sos_fits_one_advertising_frame():
    env = SemanticEnvelope(Category.TRAPPED, 7, Mobility.TRAPPED, 63, Severity.SERIOUS, 7)
    m = _msg(env)
    m.sign(KEY)
    assert len(m.to_bytes()) == COMPACT_SOS_SIZE
    assert len(m.to_bytes()) <= ADV_USABLE_PAYLOAD_BYTES, (
        "compact SOS no longer fits a legacy BLE advertising frame; "
        "a field was added without taking one away"
    )


# --- envelope -------------------------------------------------------------

def test_envelope_is_three_bytes_and_round_trips():
    env = SemanticEnvelope(Category.MEDICAL, 3, Mobility.UNRESPONSIVE,
                           int(Hazard.FIRE) | int(Hazard.GAS), Severity.SERIOUS, 5)
    packed = env.pack()
    assert len(packed) == 3
    assert SemanticEnvelope.unpack(packed) == env


@pytest.mark.parametrize("bad", [
    {"casualty_count": 8},
    {"hazard_flags": 64},
    {"confidence": 8},
])
def test_envelope_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        SemanticEnvelope(**bad)


# --- authentication -------------------------------------------------------

def test_tag_survives_ttl_decrement():
    """TTL changes at every hop, so the tag must not cover it."""
    env = SemanticEnvelope(Category.TRAPPED, 2, Mobility.TRAPPED, 0, Severity.UNKNOWN, 6)
    origin = _msg(env, ttl=8)
    origin.sign(KEY)
    for ttl in (7, 6, 5, 1):
        hop = CompactSosMessage(
            header=Header(1, MessageType.SOS, 1000, 7, ttl, 1700000000),
            grid_x=origin.grid_x, grid_y=origin.grid_y, battery=origin.battery,
            envelope=origin.envelope, auth_tag=origin.auth_tag,
        )
        assert hop.verify(KEY), f"tag failed at ttl={ttl}"


def test_tampered_envelope_is_rejected():
    env = SemanticEnvelope(Category.TRAPPED, 2, Mobility.TRAPPED, 0, Severity.UNKNOWN, 6)
    m = _msg(env)
    m.sign(KEY)
    raw = bytearray(m.to_bytes())
    raw[HEADER_END := 18] ^= 0x01          # flip a bit inside the envelope
    assert not CompactSosMessage.from_bytes(bytes(raw)).verify(KEY)


def test_wrong_key_is_rejected():
    m = _msg(SemanticEnvelope(Category.FIRE, 1, Mobility.AMBULATORY, 1, Severity.MINOR, 4))
    m.sign(KEY)
    assert not m.verify(b"some-other-key")


# --- encoder safety -------------------------------------------------------

def test_unintelligible_text_falls_back_instead_of_guessing():
    for text in ("idk something happened here maybe check",
                 "EMERGENCY DISTRESS CALL FROM DEVICE",
                 "the thing near the place is bad now please"):
        r = RuleEncoder().encode(text)
        assert not r.should_compress, f"should have fallen back: {text!r}"
        assert r.envelope.category == Category.OTHER


def test_casualty_count_is_not_silently_lowered():
    """The failure that matters: under-reporting how many people are at risk."""
    cases = [
        ("4 people trapped under rubble", 4),
        ("2 bacche fanse hue hain", 2),
        ("teen log fanse hain", 3),
        ("7 people need water", 7),
    ]
    for text, expected in cases:
        got = RuleEncoder().encode(text).envelope.casualty_count
        assert got >= expected, f"{text!r}: reported {got}, actual {expected}"


def test_negation_is_honoured_in_both_languages():
    assert RuleEncoder().encode("fire contained, nobody hurt").envelope.severity == Severity.NONE
    assert RuleEncoder().encode("aag bujh gayi, koi ghayal nahi").envelope.severity == Severity.NONE


# --- triage ---------------------------------------------------------------

def test_unresponsive_outranks_a_larger_mobile_group():
    critical = SemanticEnvelope(Category.TRAPPED, 1, Mobility.UNRESPONSIVE, 0, Severity.SERIOUS, 6)
    crowd = SemanticEnvelope(Category.SUPPLIES_NEEDED, 7, Mobility.AMBULATORY, 0, Severity.NONE, 6)
    assert urgency_score(critical) > urgency_score(crowd)


def test_decoded_text_flags_low_confidence():
    unsure = SemanticEnvelope(Category.TRAPPED, 2, Mobility.TRAPPED, 0, Severity.UNKNOWN, 1)
    assert "LOW CONFIDENCE" in decode_to_text(unsure)


def test_grid_rejects_coordinates_outside_the_deployment():
    with pytest.raises(ValueError):
        latlon_to_grid(28.6, 77.2, origin=(0.0, 0.0))   # 77209 cells, past 16 bits
