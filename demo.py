#!/usr/bin/env python3
"""
MeshSOS semantic compression - end-to-end demo.

    python demo.py

Walks one emergency message through the whole path: free text on a handset,
compressed to a 3-byte envelope, packed into a 25-byte authenticated frame,
relayed across the mesh, and expanded again on the rescuer's screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol.compact_sos as cs
from nlp.decoder import decode_to_text, urgency_score
from nlp.rules import RuleEncoder
from protocol.compact_sos import CompactSosMessage, fits_single_frame, latlon_to_grid
from protocol.mesh_protocol import Header, MessageType, SosCategory, SosMessage

SHARED_KEY = b"meshsos-gateway-key"
cs.DEFAULT_GRID_ORIGIN = (28.60, 77.20)  # Delhi deployment

MESSAGES = [
    "Collapsed stairwell, 4 people trapped, one bleeding badly",
    "main malbe ke neeche fansa hun, 3 log hain, paani bhar raha hai",
    "2 bacche fanse hue hain, ek behosh hai",
    "idk something happened here maybe check",
]

RULE = "-" * 74


def show(text: str, encoder: RuleEncoder) -> None:
    print(RULE)
    print(f"HANDSET   {text}")

    result = encoder.encode(text)
    env = result.envelope

    header = Header(1, MessageType.SOS, 1000, 7, 8, 1700000000)
    gx, gy = latlon_to_grid(28.6139, 77.2090)

    full = SosMessage(
        header=header, lat=28.6139, lon=77.2090, accuracy=10.0,
        category=SosCategory.OTHER, battery=42, text=text,
        pubkey=b"\x00" * 32, signature=b"\x00" * 64,
    )
    full_len = len(full.to_bytes())
    full_frames = -(-full_len // 27)

    print(f"ENCODER   confidence {result.confidence:.2f}  "
          f"({'compress' if result.should_compress else 'FALL BACK to full text'})")
    for reason in result.reasons:
        print(f"            - {reason}")

    if not result.should_compress:
        print(f"WIRE      {full_len} B, {full_frames} frames  (uncompressed, as designed)")
        print("RESCUER   original text delivered verbatim")
        print(f"WHY       the encoder was not confident enough to compress, so "
              f"nothing was\n          inferred. Slower, but nothing is lost.")
        return

    msg = CompactSosMessage(
        header=header, grid_x=gx, grid_y=gy, battery=42, envelope=env,
    )
    msg.sign(SHARED_KEY)
    wire = msg.to_bytes()

    print(f"ENVELOPE  {env.pack().hex()}  (3 B)   {env.to_dict()}")
    print(f"WIRE      {wire.hex()}")
    print(f"          {len(wire)} B, 1 frame   vs   {full_len} B, {full_frames} frames   "
          f"({100 * (1 - len(wire) / full_len):.0f}% smaller)")
    print(f"          fits one legacy BLE advertising frame: {fits_single_frame(len(wire))}")

    received = CompactSosMessage.from_bytes(wire)
    ok = received.verify(SHARED_KEY)
    lat, lon = cs.grid_to_latlon(received.grid_x, received.grid_y)

    print(f"GATEWAY   auth tag valid: {ok}")
    print(f"RESCUER   {decode_to_text(received.envelope)}")
    print(f"          location {lat:.4f}, {lon:.4f} (~100 m cell)   "
          f"triage urgency {urgency_score(received.envelope)}")


def main() -> int:
    encoder = RuleEncoder()
    print("=" * 74)
    print("MeshSOS - semantic compression end to end")
    print("=" * 74)
    for text in MESSAGES:
        show(text, encoder)
    print(RULE)
    print("\nThe last message is the safety case: the encoder knew it did not")
    print("understand, so the original text was sent instead of a guess.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
