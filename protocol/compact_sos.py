"""
CompactSosMessage — an authenticated SOS that fits one legacy BLE
advertising frame.

The existing SosMessage cannot: it is 124 bytes before a single character of
text is added, because it carries an Ed25519 public key (32 B) and signature
(64 B) on the wire. 96 of those 124 bytes are cryptography. Compressing the
text alone takes a 161-byte message down to 135 — six frames to five. It can
never reach one.

Three things give way to reach 27 bytes, and each is a deliberate trade:

  1. The public key stays off the wire (-32 B). The gateway resolves it from
     node_id against the enrolment record. Nodes must be enrolled first; an
     unenrolled node cannot send a compact SOS and falls back to the full one.
  2. The 64-byte signature becomes a 4-byte truncated HMAC (-60 B). This is
     NOT a digital signature: it authenticates to a gateway holding the
     shared key, and it does not prove origin to a third party. A 32-bit tag
     also means a 1-in-4-billion forgery chance per attempt, which is
     acceptable only because the gateway rate-limits and the full signed
     message follows. State this in the paper; do not call it equivalent.
  3. Exact lat/lon (8 B of float) becomes the same ~100 m grid cell the
     heartbeat already uses (-4 B). Rescuers get a cell, not a pin.

Layout, 25 bytes total:

    header          13   version, type, node_id, seq, ttl, timestamp
    grid_x           2   signed 16-bit grid cell
    grid_y           2   signed 16-bit grid cell
    battery          1   0-100 %
    envelope         3   SemanticEnvelope
    auth tag         4   truncated HMAC-SHA256
    ----------------------
    total           25   <= 27 usable bytes of a legacy advertising frame
"""

from __future__ import annotations

import hmac
import struct
from dataclasses import dataclass
from hashlib import sha256
from typing import Optional

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from nlp.envelope import ENVELOPE_BYTES, SemanticEnvelope
from protocol.mesh_protocol import HEADER_SIZE, Header, MessageType

BODY_FORMAT = ">hhB"                      # grid_x, grid_y, battery
BODY_SIZE = struct.calcsize(BODY_FORMAT)  # 5
# The full SosMessage shares message type SOS, so the version byte is what
# tells the two layouts apart on the wire.
COMPACT_SOS_VERSION = 2
AUTH_TAG_BYTES = 4
COMPACT_SOS_SIZE = HEADER_SIZE + BODY_SIZE + ENVELOPE_BYTES + AUTH_TAG_BYTES  # 25

# Usable payload of a legacy advertising frame after the Flags and 16-bit
# Service Data AD structures. See docs/DECISIONS.md section 1.1.
ADV_USABLE_PAYLOAD_BYTES = 27


@dataclass
class CompactSosMessage:
    header: Header
    grid_x: int
    grid_y: int
    battery: int
    envelope: SemanticEnvelope
    auth_tag: bytes = b"\x00" * AUTH_TAG_BYTES

    def __post_init__(self) -> None:
        if self.header.msg_type != MessageType.SOS:
            raise ValueError("CompactSosMessage requires msg_type SOS")
        if not 0 <= self.battery <= 100:
            raise ValueError(f"battery must be 0-100, got {self.battery}")
        if len(self.auth_tag) != AUTH_TAG_BYTES:
            raise ValueError(f"auth_tag must be {AUTH_TAG_BYTES} bytes, got {len(self.auth_tag)}")

    # --- authentication ---------------------------------------------------

    def unauthenticated_bytes(self) -> bytes:
        """
        The bytes the auth tag covers.

        TTL is zeroed first. It is decremented at every hop, so a tag computed
        over the live header would fail verification everywhere except the
        originating node. Excluding it means the tag authenticates the report,
        not the path it travelled - which is what the gateway needs to check.
        """
        stable_header = Header(
            version=self.header.version,
            msg_type=self.header.msg_type,
            node_id=self.header.node_id,
            seq=self.header.seq,
            ttl=0,
            timestamp=self.header.timestamp,
        )
        return (
            stable_header.pack()
            + struct.pack(BODY_FORMAT, self.grid_x, self.grid_y, self.battery)
            + self.envelope.pack()
        )

    def compute_tag(self, shared_key: bytes) -> bytes:
        """Truncated HMAC-SHA256 over everything that precedes the tag."""
        mac = hmac.new(shared_key, self.unauthenticated_bytes(), sha256).digest()
        return mac[:AUTH_TAG_BYTES]

    def sign(self, shared_key: bytes) -> None:
        self.auth_tag = self.compute_tag(shared_key)

    def verify(self, shared_key: bytes) -> bool:
        return hmac.compare_digest(self.auth_tag, self.compute_tag(shared_key))

    # --- wire format ------------------------------------------------------

    def to_bytes(self) -> bytes:
        data = self.unauthenticated_bytes() + self.auth_tag
        if len(data) != COMPACT_SOS_SIZE:
            raise AssertionError(f"compact SOS must be {COMPACT_SOS_SIZE} B, got {len(data)}")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompactSosMessage":
        if len(data) != COMPACT_SOS_SIZE:
            raise ValueError(f"expected {COMPACT_SOS_SIZE} bytes, got {len(data)}")
        header = Header.unpack(data)
        off = HEADER_SIZE
        grid_x, grid_y, battery = struct.unpack_from(BODY_FORMAT, data, off)
        off += BODY_SIZE
        envelope = SemanticEnvelope.unpack(data[off:off + ENVELOPE_BYTES])
        off += ENVELOPE_BYTES
        return cls(
            header=header,
            grid_x=grid_x,
            grid_y=grid_y,
            battery=battery,
            envelope=envelope,
            auth_tag=data[off:off + AUTH_TAG_BYTES],
        )

    def to_dict(self) -> dict:
        return {
            "version": self.header.version,
            "type": "SOS",
            "compact": True,
            "node_id": self.header.node_id,
            "seq": self.header.seq,
            "ttl": self.header.ttl,
            "timestamp": self.header.timestamp,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "battery": self.battery,
            **self.envelope.to_dict(),
        }


# --- grid conversion -------------------------------------------------------
# ~100 m cells, the same scheme the heartbeat uses.
#
# The grid is relative to a deployment origin, not absolute. Absolute
# coordinates do not fit: longitude 77.209 at 0.001-degree resolution is
# 77209, past the 32767 ceiling of a signed 16-bit field. Encoding an offset
# from an origin set at enrolment gives +/-32767 cells, about +/-3276 km,
# which covers any single disaster deployment at 100 m resolution.
#
# Both ends must agree on the origin. It is configuration, distributed with
# the enrolment record, never sent per message.

GRID_CELL_DEGREES = 0.001  # about 111 m of latitude

# Overwrite at startup for the deployment region.
DEFAULT_GRID_ORIGIN: tuple[float, float] = (0.0, 0.0)

GRID_MIN, GRID_MAX = -32768, 32767


def latlon_to_grid(
    lat: float, lon: float, origin: Optional[tuple[float, float]] = None
) -> tuple[int, int]:
    olat, olon = origin if origin is not None else DEFAULT_GRID_ORIGIN
    gx = int(round((lat - olat) / GRID_CELL_DEGREES))
    gy = int(round((lon - olon) / GRID_CELL_DEGREES))
    if not (GRID_MIN <= gx <= GRID_MAX and GRID_MIN <= gy <= GRID_MAX):
        raise ValueError(
            f"({lat}, {lon}) is outside the 16-bit grid around origin "
            f"{(olat, olon)}; sender must fall back to a full SOS"
        )
    return gx, gy


def grid_to_latlon(
    grid_x: int, grid_y: int, origin: Optional[tuple[float, float]] = None
) -> tuple[float, float]:
    olat, olon = origin if origin is not None else DEFAULT_GRID_ORIGIN
    return olat + grid_x * GRID_CELL_DEGREES, olon + grid_y * GRID_CELL_DEGREES


def fits_single_frame(nbytes: int) -> bool:
    return nbytes <= ADV_USABLE_PAYLOAD_BYTES
