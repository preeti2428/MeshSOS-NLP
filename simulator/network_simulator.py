"""
MeshSOS Discrete-Event Network Simulator.
Simulates multi-hop propagation, packet loss, mobility, fault injection
(collapses, battery deaths, partitions), and records routing metrics.
"""

from __future__ import annotations
import heapq
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from protocol.mesh_protocol import (
    Header,
    HeartbeatMessage,
    LowbatMessage,
    LowbatReason,
    MessageType,
    SosCategory,
    SosMessage,
    AckMessage,
    MeshMessage,
)
from simulator.node import SimNode
from simulator.relay_core import RelayCoreConfig


# ---------------------------------------------------------------------------
# Phase 0: BLE physical-layer constants
#
# Legacy advertising, BLE 1M PHY. An advertising PDU on air carries, besides
# the payload: preamble (1B), access address (4B), PDU header (2B) and CRC (3B)
# = 10 bytes. At 1 Mbit/s each bit takes 1 microsecond. T_IFS separates frames.
#
# Of the 31-byte advertising payload, the Flags AD structure costs 3 bytes and
# a 16-bit Service Data AD header costs 4 more, leaving 27 bytes usable for
# MeshSOS data. See docs/DECISIONS.md section 1.1.
# ---------------------------------------------------------------------------
BLE_PHY_BITRATE_BPS = 1_000_000
BLE_PDU_OVERHEAD_BYTES = 10
BLE_IFS_US = 150.0
ADV_USABLE_PAYLOAD_BYTES = 27

# Airtime used before Phase 0, kept so the legacy behaviour can be reproduced.
LEGACY_FIXED_AIR_TIME_MS = 5.0


class EventType(IntEnum):
    HEARTBEAT_TICK = 1
    RADIO_TX = 2
    NODE_FORWARD_CHECK = 3
    FAULT_INJECTION = 4
    SCHEDULED_SOS = 5


@dataclass(order=True)
class SimEvent:
    timestamp: float
    event_type: int
    node_id: int = field(compare=False)
    data: Any = field(default=None, compare=False)


@dataclass
class SimulationMetrics:
    total_transmissions: int = 0
    total_receptions: int = 0
    total_sos_generated: int = 0
    total_sos_delivered_to_gateway: int = 0
    latencies_sec: List[float] = field(default_factory=list)
    hops_traversed: List[int] = field(default_factory=list)
    dedupe_drops: int = 0
    suppression_drops: int = 0
    ttl_drops: int = 0
    per_node_tx: Dict[int, int] = field(default_factory=dict)
    per_node_rx: Dict[int, int] = field(default_factory=dict)

    # --- Phase 0: size-aware metrics ---
    total_fragments_tx: int = 0
    fragment_loss_drops: int = 0
    collision_drops: int = 0
    total_airtime_sec: float = 0.0
    payload_bytes_tx: int = 0

    @property
    def avg_fragments_per_tx(self) -> float:
        if self.total_transmissions == 0:
            return 0.0
        return self.total_fragments_tx / self.total_transmissions

    @property
    def sos_delivery_ratio(self) -> float:
        if self.total_sos_generated == 0:
            return 0.0
        return self.total_sos_delivered_to_gateway / self.total_sos_generated

    @property
    def avg_latency_sec(self) -> float:
        if not self.latencies_sec:
            return 0.0
        return sum(self.latencies_sec) / len(self.latencies_sec)

    @property
    def avg_hops(self) -> float:
        if not self.hops_traversed:
            return 0.0
        return sum(self.hops_traversed) / len(self.hops_traversed)

    @property
    def tx_per_delivered_message(self) -> float:
        if self.total_sos_delivered_to_gateway == 0:
            return float(self.total_transmissions)
        return self.total_transmissions / self.total_sos_delivered_to_gateway


class NetworkSimulator:
    def __init__(
        self,
        width: float = 1000.0,
        height: float = 1000.0,
        packet_loss_rate: float = 0.05,
        seed: int = 42,
        *,
        size_aware_airtime: bool = True,
        model_fragmentation: bool = True,
        model_collisions: bool = True,
        adv_payload_bytes: int = ADV_USABLE_PAYLOAD_BYTES,
        max_transmissions: Optional[int] = None,
    ):
        """
        size_aware_airtime: derive channel occupancy from the real serialized
            message length instead of a fixed 5 ms constant.
        model_fragmentation: a message longer than `adv_payload_bytes` is sent
            as several advertising frames. Each frame is lost independently, and
            the message is received only if every frame arrives.
        model_collisions: two frames overlapping in time at the same receiver
            destroy each other.

        Set all three False to reproduce the pre-Phase-0 behaviour exactly.
        """
        self.width = width
        self.height = height
        self.packet_loss_rate = packet_loss_rate
        self.rng = random.Random(seed)

        self.size_aware_airtime = size_aware_airtime
        self.model_fragmentation = model_fragmentation
        self.model_collisions = model_collisions
        self.adv_payload_bytes = adv_payload_bytes

        # Per-receiver busy windows, for the collision model:
        # receiver_id -> list of (start_time, end_time, reception_id)
        self.max_transmissions: Optional[int] = max_transmissions
        self.transmission_budget_exhausted = False
        self._rx_busy: Dict[int, List[Tuple[float, float, int]]] = {}
        self._collided_receptions: Set[int] = set()
        self._next_reception_id: int = 0

        self.nodes: Dict[int, SimNode] = {}
        self.event_queue: List[SimEvent] = []
        self.current_time: float = 0.0

        self.metrics = SimulationMetrics()

        # Gateway log of received packets (feeds the inference engine / dashboard)
        self.gateway_received_events: List[Dict[str, Any]] = []

        # SOS tracking for latency: (node_id, seq) -> origination_time
        self.sos_origination_times: Dict[Tuple[int, int], float] = {}
        self.delivered_sos_keys: Set[Tuple[int, int]] = set()
        # Initial TTL per SOS, so hop counts stay correct when TTL is varied.
        self.sos_initial_ttl: Dict[Tuple[int, int], int] = {}

    def add_node(self, node: SimNode):
        self.nodes[node.node_id] = node
        # Schedule initial heartbeat with slight random phase offset
        initial_hb = self.rng.uniform(0.5, max(1.0, node.heartbeat_interval))
        self.schedule_event(initial_hb, EventType.HEARTBEAT_TICK, node.node_id)

    def schedule_event(self, timestamp: float, event_type: EventType, node_id: int, data: Any = None):
        heapq.heappush(self.event_queue, SimEvent(timestamp, int(event_type), node_id, data))

    # ------------------------------------------------------------------
    # Phase 0 helpers: size -> fragments -> airtime
    # ------------------------------------------------------------------

    def message_size_bytes(self, message: MeshMessage) -> int:
        """Serialized on-air length of a message, in bytes."""
        try:
            return len(message.to_bytes())
        except Exception:
            # A message that cannot be serialized is charged one full frame.
            return self.adv_payload_bytes

    def fragment_count(self, payload_bytes: int) -> int:
        """Number of advertising frames needed to carry `payload_bytes`."""
        if not self.model_fragmentation:
            return 1
        return max(1, math.ceil(payload_bytes / self.adv_payload_bytes))

    def airtime_sec(self, payload_bytes: int, fragments: int) -> float:
        """Total channel occupancy for one logical message."""
        if not self.size_aware_airtime:
            return LEGACY_FIXED_AIR_TIME_MS / 1000.0

        us_per_bit = 1_000_000.0 / BLE_PHY_BITRATE_BPS
        remaining = payload_bytes
        total_us = 0.0
        for _ in range(fragments):
            chunk = min(remaining, self.adv_payload_bytes) if self.model_fragmentation else payload_bytes
            remaining -= chunk
            total_us += (BLE_PDU_OVERHEAD_BYTES + chunk) * 8 * us_per_bit + BLE_IFS_US
        return total_us / 1_000_000.0

    def _prune_busy(self, receiver_id: int) -> None:
        windows = self._rx_busy.get(receiver_id)
        if not windows:
            return
        now = self.current_time
        self._rx_busy[receiver_id] = [w for w in windows if w[1] >= now]

    def _register_reception(self, receiver_id: int, start: float, end: float, reception_id: int) -> bool:
        """
        Record an incoming frame window. Returns False if it collided with a
        frame already arriving at this receiver (both are then destroyed).
        """
        if not self.model_collisions:
            return True

        self._prune_busy(receiver_id)
        windows = self._rx_busy.setdefault(receiver_id, [])

        collided = False
        for (w_start, w_end, w_id) in windows:
            if start < w_end and w_start < end:  # overlap
                collided = True
                # The frame already in flight is destroyed too.
                if w_id not in self._collided_receptions:
                    self._collided_receptions.add(w_id)
                    self.metrics.collision_drops += 1

        windows.append((start, end, reception_id))

        if collided:
            self._collided_receptions.add(reception_id)
            self.metrics.collision_drops += 1
            return False
        return True

    def broadcast_packet(self, sender: SimNode, message: MeshMessage, air_time_ms: Optional[float] = None):
        # A flood with deduplication turned off does not terminate at any
        # useful scale, which is the whole point of measuring it. The budget
        # lets the run finish and records that the ceiling was reached, so
        # the storm shows up as a number instead of a hang.
        if self.max_transmissions is not None and \
                self.metrics.total_transmissions >= self.max_transmissions:
            self.transmission_budget_exhausted = True
            return

        payload_bytes = self.message_size_bytes(message)
        fragments = self.fragment_count(payload_bytes)

        if air_time_ms is not None:
            air_time = air_time_ms / 1000.0
        else:
            air_time = self.airtime_sec(payload_bytes, fragments)

        sender.tx_count += 1
        self.metrics.total_transmissions += 1
        self.metrics.total_fragments_tx += fragments
        self.metrics.total_airtime_sec += air_time
        self.metrics.payload_bytes_tx += payload_bytes
        self.metrics.per_node_tx[sender.node_id] = self.metrics.per_node_tx.get(sender.node_id, 0) + 1

        tx_start = self.current_time
        tx_time = self.current_time + air_time

        for receiver in self.nodes.values():
            if receiver.node_id == sender.node_id or not receiver.is_alive:
                continue

            # Check radio propagation range
            dist = sender.distance_to(receiver)
            if dist > sender.radio_range:
                continue

            # Stochastic channel loss, applied to EVERY fragment independently.
            # A fragmented message survives only if all of its frames arrive,
            # so a 6-frame message at 3% per-frame loss fails ~16.7% of the time
            # where a single-frame message fails 3% of the time.
            lost = False
            for _ in range(fragments):
                if self.rng.random() < self.packet_loss_rate:
                    lost = True
                    break
            if lost:
                self.metrics.fragment_loss_drops += 1
                continue

            reception_id = self._next_reception_id
            self._next_reception_id += 1

            if not self._register_reception(receiver.node_id, tx_start, tx_time, reception_id):
                continue  # collided on arrival

            # Schedule reception event at receiver
            self.schedule_event(
                tx_time,
                EventType.RADIO_TX,
                receiver.node_id,
                data={
                    "sender_id": sender.node_id,
                    "message": message,
                    "distance": dist,
                    "reception_id": reception_id,
                },
            )

    def process_radio_rx(self, receiver: SimNode, sender_id: int, message: MeshMessage, distance: float):
        receiver.rx_count += 1
        self.metrics.total_receptions += 1
        self.metrics.per_node_rx[receiver.node_id] = self.metrics.per_node_rx.get(receiver.node_id, 0) + 1

        # Record neighbour for digests
        receiver.record_neighbour(sender_id, self.current_time, distance)

        hdr = message.header
        msg_key = (hdr.node_id, hdr.seq)

        # If receiver is a gateway, log the event for backend/inference ingestion
        if receiver.is_gateway:
            event_record = message.to_dict()
            event_record["gateway_received_at"] = self.current_time
            event_record["gateway_id"] = receiver.node_id
            self.gateway_received_events.append(event_record)

            # If SOS, track delivery metrics
            if hdr.msg_type == MessageType.SOS and msg_key not in self.delivered_sos_keys:
                self.delivered_sos_keys.add(msg_key)
                self.metrics.total_sos_delivered_to_gateway += 1
                if msg_key in self.sos_origination_times:
                    latency = self.current_time - self.sos_origination_times[msg_key]
                    self.metrics.latencies_sec.append(latency)

                # Traversed hops: default initial TTL (e.g. 8) - current TTL + 1
                orig_ttl = self.sos_initial_ttl.get(msg_key, 8)
                hops = max(1, orig_ttl - hdr.ttl + 1)
                self.metrics.hops_traversed.append(hops)

                # Gateway automatically produces and floods an ACK back
                ack = AckMessage(
                    header=Header(1, MessageType.ACK, receiver.node_id, receiver.next_seq(), 8, int(self.current_time)),
                    acked_node_id=hdr.node_id,
                    acked_seq=hdr.seq,
                    gateway_id=receiver.node_id,
                    hop_count=1,
                    status=1,  # RECEIVED_BY_GATEWAY
                )
                self.broadcast_packet(receiver, ack)

        # Pass message to node's relay core
        rand_frac = self.rng.random()
        queued = receiver.relay.on_receive(message, self.current_time, random_fraction=rand_frac)
        if queued is not None:
            # Schedule forward check at queued scheduled time
            self.schedule_event(queued.scheduled_time, EventType.NODE_FORWARD_CHECK, receiver.node_id)

    def trigger_sos(self, node_id: int, text: str = "HELP NEEDED", category: SosCategory = SosCategory.OTHER, pubkey: bytes = b"\x00"*32, sig: bytes = b"\x00"*64, ttl: int = 8):
        node = self.nodes.get(node_id)
        if not node or not node.is_alive:
            return
        sos = node.create_sos(self.current_time, text=text, category=category, pubkey=pubkey, signature=sig, ttl=ttl)
        msg_key = (sos.header.node_id, sos.header.seq)
        self.sos_origination_times[msg_key] = self.current_time
        self.sos_initial_ttl[msg_key] = ttl
        self.metrics.total_sos_generated += 1
        # Broadcast locally
        self.broadcast_packet(node, sos)

    def trigger_compact_sos(
        self,
        node_id: int,
        envelope,
        ttl: int = 8,
        shared_key: bytes = b"meshsos-gateway-key",
    ):
        """
        Send a CompactSosMessage: the semantic envelope in a single 25-byte
        advertising frame, authenticated with a truncated HMAC.
        """
        from protocol.compact_sos import COMPACT_SOS_VERSION, CompactSosMessage

        node = self.nodes.get(node_id)
        if not node or not node.is_alive:
            return None

        # The simulator works in metres; one grid cell is 100 m, matching the
        # resolution the heartbeat already reports.
        msg = CompactSosMessage(
            header=Header(COMPACT_SOS_VERSION, MessageType.SOS, node.node_id, node.next_seq(), ttl, int(self.current_time)),
            grid_x=int(node.x // 100),
            grid_y=int(node.y // 100),
            battery=int(max(0, min(100, node.battery))),
            envelope=envelope,
        )
        msg.sign(shared_key)

        msg_key = (msg.header.node_id, msg.header.seq)
        self.sos_origination_times[msg_key] = self.current_time
        self.sos_initial_ttl[msg_key] = ttl
        self.metrics.total_sos_generated += 1
        self.broadcast_packet(node, msg)
        return msg

    def inject_collapse(self, center_x: float, center_y: float, radius: float):
        """Disaster collapse event: instantly destroys/disables all nodes inside region."""
        affected = []
        for node in self.nodes.values():
            if node.is_alive and math.hypot(node.x - center_x, node.y - center_y) <= radius:
                node.is_alive = False
                affected.append(node.node_id)
        return affected

    def inject_battery_death(self, node_id: int):
        """Simulates low-battery beacon followed by radio shutdown."""
        node = self.nodes.get(node_id)
        if not node or not node.is_alive:
            return
        node.battery = 3.0
        lowbat = node.create_lowbat(self.current_time, reason=LowbatReason.BATTERY_DEPLETED)
        # Broadcast final gasp beacon
        self.broadcast_packet(node, lowbat)
        # Shutdown radio
        node.is_alive = False

    def inject_user_stop(self, node_id: int):
        """User toggled stop-sharing in settings."""
        node = self.nodes.get(node_id)
        if not node or not node.is_alive:
            return
        lowbat = node.create_lowbat(self.current_time, reason=LowbatReason.USER_STOP)
        self.broadcast_packet(node, lowbat)
        node.is_alive = False

    def run_until(self, end_time: float):
        """Main simulation event loop."""
        while self.event_queue and self.event_queue[0].timestamp <= end_time:
            ev = heapq.heappop(self.event_queue)
            self.current_time = ev.timestamp
            node = self.nodes.get(ev.node_id)
            if not node:
                continue

            if ev.event_type == EventType.HEARTBEAT_TICK:
                if node.is_alive:
                    hb = node.create_heartbeat(self.current_time)
                    self.broadcast_packet(node, hb)
                    # Schedule next periodic heartbeat
                    next_time = self.current_time + node.heartbeat_interval
                    self.schedule_event(next_time, EventType.HEARTBEAT_TICK, node.node_id)

            elif ev.event_type == EventType.RADIO_TX:
                rx_data = ev.data
                # A frame marked as collided after it was scheduled never lands.
                rid = rx_data.get("reception_id")
                if rid is not None and rid in self._collided_receptions:
                    continue
                self.process_radio_rx(node, rx_data["sender_id"], rx_data["message"], rx_data["distance"])

            elif ev.event_type == EventType.NODE_FORWARD_CHECK:
                if node.is_alive:
                    ready_msg = node.relay.pop_ready_transmission(self.current_time)
                    if ready_msg:
                        self.broadcast_packet(node, ready_msg)

            elif ev.event_type == EventType.SCHEDULED_SOS:
                self.trigger_sos(node.node_id, text=ev.data.get("text", "EMERGENCY"))

        # Consolidate stats from node relay cores
        for node in self.nodes.values():
            self.metrics.dedupe_drops += node.relay.stats["dedupe_dropped"]
            self.metrics.suppression_drops += node.relay.stats["suppressed"]
            self.metrics.ttl_drops += node.relay.stats["ttl_expired"]

        return self.metrics
