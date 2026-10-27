# Threat model for the 25-byte compact SOS

The compact SOS carries a 4-byte truncated HMAC-SHA256 tag instead of the
full message's 64-byte Ed25519 signature. Four bytes of authentication
looks alarming until the question is asked properly: *against whom, to
protect what, at what cost in delivered messages?* This file asks it
properly. The numbers reuse the airtime model from the paper
(1 Mbit/s PHY, 10 B PDU overhead, 150 µs inter-frame space).

## What is being protected

One thing: **the integrity of the rescuer's dashboard.** A forged SOS
sends rescuers to an empty field; forged at scale, it buries real
victims in noise. The tag exists so that no message enters the dashboard
unless it was produced by a device holding the key.

Confidentiality is **deliberately not provided**. An SOS is a cry for
help; encrypting it so that only one gateway could read it would trade
availability for secrecy in exactly the wrong place. The privacy cost is
real and stated: anyone in radio range learns that someone at grid cell
(x, y) is in trouble.

## The adversary

Software-defined radios and phones are assumed; the adversary can
receive everything and transmit anything. They do not hold the key and
cannot break HMAC-SHA256 itself; their only route is guessing tags or
replaying. Physical capture of an enrolled phone is treated separately
below.

## Attacks and what happens to them

**Blind forgery.** A forged message needs a valid 32-bit tag; one
attempt succeeds with probability 2⁻³². A 25-byte frame occupies
430 µs of air, so one advertising channel carries at most ~2,300
frames per second. An attacker saturating the channel — which is
itself a visible denial of service — expects a first forgery after

    2³² / 2300 ≈ 1.85 × 10⁶ s  ≈  21 days

of continuous 100% transmission. At a stealthier 100 frames per
second it is about 1.4 years, for one fake dot on a map. The gateway
can further rate-limit verification failures per node id, making
sustained guessing visible in exactly the place it is aimed at.

**Replay.** The tag covers `node_id`, `seq` and `timestamp` (TTL is
zeroed, being the one field that legitimately changes in flight), so a
recorded message can only be replayed verbatim. The relay layer already
deduplicates on `(node_id, seq)`, and the gateway does the same, so a
replay within the mesh is dropped as a duplicate; a replay hours later
carries a stale timestamp the gateway can bound. The residual effect
is at most refreshing a report the victim already sent.

**Insider forgery — the honest limitation.** The prototype uses a
single group key, hard-coded in source. Anyone holding one enrolled
device can therefore forge messages *as any node id*. The deployment
design closes this at zero byte cost: at enrollment each node receives
`K_node = KDF(K_master, node_id)`, and the gateway, which already
resolves `node_id` to look up state, derives the same key per message.
Compromise of one phone then forges only that phone's identity, which
no symmetric scheme can prevent.

**Relay amplification.** Relays hold no key and forward unverified
traffic, so each injected frame with a fresh `(node_id, seq)` is
re-broadcast by the mesh: an amplification the attacker gets free.
Counter-based suppression (k = 2) and TTL already bound the
amplification per frame; a per-`node_id` token bucket at relays is the
designed mitigation and is not yet implemented. Note that an attacker
with a transmitter can always jam the band outright, so the marginal
power this attack grants is modest.

**Jamming.** Out of scope. No protocol at this layer survives a
jammed channel; the design goal is that the system fails no worse
than the radio under it.

## Implementation notes

- The Python gateway compares tags with `hmac.compare_digest`
  (constant-time). The Kotlin verifier uses `MessageDigest.isEqual`
  for the same reason. The broadcast medium offers no practical timing
  oracle, but a constant-time compare costs nothing.
- Truncating HMAC-SHA256 to its leading 4 bytes is standard practice
  (RFC 2104 §5 permits truncation; security against forgery is then
  bounded by the tag length, as computed above).

## The upgrade path

CuMAC-style accumulation: split a 16-byte tag into four 4-byte
segments and rotate them across the store-and-forward re-broadcasts
the system already performs every 10 s. The gateway acts on the first
verified segment and accumulates to full 128-bit strength over
subsequent copies, at zero additional bytes per frame. This is the
clearest improvement available and is future work, not a measured
result.
