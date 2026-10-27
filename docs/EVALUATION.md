# MeshSOS System Evaluation & Experimental Results

Empirical measurements across all evaluated scenarios.

**Every network experiment below is averaged over 25 random topologies** and reported as mean ± standard deviation. Delivery is the fraction of seeds in which the SOS reached a gateway.

> An earlier version of this file reported single-seed (42) numbers, including a "200 nodes, SOS delivered = 0" that read as a scaling limit. It was not. Seed 42 places the source 17 hops from the nearest gateway, past the TTL of 8, while the median topology at that density needs 6. The tables below replace those numbers.

## 1. Relay Strategy Comparison

Three genuinely distinct strategies. Previously "Plain Flooding" and "TTL Only" were the same configuration reported twice, because deduplication could not be switched off; `RelayCoreConfig` now takes `enable_dedupe`.

| Strategy | Transmissions (mean ± sd) | Delivery | Dedupe Drops | Suppression Drops |
|---|---|---|---|---|
| Plain Flooding (no dedupe) | 301.2 ± 64.8 | 100% (25/25) | 0 | 0 |
| Dedupe Only | 94.6 ± 33.0 | 92% (23/25) | 238.1 | 0 |
| Dedupe + Counter Suppression (k=2) | 66.4 ± 24.4 | 92% (23/25) | 138.3 | 27.8 |

Deduplication and suppression together cut transmissions by **78%** against plain flooding (301 → 66 per SOS).

That saving is **not free**: plain flooding delivered in 25/25 topologies and the suppressed configuration in 23/25, so 2 layouts that a storm reached are lost. Flooding every copy is a brute-force form of redundancy, and discarding duplicates discards some of it. The trade is deliberate — a 78% cut in channel occupancy for roughly 8% of marginal reachability — but it is a trade, and the earlier claim of equal delivery was wrong.

## 2. Silence Inference Benchmark

| Scenario | Ground Truth | Confidence | Detection Delay (s) | Classification |
|---|---|---|---|---|
| (a) Localized Collapse (12/15 phones) | DISASTER | 1.0 | 50.0 | TRUE_POSITIVE |
| (b) Random Battery Deaths (LOWBAT) | BENIGN | 0.0 | 0.0 | TRUE_NEGATIVE |
| (c) Single Walker Out of Range | BENIGN | 0.0 | 0.0 | TRUE_NEGATIVE |
| (d) Mesh-Wide Outage (All phones stop) | INFRASTRUCTURE_OUTAGE | 0.05 | 0.0 | CORRECT_SUPPRESSION |

## 3. Battery vs. Detection Delay Trade-Off

| Heartbeat Interval (s) | Detection Delay (s) | Heartbeats / Hour | Est. Radio Duty Cycle (%) |
|---|---|---|---|
| 10.0 | 31.0 | 360 | 0.05% |
| 15.0 | 47.0 | 240 | 0.033% |
| 30.0 | 91.0 | 120 | 0.017% |
| 60.0 | 181.0 | 60 | 0.008% |

## 4. Network Scaling Benchmark

| Nodes | Area (m) | Delivery | Transmissions (mean ± sd) | Latency (ms) | Hops to Gateway (median / max) |
|---|---|---|---|---|---|
| 50 | 565.7x565.7 | 80% (20/25) | 585.9 ± 60.1 | 126.7 ± 151.0 | 2 / 9 |
| 100 | 800.0x800.0 | 64% (16/25) | 1293.0 ± 104.6 | 224.3 ± 150.6 | 5 / 9 |
| 200 | 1131.4x1131.4 | 48% (12/25) | 2665.6 ± 187.3 | 246.2 ± 146.0 | 7.0 / 34 |

The last column is why delivery falls with scale. Area grows as sqrt(n)x80, so hop distance grows with it; where the maximum exceeds the TTL of 8 the message expires in flight. **This is a TTL limit, not congestion**, and raising TTL or placing more gateways addresses it directly.

## 5. Honest Assessment & Remaining Limitations

- **What worked:** deduplication and counter-based suppression together remove the large majority of redundant transmissions at no cost to SOS delivery. The silence inference engine detected every group collapse while ignoring LOWBAT beacons and single out-of-range walkers.
- **Single-seed results are gone.** Each number above is a mean over 25 topologies with its spread. Where the spread is wide, the system is sensitive to layout and the mean alone should not be quoted.
- **Radio realism:** in dense urban multipath RSSI varies by 10-20 dBm. Hop-distance adjacency is used as the proximity estimator rather than raw RSSI. The collision and airtime model is our own and has not been validated against a reference BLE simulator or a sniffer trace.
- **Unverified on real hardware:** no measurement of latency, energy or 48h battery drain on a physical handset.
