# MeshSOS — Semantic Compression: Results

Measured on this repository, 27 October 2026. Every number below is
reproducible with the commands in the last section.

---

## 1. What was built

A free-text emergency report is compressed on the handset into a 3-byte
semantic envelope and sent as a 25-byte authenticated frame — small enough
for one legacy BLE advertising packet. When the encoder is not confident, it
sends the original text instead.

| Component | File |
|---|---|
| 3-byte semantic envelope | `nlp/envelope.py` |
| Rule-based encoder, no training | `nlp/rules.py` |
| Envelope → rescuer text, triage score | `nlp/decoder.py` |
| 25-byte authenticated SOS | `protocol/compact_sos.py` |
| Size-aware mesh simulator | `simulator/network_simulator.py` |
| Evaluation harness | `simulator/eval_compression.py` |

---

## 2. Message size

The existing `SosMessage` is 124 bytes before a single character of text,
because it carries an Ed25519 public key (32 B) and signature (64 B) on the
wire. **96 of those 124 bytes are cryptography.** Compressing only the text
takes a typical report from 6 frames to 5; it can never reach 1.

Reaching one frame required three trades, each stated in
`protocol/compact_sos.py`:

| Change | Saved |
|---|---|
| Public key resolved from `node_id` at the gateway, not sent | 32 B |
| 64-byte Ed25519 signature → 4-byte truncated HMAC | 60 B |
| Float lat/lon → the ~100 m grid cell the heartbeat already uses | 4 B |

Result:

| | Bytes | Frames (27 B each) |
|---|---|---|
| Full `SosMessage` with text | 162–187 | 6–7 |
| `CompactSosMessage` | **25** | **1** |
| Reduction | **~86%** | 6–7 → 1 |

The 4-byte tag is **not** a digital signature. It authenticates to a gateway
holding the shared key and does not prove origin to a third party. A 32-bit
tag also means a 1-in-4-billion forgery chance per attempt. Both are
acceptable only because the full signed message follows; neither should be
described as equivalent to Ed25519.

---

## 3. Encoder fidelity

25 hand-written messages (`nlp/sample_messages.py`), English and romanised
Hindi. **These are not a benchmark dataset** — they were written by the
authors to exercise the encoder. A real evaluation needs HumAID /
CrisisBench with human-annotated envelope fields, which is not done.

| Field | Correct |
|---|---|
| casualty_count | 25/25 — 100.0% |
| mobility | 24/25 — 96.0% |
| severity | 24/25 — 96.0% |
| category | 23/25 — 92.0% |
| hazard_flags | 23/25 — 92.0% |
| **all fields exact** | **19/25 — 76.0%** |
| fell back to full text | 7/25 — 28.0% |

No training data, no model file, no accelerator. This is the baseline any
learned model has to beat, and it has not been beaten yet because the
learned model does not exist.

### 3a. Where the encoder breaks: 10,000 messages, stratified

25 messages cannot show where an encoder fails. This set has 10,000, split
by how far the wording sits from what the lexicon knows, with labels true
by construction (`nlp/testset.py`). Still generated text, so read it as a
coverage diagnostic, not a benchmark score.

| Stratum | Category | Count | Mobility | Hazard | Severity | **All fields** | Fallback |
|---|---|---|---|---|---|---|---|
| known_vocab | 79.8% | 100.0% | 90.7% | 83.7% | 85.7% | **58.6%** | 16.2% |
| **synonyms (OOV)** | **8.3%** | 60.3% | 60.8% | 31.8% | 48.4% | **0.1%** | **97.0%** |
| spelling | 69.8% | 100.0% | 92.4% | 77.5% | 86.2% | **50.9%** | 25.2% |
| code_mixed | 79.9% | 100.0% | 91.2% | 85.7% | 84.3% | **60.4%** | 17.4% |
| unintelligible | — | — | — | — | — | — | **100.0%** |

2,000 messages per stratum, 0.100 ms each, 0 KB model.

Four things this says that a single number cannot:

**The out-of-vocabulary cliff is total, and it is safe.** On synonyms the
encoder recovers the category 8.3% of the time — barely above nothing. But
it does not guess: it refuses to compress on **97.0%** of them and sends
the original text. The failure mode is slower transmission, not a wrong
report.

**Spelling costs about eight points, not the system.** Real typing
("pani bhr rha hai", "mai fnsa hu") drops all-fields accuracy from 58.6%
to 50.9% and raises fallback from 16.2% to 25.2%. Worth fixing with fuzzy
matching; not a crisis.

**Code-mixed input is not the weak point.** At 60.4% it edges out pure
known-vocabulary text. Hindi–English mixing was designed for from the
start, and the lexicons carry both languages together.

**Refusal on unintelligible input is perfect.** 100% of 2,000 gibberish
messages fell back rather than inventing a category.

Counts are the strongest field (100% where the vocabulary is known) and
hazards the weakest (83.7%), because hazards co-occur and a message often
implies one the writer never named.

**The honest reading:** this encoder is a lexicon. It performs where the
lexicon reaches and refuses where it does not. That is defensible for
deployment and insufficient as a research result — closing the synonyms
column is the work that matters, and section 3b explains why a learned
model is not yet the answer.

---

## 3b. A learned encoder, and why it is not the one shipping

HumAID and CrisisBench were not reachable from the build environment
(`crisisnlp.qcri.org` does not resolve here), so there is no real annotated
corpus. Instead a template generator (`nlp/synth.py`) produced 6,000
messages whose field values are known by construction — ground truth, not
rule output, so the model is not merely imitating the baseline.

A character n-gram TF-IDF model with one logistic-regression head per field
was trained on it (`nlp/train_model.py`) and tested twice.

**On held-out synthetic data** — same generator, unseen samples:

| Field | Model | Rules |
|---|---|---|
| category | 100.0% | 37.4% |
| casualty_count | 100.0% | 88.2% |
| mobility | 100.0% | 77.1% |
| severity | 99.7% | 77.1% |
| hazard_flags | 100.0% | 64.8% |
| **all fields** | **99.7%** | **10.3%** |

**On the 25 hand-written messages** — never templated, out of distribution:

| Field | Model | Rules |
|---|---|---|
| category | 44.0% | 92.0% |
| casualty_count | 92.0% | 100.0% |
| mobility | 68.0% | 96.0% |
| severity | 76.0% | 96.0% |
| hazard_flags | 68.0% | 92.0% |
| **all fields** | **28.0%** | **76.0%** |

The model learned the generator, not the task. It wins by 10x on data that
looks like its training set and loses by nearly 3x on text that does not.

Cost makes the gap worse:

| | Size | Latency |
|---|---|---|
| Model, fp32 | 1,435 KB | 0.83 ms/message |
| Model, int8 | 808 KB | — |
| Rules | **0 KB** | **0.10 ms/message** |

**The rules encoder is what ships**, and it is what the Android app uses.
The learned model is reported because the comparison is the finding: a
model trained on synthetic data is worse than keyword matching on real
text, while appearing excellent on its own held-out split. Quoting only the
synthetic column would have been a false result.

What this does **not** show is that learned encoders are a dead end — only
that this one, trained on this data, is. That question needs HumAID /
CrisisBench with annotated envelope fields.

### 3c. Cascade: the two encoders fail in opposite places

Scored per stratum, the comparison is not "the model lost". On the
out-of-vocabulary column the model wins by roughly five times, because
character n-grams reach wording no keyword list contains.

So route between them on the confidence the rule encoder already produces:
rules when confident, model when not. Nothing extra is trained.

| Stratum | Rules | Neural | **Cascade** |
|---|---|---|---|
| known_vocab | 79.8% | 62.3% | **81.2%** |
| **synonyms (OOV)** | 8.3% | 39.1% | **38.8%** |
| spelling | 69.8% | 61.7% | **74.8%** |
| code_mixed | **79.9%** | 50.5% | 75.8% |
| **Overall** | 59.5% | 53.4% | **67.7%** |

Category accuracy, 8,000 messages (the unintelligible stratum is excluded:
its gold is OTHER and both encoders answer OTHER, which would inflate every
figure). Unintelligible input is still refused 100% of the time, so the
safety property does not depend on which encoder answered.

The cascade is not implemented in the shipped app — the Android encoder is
rules only. It is reported because it is measured, and because it says what
the next engineering step is.

---

## 4. Network effect

60 trials per cell, pairing seed and message round-robin. TTL 8, 3%
per-frame loss, 27-byte frames.

**A** full text · **B** always compressed · **C** compressed when confident

| Nodes | A | B | C | C vs A | Airtime saved (C) |
|---|---|---|---|---|---|
| 50 | 80.0% | 88.3% | 86.7% | **+8.3%** | 15.7% |
| 100 | 76.7% | 85.0% | 80.0% | **+4.3%** | 12.0% |
| 200 | 53.3% | 60.0% | 60.0% | **+12.5%** | 8.5% |

![Delivery by condition](figures/fig1_delivery.png)

The gain is largest at 200 nodes, where the mesh is most loaded. **C is the
shipping configuration and the number to quote.** B is the ceiling — it
assumes the encoder is never unsure, which is not true 23% of the time.

---

## 5. The safety case

A compression failure here is not a metric, it is a person. Reading "2
children trapped" as "1 adult trapped" sends the wrong team.

Three things guard against it:

1. **The encoder reports its own confidence**, and it travels in the
   envelope, so the rescuer's screen shows what was inferred.
2. **Below threshold, nothing is compressed** — the original text is sent.
   This fired on 7 of 25 messages, all correctly.
3. **A regression test asserts counts are never lowered**
   (`test_casualty_count_is_not_silently_lowered`).

`python demo.py` ends on a message the encoder refuses to compress. That
refusal is the feature.

---

## 6. What is not done

- **The learned model is trained on synthetic data** and loses badly to the
  rule baseline on real text (section 3b). Nothing here shows a learned
  encoder is better — only that this one is not.
- **No real dataset.** 25 hand-written messages are a smoke test, not
  evidence. HumAID / CrisisBench with annotated fields is the next step.
- **No hardware.** Simulated throughout. Nothing has run on a handset, and
  no battery measurement exists.
- **Simplified radio.** Collisions are modelled as overlapping frames at one
  receiver; no multipath, capture effect, or hardware-specific behaviour.
- **The original 200-node "0 delivered" was a single unlucky seed** (seed 42
  puts the gateway 17 hops away; the median across seeds is 6). The old
  `docs/EVALUATION.md` numbers are single-seed and must be re-run.

---

## 7. Reproducing

```bash
python demo.py                                   # end-to-end walkthrough
python -m pytest tests/test_nlp_compact.py -q     # 14 invariants
python simulator/test_phase0_size_aware.py        # size-awareness check
python simulator/eval_compression.py 60           # the table in section 4
python simulator/make_figure.py                   # the figure
python simulator/ttl_vs_size_experiment.py 25     # TTL vs size separation
```
