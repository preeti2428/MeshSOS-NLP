# MeshSOS

An infrastructure-free emergency mesh network for smartphones, with
semantic compression that fits a complete authenticated distress report
into **one legacy BLE advertising frame**.

When cellular infrastructure fails, phones relay for one another over
Bluetooth Low Energy until a message reaches a device that still has
connectivity. The constraint is the advertising frame: 31 bytes, about 27
usable. Our SOS message was 124 bytes before a single character of text.

```
python demo.py
```

---

## What this does

```
handset text  ->  3-byte envelope  ->  25-byte authenticated frame
                                              |
                                        mesh relay (BLE)
                                              |
                        gateway  ->  expanded report + triage score
```

A free-text report is encoded on the handset into 3 bytes carrying
category, casualty count, mobility, hazard flags, severity and the
encoder's own confidence. **Below a confidence threshold nothing is
compressed** and the original text is sent instead - the encoder does not
guess at what it did not understand.

| | Bytes | Frames |
|---|---|---|
| Full SOS with text | 162-187 | 6-7 |
| Compact SOS | **25** | **1** |

Getting there needed more than compressing text: 96 of the old 124-byte
floor was cryptography. See [`RESULTS.md`](RESULTS.md) section 2 for the
three trades that made 27 bytes reachable, and what each gives up.

## Measured effect

60 trials per cell, TTL 8, 3% per-frame loss:

| Nodes | Full text | Compressed when confident |
|---|---|---|
| 50 | 80.0% | 86.7% |
| 100 | 76.7% | 80.0% |
| **200** | **53.3%** | **60.0%** |

![Delivery by condition](figures/fig1_delivery.png)

## The encoder is keyword matching, and that is the finding

A learned model trained on synthetic data scored **99.7%** on its own
held-out split and **28.0%** on hand-written messages, where the rule
baseline scored **76.0%** at zero model size and an eighth of the latency.
It learned the generator, not the task.

The rules encoder is what the Android app ships. Both columns are reported
in [`RESULTS.md`](RESULTS.md) section 3b, because quoting only the
synthetic one would have been a false result.

## Layout

| Path | What |
|---|---|
| `nlp/` | envelope, rule encoder, decoder, synthetic corpus, training |
| `protocol/` | wire format, including the 25-byte compact SOS |
| `simulator/` | size-aware discrete-event mesh simulator and experiments |
| `inference/` | silence detection, SOS hotspots, active coverage |
| `backend/` | FastAPI ingest and map layers |
| `dashboard/` | rescuer web interface |
| `android/` | Kotlin app, including the on-device encoder |
| `RESULTS.md` | every measured number, with its method |
| `docs/DECISIONS.md` | verified BLE and Android platform constraints |

## Running it

```bash
pip install -r requirements.txt

python demo.py                                 # end-to-end walkthrough
python -m pytest tests/ -q                     # invariants, byte budget included
python simulator/eval_compression.py 60        # the delivery table above
python simulator/make_figure.py                # the figure
python nlp/train_model.py                      # the learned-model comparison
```

Backend and dashboard:

```bash
./start.sh        # or: docker compose up
```

The Android APK is built by GitHub Actions on every push; download it from
the workflow run's artifacts.

## Status

Simulated throughout. Nothing has run on a handset, there is no battery
measurement, and fidelity is measured on 25 hand-written messages rather
than an annotated corpus. [`RESULTS.md`](RESULTS.md) section 6 lists every
limitation. The shared key in `MeshForegroundService` is a placeholder and
must not ship.
