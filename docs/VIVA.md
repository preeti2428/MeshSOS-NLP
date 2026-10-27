# Viva sheet — what to show, question by question

Each section has the honest answer and the exact command. Run
`python show_everything.py` to walk the whole chain in one go.

---

## "Training me kaun sa dataset use kiya?"

**Answer, and say the second sentence too:**

> Training ke liye **6,000 synthetic messages** use kiye, jo
> `nlp/synth.py` templates se banate hain. **Lekin jo encoder app me
> ship hota hai wo trained hai hi nahi — wo rule-based hai.** Humne
> model train kiya, usse rules se compare kiya, aur model haar gaya.
> Wahi hamara ek result hai.

```bash
python nlp/train_model.py        # ~10 seconds, trains live
```

It prints this, and the last two rows are the point:

| | model | rules |
|---|---|---|
| on its own generated split | 99.7% | — |
| on hand-written messages | **28.0%** | **72.0%** |
| model size | 1435 KB | **0 KB** |
| per message | 0.788 ms | **0.083 ms** |

A model that scores 99.7% on data from its own generator and 28% on
text a person wrote has learned the generator, not the task. That is
why the phone ships rules.

---

## "Saara data dikhao"

Six datasets. Three we made, three are real.

| Dataset | Size | Made by | Used for | File |
|---|---:|---|---|---|
| Synthetic | 6,000 | us (`nlp/synth.py`) | **training** | generated on run |
| Stratified test | 10,000 | us (`nlp/testset.py`) | testing by failure type | generated on run |
| Hand-written | 25 | us | smoke test | `nlp/sample_messages.py` |
| **Disaster Response Messages** | **10,398** | Figure Eight / Appen | **real evaluation** | `data/raw/` |
| **Elicited Hindi** | **547** | 56 volunteers | code-mixed evaluation | `data/hindi/` |
| HumAID | 76,484 | QCRI | **rejected, with proof** | `data/annotated/humaid_rejected/` |

```bash
python nlp/compare_corpora.py    # why HumAID was rejected
```

The real one: **Disaster Response Messages**, `direct` genre — actual SMS
sent by people inside the 2010 Haiti earthquake. Downloaded from two
independent mirrors and checked byte-identical, so nobody modified it.

**If the teacher asks why not a standard tweet dataset:** we measured it.
HumAID carries 12.3% first-person appeals; DRM carries 41.6%. Tweets are
written by observers, our system carries messages from participants.

---

## "NLP kahan use kiya?"

In `nlp/`. The job is to turn free text into **3 bytes** so an emergency
report fits one Bluetooth frame.

```bash
python demo.py                   # one message, end to end
```

The chain:

```
"3 log malbe me fanse hain, ek behosh"
        ↓   nlp/rules.py          lexicon match, negation, number binding
  category=TRAPPED casualty=3 mobility=UNRESPONSIVE severity=SERIOUS
        ↓   nlp/envelope.py       bit-pack into 24 bits
  3 bytes
        ↓   protocol/compact_sos.py
  25-byte authenticated message → 1 frame instead of 6
        ↓   nlp/decoder.py        at the gateway
  readable text + a triage score
```

NLP techniques actually used: lexicon matching over English and
romanised Hindi together, negation detection on **both** sides (English
negates before — "no injuries"; Hindi after — "ghayal nahi"), numeric
extraction bound to a person word within 28 characters so "4 people"
counts and "4th floor" does not, and a confidence score that decides
whether to compress at all.

---

## "Training kahan kiya, dikhao"

Two places:

```bash
python nlp/train_model.py        # runs on this laptop, ~10 s
```

and `notebooks/MeshSOS_training.ipynb` in Google Colab, which shows the
loss curve falling 1.78 → 0.003 over 30 epochs. Open that if they want
to see training visually.

**Hyperparameters** (`nlp/train_model.py`):

| | |
|---|---|
| Features | TF-IDF, character n-grams, 2–4 characters |
| Vocabulary | 4,000 |
| Weighting | sublinear term frequency |
| Classifier | Logistic regression, `max_iter=1000`, `C=4.0` |
| Heads | one per field, plus one per hazard bit |

**Why character n-grams and not words:** people spell romanised Hindi
differently every time — *paani*, *pani*, *panee*. Character pieces
survive that; whole words do not.

---

## "Results kya mile?"

```bash
python nlp/derived_gold.py test  # real messages, held-out half
python nlp/eval_hindi.py test    # code-mixed Hindi
python nlp/eval_encoders.py 800  # by failure type
```

On **1,524 real held-out messages** with labels from Figure Eight's own
human annotators:

| | |
|---|---|
| Category correct | 47.7% |
| Compressed | 21.7% |
| **Of what it compressed, correct** | **81.9%** |
| Refused non-emergencies | 99.3% |

The third row is the one that matters. The encoder does not guess: when
it is unsure it sends the full text instead, and that is why what does
get compressed is right four times in five.

On **code-mixed Hindi**, the honest figure is **29.6%** — the three
scenarios whose question wording did not leak lexicon words. The other
seven scored 60.2%, but we had shown volunteers those words in the
question, so that number is our form's fault, not the encoder's merit.

---

## "Kya kya use kiya?"

| Part | Technology |
|---|---|
| Phone app | Kotlin, Android SDK 34, min Android 8 |
| Radio | Bluetooth LE legacy advertising, 31-byte frame |
| On-device NLP | **pure Kotlin, no model file, no internet** |
| Research models | scikit-learn — TF-IDF, logistic regression, MLP |
| Security | HMAC-SHA256 truncated to 4 bytes |
| Server | Python, FastAPI, SQLite |
| Dashboard | HTML, JavaScript, Leaflet |
| Simulator | Python, discrete-event, size-aware |
| Build | GitHub Actions → APK on every push |
| Tests | pytest, 37 passing |

---

## Base papers, their limits, and what we added

Thirteen references; three are the ones a comparison question will be
about.

### ResQMesh (SSRN 2026) — the closest work

On-device ML in a BLE mesh messenger: ~90 keyword patterns plus an INT8
Conv1D TensorFlow Lite classifier, 9 categories, 4 urgency tiers,
macro-F1 0.920 on 405 held-out messages at 0.19–1.97 ms.

**Its limit:** it uses the category to **order** messages in a priority
queue. The message itself is unchanged. A report too large for one frame
is still too large after classification, and no payload size, MTU or
fragmentation behaviour is reported; the latency gain is measured at the
queue, not over the air.

**Ours:** the category **resizes the payload**, and we measure what that
does at the network layer.

> **Say this before you are asked.** The full text is behind the SSRN
> paywall; our description comes from the abstract and metadata. Do not
> pretend otherwise — a single follow-up question will expose it and the
> rest of the viva will be spent recovering.
>
> *"Sir, ek baat pehle hi bata deti hun — ResQMesh ka full text paywall
> ke peeche hai, maine abstract aur metadata se kaam kiya. Par mera
> comparison full text pe depend nahi karta, kyunki fark structural hai.
> Wo category se messages ko queue me aage lagate hain; mai category se
> message ka size badalti hun — 161 bytes se 25, 6 frames se 1."*
>
> Then the point that settles it:
>
> *"Aur maine naapa hai ki prioritisation jo theek NAHI kar sakti wo kya
> hai. Queue me aage lagane se ye nahi badalta ki 6 me se saare 6
> fragments pahunchne chahiye. 200 nodes pe, ek frame bनाम saat — +31.8%
> zyada reports pahunchti hain. Ye fayda size se aata hai,
> prioritisation se nahi."*
>
> If pushed — *"to aapne unka paper padha hi nahi?"* — name exactly what
> you would check if you had it: their payload size, whether they report
> MTU or fragmentation, and whether latency was measured at the queue or
> over the air. Add that if they measured over the air, the related work
> needs revising. That answer reads as a careful researcher. A bluff
> reads as a student who did not read.

### OEPB (IETF draft-sharma-oepb-00, 2026)

A specification for authenticated offline emergency broadcast.

**Its limit:** a 256-byte message, 104 bytes of signed overhead alone.
On BLE 4.x the draft states that the 31-byte advertisement **requires**
fragmentation and reassembly — it accepts fragmentation as normal.

**Ours:** we asked whether fragmentation can be avoided rather than
accommodated.

### RelayZero (technical report, 2026)

Proposes the same idea we do: semantic compression for a disaster mesh.

**Its limit:** no evaluation, no radio (tested over a TCP socket), and
figures that contradict each other.

**Ours:** real BLE constraints, a size-aware simulator, 500 paired
trials per cell and significance tests.

### The four gaps, and the one that matters

| Gap in the literature | What we did |
|---|---|
| Nobody targets the frame boundary | 25 bytes, one frame |
| Nobody measures the network effect | size-aware simulator, +31.8% at 200 nodes |
| **Nobody confronts the crypto budget** | **96 of 124 bytes are signature and key** |
| Corpora collect observers, not participants | measured: 12.3% vs 41.6% first-person appeals |

The third is the one to lead with. Everyone assumed the text was the
problem. Measured, an SOS with **zero characters of text** is still 124
bytes, of which 96 are the Ed25519 signature and public key. Compressing
text alone could never have reached one frame — the security had to
shrink too. That finding is in no prior paper.

### What we have that the closest work does not

| | ResQMesh (from its abstract) | This work |
|---|---|---|
| Payload size | not reported | 161 B → **25 B** |
| Frames per message | not mentioned | 6 → **1** |
| Network delivery | not measured | **+31.8%** at 200 nodes, p<0.001 |
| Real-message evaluation | 405 messages | **3,185** real + **547** Hindi |
| Crypto budget | — | **96 of 124 bytes** |

---

## Questions they may push on

**"Model train kiya to use kyun nahi kiya?"**
Kyunki wo hand-written text pe 28% deta hai aur rules 72%. Aur rules 0 KB
ke hain, model 1435 KB. Paper me dono numbers hain.

**"Accuracy itni kam kyun hai — 47%?"**
Kyunki ye asli messages pe hai, apne banaye hue data pe nahi. Apne data pe
79% aata hai. Asli number chhota hai aur wahi sach hai. Aur jo system
compress karta hai usme 81.9% sahi hai — baaki pe wo poora text bhejta
hai, galat guess nahi karta.

**"Real data pe kyun nahi train kiya?"**
Kyunki real data pe humare fields ke labels nahi hain. Do log 500 messages
label kar rahe hain; wo kaam chal raha hai. Abhi jo gold use kiya wo
Figure Eight ke apne human labels se banaya hai, aur mapping poori likhi
hui hai.

**"Novelty kya hai?"**
Ek authenticated emergency report **ek hi 27-byte frame** me. Aur ye
finding ki 124 bytes me se **96 bytes cryptography** hai — text compress
karne se kabhi ek frame tak nahi pahuncha ja sakta. Ye kisi paper me
nahi hai.
