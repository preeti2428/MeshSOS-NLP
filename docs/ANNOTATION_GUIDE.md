# Annotation guide — MeshSOS semantic envelope

Two people label the same messages independently, then disagreements are
discussed and settled. Agreement is reported as Cohen's kappa per field.
**Do not label together.** Agreement measured after discussion means nothing.

Label what the message **says**, not what you infer about the situation.
If a field is not stated, mark it unknown. Guessing inflates agreement and
teaches the encoder to guess too.

---

## category — one value

Pick the **most urgent** that the message states. Order when several apply:

`TRAPPED > COLLAPSE > MEDICAL > FIRE > FLOOD > EVACUATION > SUPPLIES_NEEDED > RESCUE_NEEDED > OTHER`

| Value | Use when the message says |
|---|---|
| `TRAPPED` | people cannot get out — pinned, buried, enclosed |
| `COLLAPSE` | a structure has come down (nobody stated as trapped) |
| `MEDICAL` | injury, illness, someone needs medical help |
| `FIRE` | fire or smoke is the emergency |
| `FLOOD` | water is the emergency |
| `EVACUATION` | people are leaving or need to leave |
| `SUPPLIES_NEEDED` | food, water, medicine, shelter needed |
| `RESCUE_NEEDED` | help is asked for, with no specific emergency named |
| `OTHER` | an emergency message that fits none of the above |

Not an emergency report at all (news, sympathy, donation appeals, commentary)
→ leave the whole row blank and write `SKIP` in `notes`.

## casualty_count — 0 to 7

Number of **people at risk**, as stated. `7` means seven or more.
`0` means **not stated** — not "nobody".

- "4 people trapped" → `4`
- "my family is stuck" → `0` (no number given)
- "dozens are missing" → `7`
- "I am trapped" → `1`

Count people in danger, not rescuers, not the dead unless they are the subject
of the report.

## mobility — one value

| Value | Use when |
|---|---|
| `UNKNOWN` | not stated |
| `AMBULATORY` | they can move or walk |
| `TRAPPED` | they cannot move |
| `UNRESPONSIVE` | someone is unconscious or not responding |

If several people differ, record the **worst** (unresponsive beats trapped
beats ambulatory).

## hazard_flags — zero or more

Tick every hazard the message states. A category often implies one: a fire
report has `FIRE`, a flood report has `WATER`, a collapse has `STRUCTURAL`.

`FIRE` · `WATER` · `GAS` · `STRUCTURAL` · `ELECTRICAL` · `CHEMICAL`

Write them comma-separated, e.g. `WATER,STRUCTURAL`. None → leave blank.

## severity — one value

| Value | Use when |
|---|---|
| `UNKNOWN` | injuries not mentioned |
| `NONE` | explicitly says nobody hurt |
| `MINOR` | minor injuries |
| `SERIOUS` | heavy bleeding, unconscious, critical, life-threatening |

`UNKNOWN` and `NONE` are different. "We are trapped" is `UNKNOWN`.
"We are trapped but nobody is hurt" is `NONE`.

---

## Hard cases, decided in advance

**Retweets and news reports.** HumAID contains many. If the message reports
*about* an event rather than *from* it ("Death toll rises to 233"), it is
still labelled — the encoder will see such text — but note `THIRD_PARTY` in
`notes` so these can be analysed separately.

**Mixed messages.** "No food, and water is rising" — water is the more urgent
emergency, so `FLOOD`, and `SUPPLIES_NEEDED` is lost. That loss is real and
the paper should report it.

**Numbers that are not people.** "4th floor", "2 buildings" → `casualty_count`
is `0`.

**Sarcasm, jokes, abuse.** → `SKIP`.

**Unsure.** Label your best reading and write why in `notes`. Do not leave a
field blank hoping the other annotator fills it.

---

## After labelling

```bash
python nlp/score_annotations.py data/annotated/annotator_A.csv data/annotated/annotator_B.csv
```

It prints Cohen's kappa per field and writes the disagreements to a file to
work through. Kappa below 0.6 on a field means the guide is unclear for that
field, not that an annotator was careless — fix the guide and relabel.
