# Choosing the evaluation corpus

This file records a decision that cost a day and would have cost a week if
it had been found later: **HumAID is the wrong corpus for this system**, and
the Figure Eight *Disaster Response Messages* corpus is the right one.

## The question

MeshSOS compresses messages written by people who are **inside** an
emergency and are asking for help. To measure the encoder honestly, the
test messages must come from that population.

## What we tried first

HumAID (76,484 labelled disaster tweets) is the standard choice and is easy
to obtain. We drew 520 messages and built annotation sheets for them. Before
spending two people × 8 hours on labelling, we read a sample. Almost none of
it was a person asking for help:

> It breaks my heart to see my country like this. More than 60 dead …
> RT @pootan_viking: BBC News — California wildfires: More than 1,000 missing
> Death toll keeps rising please donate if you can, anything helps

## The measurement

`nlp/compare_corpora.py` applies one identical set of counters to both
corpora. A "first-person appeal" is a message containing both a first-person
pronoun and a need or help word — a crude proxy, but the same crude proxy
on both sides, so the comparison is fair.

| | HumAID (520) | DRM `direct` (10,782) |
|---|---:|---:|
| First-person pronoun | 21.9% | **70.9%** |
| Need / help word | 27.1% | **47.4%** |
| **Both — a first-person appeal** | **12.3%** | **41.6%** |
| Retweet | 17.1% | 0.0% |
| Contains a hashtag | 54.0% | 0.7% |
| Median length | 116 chars | **75 chars** |

DRM `direct` contains **3.4× more first-person appeals**, no retweets,
effectively no hashtags, and messages a third shorter — much closer to what
a 27-byte radio frame has to carry.

## What we chose

**Disaster Response Messages (Figure Eight / Appen), `genre == "direct"`.**

These are SMS and web-form messages sent during the 2010 Haiti earthquake,
the 2010 Pakistan floods and the 2012 storms, by people inside the event,
translated into English.

```
we need TENTE, food, everything. .. .. .. .. ...
I need food and care I lived in Delma my house was broken everyone died
We have 401 destroyed and damaged houses, 1987 people temporarily homeless
please we need water, food and tents, we have 101 children .. please come rescue
```

After cleaning (330 translator notes, 44 duplicates, 6 empty or very short
rows removed) the pool is **10,402** messages.

## Two limitations we state rather than hide

1. **The messages are translations.** They were written in Haitian Creole,
   French or Urdu. The grammar is often broken and some words are
   mistranslated. This is a real limitation of the evaluation, but broken
   grammar is also what a real SOS looks like, so it is not purely a loss.

2. **There is no Hindi.** Every `original` field in the corpus is Latin
   script — Creole and French, no Devanagari and no romanised Urdu. The
   code-mixed Hindi claim therefore cannot be tested on any public corpus,
   and is being evaluated on messages elicited from volunteers instead
   (see `data/hindi/README.md`).

## Why the HumAID sheets are kept

`data/annotated/humaid_rejected/` holds the original 520-message sheets.
They are kept so the measurement above can be reproduced, and because a
negative result that was found before the work was wasted is worth
reporting.
