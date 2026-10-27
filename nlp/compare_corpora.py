"""
Measure why a tweet corpus is the wrong population for a mesh SOS system.

MeshSOS carries messages written by people INSIDE an emergency asking for
help. Crisis tweet corpora (HumAID, CrisisBench) are written largely by
observers: news accounts, retweets, sympathy and commentary. This script
measures that difference on three sets with one identical set of counters,
so the choice of corpus rests on numbers rather than on an impression.

Headline result (see docs/CORPUS_CHOICE.md):
  first-person appeals   HumAID 12.3%   vs   DRM direct 41.6%   (3.4x)
  retweets               HumAID 17.1%   vs   DRM direct  0.0%
  hashtags               HumAID 54.0%   vs   DRM direct  0.7%

Run:  python nlp/compare_corpora.py
"""

import csv, re, sys
csv.field_size_limit(10**7)

FIRST = re.compile(r"\b(i|we|my|our|me|us|i'm|we're|im)\b", re.I)
NEED  = re.compile(r"\b(need|needs|help|please|send|stuck|trapped|rescue|save|"
                   r"hungry|thirsty|nothing to eat|no food|no water|give us|"
                   r"we are|i am|assist|aid us|come)\b", re.I)
RT    = re.compile(r"^\s*RT\s+@", re.I)
URL   = re.compile(r"https?://|www\.")
HASH  = re.compile(r"#\w+")

def stats(name, texts):
    n = len(texts)
    fp   = sum(1 for t in texts if FIRST.search(t))
    need = sum(1 for t in texts if NEED.search(t))
    both = sum(1 for t in texts if FIRST.search(t) and NEED.search(t))
    rt   = sum(1 for t in texts if RT.match(t))
    url  = sum(1 for t in texts if URL.search(t))
    hsh  = sum(1 for t in texts if HASH.search(t))
    ln   = sorted(len(t) for t in texts)
    print(f"\n{name}  (n={n})")
    print(f"  first-person pronoun        {100*fp/n:5.1f}%")
    print(f"  need / help word            {100*need/n:5.1f}%")
    print(f"  BOTH  (first-person appeal) {100*both/n:5.1f}%   <-- the population we carry")
    print(f"  retweet                     {100*rt/n:5.1f}%")
    print(f"  contains a URL              {100*url/n:5.1f}%")
    print(f"  contains a hashtag          {100*hsh/n:5.1f}%")
    print(f"  median length               {ln[n//2]} chars")

h = [r['tweet_text'] for r in csv.DictReader(
        open('data/annotated/humaid_rejected/annotator_A.csv', encoding='utf-8'))]
stats("HumAID  (520 sampled tweets)", h)

d = [r['message'] for r in csv.DictReader(
        open('data/raw/disaster_messages.csv', encoding='utf-8')) if r['genre']=='direct']
stats("DRM direct  (all 10,782)", d)

s = [r['message'] for r in csv.DictReader(
        open('data/annotated/drm_annotator_A.csv', encoding='utf-8'))]
stats("DRM direct  (our 500-message sample)", s)
