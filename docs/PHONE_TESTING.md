# Testing MeshSOS on real handsets

## Getting the APK

Every push to `main` builds a debug APK and attaches it to a rolling
pre-release. The link is public and always serves the current build:

```
https://github.com/preeti2428/MeshSOS-NLP/releases/latest
```

Send that link to each tester. Tap `MeshSOS-latest.apk` on the phone.

The build is **unsigned debug**. Android will warn. Each phone needs
*Install unknown apps* allowed for whichever app opened the link
(usually Chrome). Settings → Apps → Chrome → Install unknown apps.

Minimum Android 8. Bluetooth Low Energy required, which is every phone
from about 2015 onward.

## Permissions each phone must grant

| Permission | Why | Android |
|---|---|---|
| Nearby devices | BLE advertise and scan | 12+ |
| Location | BLE scanning is gated behind it | 8–11 |
| Notifications | the foreground service notice | 13+ |

On Android 12 and newer the phone asks for *Nearby devices*, not
location. If scanning silently finds nothing, this permission is almost
always why.

Also turn **battery optimisation off** for MeshSOS on every phone
(Settings → Apps → MeshSOS → Battery → Unrestricted). Otherwise the
system suspends the foreground service after a few minutes and the
phone stops relaying — which looks exactly like a protocol bug.

## What to actually measure

Ten phones is enough to test things the simulator cannot. Run these in
order and write the numbers down; they go straight into the paper's
hardware section, which is currently empty.

**1. Encoder latency on real hardware.** Type each of 10 messages into
the SOS box and watch the preview line. Every number in the paper
(0.081 ms) is desktop Python. Use the oldest phone you have — that is
the number worth reporting, not the fastest.

**2. Does the preview agree with the desktop encoder?** Type the same
message on the phone and run it through `python nlp/rules.py` on the
laptop. The Kotlin and Python encoders are meant to be identical. Any
disagreement is a real bug and worth finding before the demo.

**3. One hop.** Two phones, a few metres apart, aeroplane mode on both
so there is no internet anywhere. Send an SOS from one. Does the other
receive it? Time it.

**4. Multi-hop.** Spread all ten phones down a corridor, far enough
apart that each only reaches its neighbours. Send from one end. Does it
arrive at the other? How many hops, how long? This is the headline
demo.

**5. Range.** Two phones, walk apart until delivery stops. Record the
distance. The simulator assumes 110 m; the real number indoors will be
much lower and that is worth knowing.

**6. Compact versus full text.** Send one message the encoder
understands (green preview) and one it does not (orange). Both should
arrive. The compact one should arrive sooner. That difference is the
entire thesis, visible on a table.

**7. Battery.** Leave one phone running the service for an hour with
the screen off. Note the percentage before and after.

## Expect the compact path to be rare

On the Figure Eight corpus the encoder reaches its confidence threshold
on 13.6% of real messages (`python nlp/derived_gold.py test`). Most of
what a tester types will fall back to full text, and that is the
measured behaviour, not a fault in the build. To demonstrate the compact
path deliberately, type something that names the emergency plainly:

```
3 log malbe me fanse hain, ek behosh
building collapsed 4 people trapped inside
flood water rising 5 of us stuck on roof
```

Watch the preview line turn green before sending.

## If nothing is received

In order of how often it is the cause:

1. Nearby devices / location permission not granted
2. Battery optimisation still on, service suspended
3. Bluetooth off, or the phone is in battery saver
4. Both phones too far apart indoors — walls cut BLE hard
5. Android 8 or 9 phones advertise less reliably; try two newer ones first
