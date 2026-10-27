# Paper

`meshsos_ieee.tex` — the paper in IEEE conference format.

## Compiling it

IEEEtran is not installed in most local TeX setups, so use Overleaf:

1. [overleaf.com](https://www.overleaf.com) → **New Project** → **Upload Project**
2. Upload `meshsos_ieee.tex`
3. It compiles as-is. IEEEtran ships with Overleaf.

Alternatively start from Overleaf's own *IEEE Conference Template* and paste
the body in.

Locally it needs `texlive-publishers` (Debian/Ubuntu) or the `IEEEtran`
package via `tlmgr`.

## Before submitting

**Authors are filled in** — the five from `draft_1`, all Dept. of CSE
(AI&ML), KIET Deemed to be University, Ghaziabad. Check the order: IEEE
reads left to right, then down, and that is the sequence used in citations
and indexing.

**Cite the team's other paper.** *ResQTrail: Lifecycle-Aware SOS
Dissemination over Smartphone Mesh Networks for Flood Rescue* is a separate
paper by the same authors on a related system. Two papers from one group on
smartphone mesh SOS that do not reference each other invites a duplicate-
submission question, and the answer is a one-line citation. The systems
differ in transport — ResQTrail uses Google Nearby Connections, this work
targets raw legacy BLE advertising — and saying so in a sentence turns an
awkward overlap into a clear division of scope.

**Add Figure 1.** `figures/fig1_delivery.pdf` is the delivery chart. Insert it
in §V-F with:

```latex
\begin{figure}[htbp]
\centerline{\includegraphics[width=\columnwidth]{figures/fig1_delivery.pdf}}
\caption{SOS delivery by condition and node count.}
\label{fig:delivery}
\end{figure}
```

**Check the references.** Several entries are preprints or grey literature and
are marked as such in the text. Verify each one and complete the fields your
venue requires. [3] (ResQMesh) and [7] (OEPB) are the two closest pieces of
prior work and the two a reviewer is most likely to know.

**Read it before you submit it.** Every number here came from this repository
and can be regenerated with the commands in `RESULTS.md`, but you will be
asked where each one came from.

## preview_NOT_ieee_format.pdf

A syntax check only, compiled with `article` because IEEEtran was unavailable
in the build environment. Use it to judge length and content, not layout —
the real formatting comes from Overleaf.
