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

## meshsos_preview.pdf

A rendering of the same source, compiled here with the `article` class
dressed to look like IEEEtran: letter paper, two columns, Times 10pt, IEEE
margins, Roman-numeral small-caps headings. IEEEtran itself is not installed
in the build environment and CTAN was unreachable, so this is an
approximation, not the real class.

Good enough to read, circulate and judge length. **Compile on Overleaf for
anything you submit** — spacing, float placement and the reference style will
shift slightly, and only IEEEtran output is actually IEEE format.
