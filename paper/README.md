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

**Fill in the authors.** The file carries the template's six placeholder
blocks (`1st Given Name Surname`, …). Replace them with real names,
departments, institutions, cities and emails, and delete any blocks you do
not need — the class takes one to six authors.

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
