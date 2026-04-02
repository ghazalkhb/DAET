#!/usr/bin/env python3
# Convert sn-article.tex (Springer Nature) -> daet-ist.tex (Elsevier CAS-SC)
import re

SRC  = r"d:\Naser\2\Journal Paper\journal_v2\sn-article.tex"
DST  = r"d:\Naser\2\IST Journal\daet-ist\daet-ist.tex"

# ------------------------------------------------------------------ #
# New preamble + front matter  (CAS-SC / IST style)                  #
# ------------------------------------------------------------------ #
PREAMBLE = r"""%% IST Journal (Information and Software Technology) -- Elsevier CAS Single-Column
%% Converted from: Journal Paper/journal_v2/sn-article.tex  (Springer Nature)
%% Target journal: Information and Software Technology (Elsevier)
%% Compile:  pdflatex -> bibtex -> pdflatex -> pdflatex

\PassOptionsToPackage{compatibility=false}{caption}
\documentclass[a4paper,fleqn]{cas-sc}

\usepackage[authoryear,longnamesfirst]{natbib}

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage{algorithm}
\usepackage[noend]{algpseudocode}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{url}

\algnewcommand\algorithmicforeach{\textbf{for each}}
\algdef{S}[FOR]{ForEach}[1]{\algorithmicforeach\ #1\ \algorithmicdo}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\raggedbottom
\emergencystretch=3em
\hbadness=10000
\vbadness=10000

%% ---------------------------------------------------------------
\begin{document}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{Adaptive Execution Tracing: Large-Scale Comparative Validation}
\shortauthors{M.A.~Khan et~al.}

\title[mode=title]{Towards Efficient Performance Debugging: A Dynamic and Adaptive
Execution Tracing Technique with Large-Scale Comparative Validation}

\author[1]{Mohammed Adib Khan}
\fnmark[1]
\ead{khanm@brocku.ca}

\author[1]{Morteza Noferesti}
\cormark[1]
\fnmark[1]
\ead{mnoferesti@brocku.ca}

\author[1]{Naser Ezzati-Jivan}
\fnmark[1]
\ead{nezzatijivan@brocku.ca}

\affiliation[1]{organization={Computer Science Department, Brock University},
               addressline={1812 Sir Isaac Brock Way},
               city={St.~Catharines},
               postcode={L2S-3A1},
               state={ON},
               country={Canada}}

\cortext[1]{Corresponding author}
\fntext[1]{These authors contributed equally to this work.}

% -------------------------------------------------------------------
\begin{abstract}
Performance debugging in production systems is hampered by high tracing overhead
and the challenge of determining \emph{when} to collect data.
This paper presents an extended evaluation of the dynamic adaptive execution
tracing (DAET) framework.
The framework operates in three phases:
(1)~Tracing-Function Selection via call-stack profiling and
coefficient-of-variation (CoV) ranking,
(2)~Change Detection based on anomaly scoring with a dynamic threshold, and
(3)~Trace Configuration Adjustment that activates tracepoints only when the
anomaly score exceeds a tolerance-based gate.

This journal extension provides a large-scale, statistically grounded evaluation
of multiple change-detection engines (ARIMA, LAST, SMA, EWMA, and a compact
Transformer) across two Mozilla performance-telemetry benchmarks under a unified
60\%/20\%/20\% chronological protocol, including dual-dataset validation,
replay-based deployment analysis, and a fair multi-seed Transformer comparison.

On Firefox-Android (136 of 342 test-alerted signatures),
SMA($w=40$) achieves the highest detection rate (72.8\%)
among classical methods, with EWMA($\alpha=0.05$) performing comparably,
while ARIMA outperforms LAST.
The compact Transformer (WINDOW=20, $d_{\text{model}}=16$) achieves
a detection rate of 74.4\%, competitive with the best classical methods
and stable across five seeds.
On mozilla-beta, SMA again leads (91.7\%), followed closely by EWMA and ARIMA,
while LAST remains weakest.

The qualitative ranking SMA~$\approx$~EWMA~$\geq$~ARIMA~$>$~LAST
is consistent across both datasets, providing preliminary evidence of
robustness within the Mozilla CI context; all metrics measure
\emph{agreement with Treeherder alerts} rather than ground-truth anomaly
validity, and precision remains below 0.12 across all methods.

All detectors reduce trace storage by 67--90\%, with classical methods achieving
higher reduction than the Transformer (67--71\%).
The ablation study shows that default parameters correspond to a high-recall
operating point, while alternative settings trade recall for more selective
detection.

These results indicate that simple statistical methods provide computationally
efficient baselines for adaptive tracing that are competitive with lightweight
learned models under fair evaluation; their utility for actual debugging
effectiveness requires further validation beyond alert-agreement metrics.
\end{abstract}

% Research highlights (required by IST / Elsevier)
\begin{highlights}
\item Large-scale comparative evaluation of five change detectors (ARIMA, LAST,
SMA, EWMA, Transformer) on 342 Mozilla Firefox-Android and 1{,}477 mozilla-beta
performance signatures under a fair 60\%/20\%/20\% chronological protocol with
an independent held-out validation set.
\item SMA ($w\!=\!40$) and EWMA ($\alpha\!=\!0.05$) achieve F1\,=\,0.104/0.103
on Firefox-Android and F1\,=\,0.085/0.083 on mozilla-beta, leading all classical
methods; the qualitative ranking
SMA\,$\approx$\,EWMA\,$\geq$\,ARIMA\,$>$\,LAST is consistent across both
datasets.
\item All four classical detectors reduce trace storage by 89--91\%; SMA achieves
the highest replay alert coverage (93.0\%) at zero training cost and $16\times$
lower memory than ARIMA.
\item Parameter ablation across $\beta$, $\theta$, $\tau$, ARIMA order, and SMA
window confirms stable high-recall operating regions; $\beta\!=\!0.20$ yields
ARIMA F1\,=\,0.266 vs.\ the default 0.144, providing actionable tuning guidance.
\item A compact multi-seed Transformer (five seeds, identical three-way protocol)
achieves F1\,=\,0.105\,$\pm$\,0.002 on Firefox-Android, confirming that learned
backbones match top classical methods on this Treeherder-agreement benchmark.
\end{highlights}

% Keywords -- each separated by \sep
\begin{keywords}
kernel tracing \sep application tracing \sep execution tracing \sep
adaptive performance monitoring \sep performance debugging \sep
anomaly detection \sep ARIMA \sep change detection \sep Mozilla telemetry
\end{keywords}

% -------------------------------------------------------------------
\maketitle

"""

# ------------------------------------------------------------------ #
# Read source, extract body  (from \section{Introduction} onward)    #
# ------------------------------------------------------------------ #
with open(SRC, encoding="utf-8") as f:
    src_lines = f.readlines()

# Find the line index of \section{Introduction}
intro_idx = None
for i, line in enumerate(src_lines):
    if r"\section{Introduction}" in line:
        intro_idx = i
        break

if intro_idx is None:
    raise RuntimeError("Could not find \\section{Introduction} in source file")

print(f"Body starts at line {intro_idx + 1}")

body_lines = src_lines[intro_idx:]   # everything from Introduction onward

# ------------------------------------------------------------------ #
# Apply back-matter substitutions                                     #
# ------------------------------------------------------------------ #
new_body = []
for line in body_lines:
    # Remove Springer-only \backmatter
    if line.strip() == r"\backmatter":
        new_body.append("% (backmatter removed for CAS-SC)\n")
        continue
    # Replace Springer \bmhead{Acknowledgements} -> standard section
    line = re.sub(r"\\bmhead\{Acknowledgements\}", r"\\section*{Acknowledgements}", line)
    new_body.append(line)

# ------------------------------------------------------------------ #
# Write output                                                        #
# ------------------------------------------------------------------ #
with open(DST, "w", encoding="utf-8") as f:
    f.write(PREAMBLE)
    f.writelines(new_body)

print(f"Written: {DST}")
print(f"Total lines: {len(PREAMBLE.splitlines()) + len(new_body)}")
