\LoadClass[a4paper,11pt]{article}

% Essential packages
\RequirePackage{graphicx}
\RequirePackage{comment}
\RequirePackage{amssymb}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{comment}
\RequirePackage{caption}
\RequirePackage{multirow}
\RequirePackage{listings}
\RequirePackage{xcolor}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{enumitem}
\RequirePackage{titlesec}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{tcolorbox}
\RequirePackage{fancyhdr}
\RequirePackage{mdframed}
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\RequirePackage{parskip}
\RequirePackage{emptypage}
\RequirePackage{svg}

% Command to get the current month name
\newcommand{\MonthName}{
  \ifcase\month\or January\or February\or March\or April\or May\or June\or July\or August\or September\or October\or November\or December\fi}

\makeatletter
\def\proofname{\bfseries Proof}
\newcommand\HHUGE{\@setfontsize\Huge{35}{45}}
\makeatother

\pgfplotsset{compat=1.18}

\newcommand{\authors}[1]{\begin{tabular}{@{}ll@{}}Davide Donà &\quad \texttt{264467}\\Andrea Blushi&\quad \texttt{264468}\end{tabular}}

\newcommand{\maketitlepage}{
  \begin{titlepage}
    \begin{center}
      \Large University of Trento\\[-3pt]
      Master's Degree in Computer Science

      \vspace{1.25cm}

      \IfFileExists{../svg-inkscape/logo_svg-tex.pdf_tex}{%
        \begingroup
        \graphicspath{{../svg-inkscape/}}%
        \def\svgwidth{0.7\textwidth}%
        \input{../svg-inkscape/logo_svg-tex.pdf_tex}%
        \endgroup
      }{%
        \includesvg[width=0.7\textwidth]{../media/logo.svg}%
      }

      \vspace{1cm}
      \HHUGE\textbf{Computer Vision project}

      \vspace{0.5cm}

      \LARGE Computer Vision

      \vspace{-.3cm}

      \textsc{a.y.} 2025/2026

      \vspace{3.5cm}

      \LARGE\authors

      \vspace{3.5cm}
      \vfill

      \Large\MonthName\ \the\year
    \end{center}
  \end{titlepage}
  \pagenumbering{Roman}
}

% add after toc these commands:
\let\oldtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{
  \oldtableofcontents
  \newpage
  \pagenumbering{arabic}
}
