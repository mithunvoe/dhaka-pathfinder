#!/usr/bin/env bash
# =====================================================================
# Turn every markdown document into a PDF.
#
#   ./run.sh pdfs          (or: scripts/build_pdfs.sh)
#
# Output goes to pdf/ : one PDF per document, plus a single combined
# dossier with all of them in reading order.
#
# Needs pandoc + xelatex. We use xelatex rather than pdflatex because the
# docs are full of characters pdflatex cannot set - the maths symbols
# (in-text), the box-drawing characters in the directory trees, the greek.
# DejaVu has all of them.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="pdf"
mkdir -p "$OUT"

for tool in pandoc xelatex; do
  command -v "$tool" >/dev/null || { echo "error: $tool not installed"; exit 1; }
done

# Documents in the order a human should read them, not alphabetical.
# Format:  path | Title | Subtitle
DOCS=(
  "docs/PITCH.md                 | The Pitch                        | What to say, out loud, tomorrow"
  "docs/START_HERE.md            | Start Here                       | Assignment 3, from zero"
  "docs/MATH_EXPLAINED.md        | The Maths, Decoded               | Every equation, with real numbers"
  "statement.md                  | Formal Problem Statements        | Assignment 3, Parts A and B"
  "README.md                     | Assignment 3 - Overview          | Results, layout, how to run it"
  "docs/PART3_viva.md            | Viva Preparation                 | Question bank, both parts"
  "docs/PART5_code_defense.md    | Code Defense                     | Line by line, and the screen questions"
  "docs/PART7_literature_applied.md | Literature Applied            | Which paper changed which line"
  "docs/PART1_foundations.md     | Foundations: Population Methods  | Swarm theory"
  "docs/PART6_decision_making.md | Foundations: Decision Making     | MDP theory"
  "docs/PART4_problem_design.md  | Problem Design                   | The problems we considered and rejected"
  "docs/PART2_applications.md    | Applications                     | Twelve cited real-world uses"
  "../DEMO_DAY.md                | Demo Day                         | How to show all three assignments"
)

# DejaVu Serif has no star or ballot-X, so those come out as tofu boxes. Map just
# those characters to DejaVu Sans, which does have them. Also shrink table type a
# little - some of the comparison tables are four columns of prose.
HEADER=$(mktemp /tmp/pdfhdr.XXXX.tex)
cat > "$HEADER" <<'TEX'
\usepackage{newunicodechar}
\newfontfamily\symbolfont{DejaVu Sans}
\newunicodechar{★}{{\symbolfont ★}}
\newunicodechar{✗}{{\symbolfont ✗}}
\newunicodechar{✓}{{\symbolfont ✓}}
\newunicodechar{─}{{\symbolfont ─}}
\newunicodechar{│}{{\symbolfont │}}
\newunicodechar{└}{{\symbolfont └}}
\newunicodechar{├}{{\symbolfont ├}}
% Four columns of prose do not fit at 10pt. Let tables set a size smaller.
\let\oldlongtable\longtable
\def\longtable{\small\oldlongtable}
TEX
trap 'rm -f "$HEADER"' EXIT

PANDOC_ARGS=(
  --pdf-engine=xelatex
  --from=gfm+tex_math_dollars+pipe_tables+footnotes
  # Force explicit column widths so wide tables WRAP instead of running off the
  # page and truncating themselves. See scripts/pdf-tables.lua.
  --lua-filter="scripts/pdf-tables.lua"
  --toc --toc-depth=2
  --number-sections
  -H "$HEADER"
  -V geometry:"a4paper,margin=2cm"
  -V mainfont="DejaVu Serif"
  -V sansfont="DejaVu Sans"
  -V monofont="DejaVu Sans Mono"
  -V fontsize=10pt
  -V colorlinks=true
  -V linkcolor=NavyBlue
  -V urlcolor=NavyBlue
  -V toccolor=black
  -V documentclass=article
  --highlight-style=tango
)

echo "Building PDFs into $OUT/"
built=()
for entry in "${DOCS[@]}"; do
  IFS='|' read -r src title subtitle <<< "$entry"
  src="$(echo "$src" | xargs)"
  title="$(echo "$title" | xargs)"
  subtitle="$(echo "$subtitle" | xargs)"

  [ -f "$src" ] || { echo "  SKIP (missing): $src"; continue; }

  base="$(basename "${src%.md}")"
  dest="$OUT/${base}.pdf"

  if pandoc "$src" "${PANDOC_ARGS[@]}" \
        -V title="$title" \
        -V subtitle="$subtitle" \
        -V date="$(date +'%d %B %Y')" \
        -o "$dest" 2> "$OUT/.${base}.log"; then
    printf "  ok    %-28s -> %s (%s pages)\n" "$src" "$dest" \
           "$(pdfinfo "$dest" 2>/dev/null | awk '/^Pages/{print $2}')"
    built+=("$dest")
  else
    printf "  FAIL  %-28s  (see %s)\n" "$src" "$OUT/.${base}.log"
  fi
done

# One combined dossier, in the same reading order.
if command -v pdfunite >/dev/null && [ "${#built[@]}" -gt 0 ]; then
  pdfunite "${built[@]}" "$OUT/ALL_DOCS.pdf" 2>/dev/null \
    && printf "\n  ok    combined dossier      -> %s (%s pages)\n" \
       "$OUT/ALL_DOCS.pdf" "$(pdfinfo "$OUT/ALL_DOCS.pdf" | awk '/^Pages/{print $2}')"
fi

rm -f "$OUT"/.*.log
echo
echo "Note: report/main.pdf is the actual submission and is built separately"
echo "      with ./run.sh report - it is not one of these."
