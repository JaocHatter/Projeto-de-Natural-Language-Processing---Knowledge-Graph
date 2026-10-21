#!/usr/bin/env bash
# Filters UMLS MRCONSO.RRF + MRSTY.RRF down to the 14 clinical semantic types
# this project uses, producing one CSV per type plus a consolidated file.
#
# NOTE: reconstructed from data/README.md's spec (the original author's copy
# was never committed to this repo -- see git history). Behavior should match
# what that document describes; if a teammate's original script turns up,
# prefer it and diff against this one rather than assuming they're identical.
#
# Must be run from inside data/external/ (uses relative paths):
#   cd data/external
#   bash filter_umls_mrsty.bash
#
# Requires two UMLS release zips already in this directory (see
# data/README.md for where to get them) and ~10 GB free disk space.
set -euo pipefail

REL="2026AA"
ZIP_MRCONSO="umls-${REL}-mrconso.zip"
ZIP_FULL="umls-${REL}-metathesaurus-full.zip"

WORK_DIR="umls_work"
OUT_DIR="umls_csvs"

# The 14 semantic types this project keeps (see project README.md for the
# TUI -> meaning table).
TIPOS=(T047 T200 T023 T191 T033 T037 T046 T059 T060 T061 T121 T029 T184 T034)
declare -A NAMES=(
    [T047]=Disease_or_Syndrome        [T200]=Clinical_Drug
    [T023]=Body_Part_Organ            [T191]=Neoplastic_Process
    [T033]=Finding                    [T037]=Injury_or_Poisoning
    [T046]=Pathologic_Function        [T059]=Laboratory_Procedure
    [T060]=Diagnostic_Procedure       [T061]=Therapeutic_Procedure
    [T121]=Pharmacologic_Substance    [T029]=Body_Location_or_Region
    [T184]=Sign_or_Symptom            [T034]=Laboratory_or_Test_Result
)

# Vocabularies kept, as an *unanchored* regex -- deliberately so: MTH also
# admits MTHSPL/MTHICD9/MTHMST/MTHICPC2*, and ICD10CM admits CCSR_ICD10CM.
# src/build_gazetteer.py filters that leak back out downstream by matching
# vocabularies exactly. Set VOCABS="" to keep every vocabulary.
VOCABS="SNOMEDCT_US|MSH|LNC|RXNORM|ICD10CM|MTH"

HEADER="cui,tui,termino_norm,termino_original,vocabulario,tipo_termino"

mkdir -p "$WORK_DIR" "$OUT_DIR"

echo "============================================================"
echo "UMLS FILTER (release $REL)"
echo "============================================================"

echo
echo "[1/5] Extracting MRCONSO.RRF / MRSTY.RRF"
if [ -f "$WORK_DIR/MRCONSO.RRF" ]; then
    echo "  MRCONSO.RRF already present, skipping"
else
    [ -f "$ZIP_MRCONSO" ] || { echo "ERROR: missing $ZIP_MRCONSO in $(pwd)" >&2; exit 1; }
    echo "  extracting from $ZIP_MRCONSO ..."
    unzip -p "$ZIP_MRCONSO" "*/META/MRCONSO.RRF" > "$WORK_DIR/MRCONSO.RRF"
fi
if [ -f "$WORK_DIR/MRSTY.RRF" ]; then
    echo "  MRSTY.RRF already present, skipping"
else
    [ -f "$ZIP_FULL" ] || { echo "ERROR: missing $ZIP_FULL in $(pwd)" >&2; exit 1; }
    echo "  extracting from $ZIP_FULL (this is the slow one -- 5.5 GB archive) ..."
    unzip -p "$ZIP_FULL" "*/META/MRSTY.RRF" > "$WORK_DIR/MRSTY.RRF"
fi
echo "  MRCONSO.RRF: $(du -h "$WORK_DIR/MRCONSO.RRF" | cut -f1)"
echo "  MRSTY.RRF  : $(du -h "$WORK_DIR/MRSTY.RRF" | cut -f1)"

echo
echo "[2/5] Filtering MRCONSO (English, non-suppressed, target vocabularies)"
# RRF columns (1-indexed, pipe-delimited): 1=CUI 2=LAT 12=SAB 13=TTY 15=STR 17=SUPPRESS
awk -F'|' -v vocabs="$VOCABS" '
    BEGIN { OFS="\t" }
    $2 == "ENG" && $17 == "N" && $12 ~ vocabs {
        str = $15
        gsub(/\t/, " ", str)
        print $1, $12, $13, str
    }
' "$WORK_DIR/MRCONSO.RRF" > "$WORK_DIR/conso_filtrado.tsv"
echo "  $(wc -l < "$WORK_DIR/conso_filtrado.tsv") rows kept"

echo
echo "[3/5] Sorting filtered MRCONSO by CUI (for the join)"
LC_ALL=C sort -t $'\t' -k1,1 "$WORK_DIR/conso_filtrado.tsv" > "$WORK_DIR/conso_sorted.tsv"

echo
echo "[4/5] Building a CUI list per semantic type from MRSTY"
for tui in "${TIPOS[@]}"; do
    awk -F'|' -v tui="$tui" '$2 == tui { print $1 }' "$WORK_DIR/MRSTY.RRF" \
        | LC_ALL=C sort -u > "$WORK_DIR/cuis_${tui}.txt"
    echo "  $tui: $(wc -l < "$WORK_DIR/cuis_${tui}.txt") CUIs"
done

echo
echo "[5/5] Joining terms with each semantic type's CUI list"
consolidated="$OUT_DIR/umls_clinico_todos.csv"
echo "$HEADER" > "$consolidated"

for tui in "${TIPOS[@]}"; do
    out="$OUT_DIR/umls_${tui}_${NAMES[$tui]}.csv"
    echo "$HEADER" > "$out"
    LC_ALL=C join -t $'\t' -1 1 -2 1 "$WORK_DIR/cuis_${tui}.txt" "$WORK_DIR/conso_sorted.tsv" \
        | awk -F'\t' -v tui="$tui" '
            BEGIN { OFS="," }
            function csvq(s) { gsub(/"/, "\"\"", s); return "\"" s "\"" }
            {
                cui = $1; sab = $2; tty = $3; str = $4
                norm = tolower(str)
                print cui, tui, csvq(norm), csvq(str), csvq(sab), csvq(tty)
            }
        ' | tee -a "$out" >> "$consolidated"
    echo "  $tui -> $out ($(( $(wc -l < "$out") - 1 )) terms)"
done

echo
echo "============================================================"
echo "Done. Consolidated: $consolidated ($(( $(wc -l < "$consolidated") - 1 )) rows)"
echo "Next: python3 ../../src/build_gazetteer.py   (run from the project root)"
echo "============================================================"
