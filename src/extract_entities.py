#!/usr/bin/env python3
"""
Longest-match entity extraction over the clinical case corpus.

Reads  : data/raw/cases.csv          (case_text + case-level age/gender)
Uses   : data/interim/gazetteer.db   (built by build_gazetteer.py)
Writes : data/processed/entities.csv (one row per entity span)

Strategy: at each token position try the longest window first (6 tokens down
to 1) and jump past whatever matched. This is what resolves entity
delimitation -- "acute pancreatitis" (C0001339) is a different concept from
"pancreatitis" (C0030305), and the longer span is the correct one.

Usage:
    python3 src/extract_entities.py
    python3 src/extract_entities.py --max-n 6 --keep-stopterms
"""

import argparse
import collections
import csv
import re
import sqlite3
import sys
from pathlib import Path

# Normalization MUST come from the gazetteer builder. A second copy that drifts
# from it causes silent misses -- no error, just missing entities.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gazetteer import normalize, sort_key  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "data" / "raw" / "cases.csv"
DEFAULT_DB = PROJECT_ROOT / "data" / "interim" / "gazetteer.db"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "entities.csv"

# Sliding-window size. Measured on this corpus: raising it to 8 finds exactly
# one extra entity across all 56 cases, so 6 is the right trade.
DEFAULT_MAX_N = 6

# UMLS semantic type -> project ontology. T033 (Finding) is deliberately its
# own type rather than folded into Symptom: it is a grab-bag holding both real
# findings ("no evidence of malignancy") and vaguer ones, and merging it would
# swamp the much cleaner T184 signal.
TUI_TO_ENTITY = {
    "T184": "Symptom",
    "T033": "Finding",
    "T047": "Diagnosis", "T191": "Diagnosis",
    "T046": "Diagnosis", "T037": "Diagnosis",
    "T059": "Exam", "T060": "Exam", "T034": "Exam",
    "T061": "Treatment", "T121": "Treatment", "T200": "Treatment",
    "T023": "BodyPart", "T029": "BodyPart",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']*|\d+(?:[.,]\d+)*")

# Figure/table references. "Fig" is itself a UMLS term (the fruit), so these
# must go before tokenizing or every citation becomes a false entity.
FIGURE_RE = re.compile(
    r"\[\s*(?:Figure|Fig|Table)[^\]]{0,40}\]"
    r"|\(\s*(?:Figure|Fig|Table)s?\.?\s*[\dA-Za-z\-,\s]{0,20}\)"
    r"|\b(?:Figure|Fig|Table)s?\.\s*\d+[a-z]?",
    re.IGNORECASE,
)

LOOKUP = """
    SELECT cui, tui, term_pref, vocabulary, tty, is_stopterm
    FROM terms WHERE term_norm = ? ORDER BY priority LIMIT 1
"""
LOOKUP_SORTED = """
    SELECT cui, tui, term_pref, vocabulary, tty, is_stopterm
    FROM terms WHERE sort_key = ? ORDER BY priority LIMIT 1
"""


def mask_figure_refs(text: str) -> str:
    """Blank out figure/table citations, preserving length so that character
    offsets still index into the original text."""
    return FIGURE_RE.sub(lambda m: " " * len(m.group()), text)


def is_bad_abbreviation(surface: str, term_pref: str, n_tokens: int) -> bool:
    """Reject a lowercase word that only matched because it collides with an
    all-caps abbreviation ("us" the pronoun vs. "US" ultrasound). Applies only
    to short single-token abbreviations, so "rib" or "eye" are unaffected."""
    return (
        n_tokens == 1
        and term_pref.isupper()
        and len(term_pref) <= 4
        and not surface.isupper()
    )


def extract(text: str, con: sqlite3.Connection, max_n: int,
            keep_stopterms: bool = False) -> list[dict]:
    """Longest-match extraction. Returns spans with offsets into `text`."""
    search_text = mask_figure_refs(text)
    toks = [(m.group(), m.start(), m.end())
            for m in TOKEN_RE.finditer(search_text)]

    found, i = [], 0
    while i < len(toks):
        hit = None
        for n in range(min(max_n, len(toks) - i), 0, -1):
            window = toks[i:i + n]
            norm = normalize(" ".join(t[0] for t in window))
            row = con.execute(LOOKUP, (norm,)).fetchone()
            if row is None and n > 1:
                # word-order fallback: "pancreatitis, acute" == "acute pancreatitis"
                row = con.execute(LOOKUP_SORTED, (sort_key(norm),)).fetchone()
            if row is None:
                continue

            cui, tui, term_pref, vocab, tty, is_stop = row
            start, end = window[0][1], window[-1][2]
            surface = text[start:end]

            if is_stop and not keep_stopterms:
                continue
            if is_bad_abbreviation(surface, term_pref or "", n):
                continue
            if tui not in TUI_TO_ENTITY:
                continue

            hit = {
                "start": start, "end": end, "surface_text": surface,
                "term_norm": norm, "cui": cui, "tui": tui,
                "entity_type": TUI_TO_ENTITY[tui], "n_tokens": n,
                "vocabulary": vocab, "tty": tty,
            }
            break

        if hit:
            found.append(hit)
            i += hit["n_tokens"]   # skip the tokens we consumed
        else:
            i += 1
    return found


def main():
    ap = argparse.ArgumentParser(description="Extract UMLS entities from case reports.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
    ap.add_argument("--keep-stopterms", action="store_true",
                    help="do not filter terms flagged as too generic")
    args = ap.parse_args()

    for path in (args.cases, args.db):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            sys.exit(1)

    print("=" * 62)
    print("ENTITY EXTRACTION")
    print("=" * 62)
    print(f"\n  cases : {args.cases}")
    print(f"  db    : {args.db}")
    print(f"  out   : {args.out}\n")

    con = sqlite3.connect(args.db)
    with open(args.cases, newline="", encoding="utf-8", errors="replace") as f:
        cases = list(csv.DictReader(f))

    fields = ["case_id", "article_id", "age", "gender", "start", "end",
              "surface_text", "term_norm", "cui", "tui", "entity_type",
              "n_tokens", "vocabulary", "tty"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_type = collections.Counter()
    surfaces = collections.Counter()
    total = 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            text = case.get("case_text") or ""
            for ent in extract(text, con, args.max_n, args.keep_stopterms):
                # offsets must index back into the untouched case_text
                assert text[ent["start"]:ent["end"]] == ent["surface_text"]
                writer.writerow({
                    "case_id": case.get("case_id"),
                    "article_id": case.get("article_id"),
                    "age": case.get("age"),
                    "gender": case.get("gender"),
                    **ent,
                })
                total += 1
                by_type[ent["entity_type"]] += 1
                surfaces[ent["surface_text"].lower()] += 1
    con.close()

    print(f"  cases processed : {len(cases)}")
    print(f"  entities        : {total:,}  ({total / max(len(cases), 1):.0f} per case)")
    print("\n  By entity type:")
    for name, count in by_type.most_common():
        bar = "#" * min(40, count // 20)
        print(f"    {name:<12} {count:>6,}  {bar}")
    print("\n  Most frequent surface forms:")
    for surface, count in surfaces.most_common(15):
        print(f"    {count:>4}  {surface}")
    print(f"\n  Written to {args.out}")


if __name__ == "__main__":
    main()
