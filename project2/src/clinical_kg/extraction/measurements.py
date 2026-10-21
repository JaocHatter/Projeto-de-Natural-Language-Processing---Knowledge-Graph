#!/usr/bin/env python3
"""
Value + unit extraction, linked to the nearest preceding entity.

Classic-technique complement to extract_entities.py: a "850 U/L" or
"10-140 U/L" or "4 cm" on its own is meaningless for the knowledge graph
until it is attached to what it measures (a lab exam result, a lesion size,
a medication dose, a symptom duration, ...). This script finds
value(+range)+unit spans with a closed unit vocabulary, then links each one
to the nearest entity that precedes it in the same sentence (falling back to
a character window when the sentence has no earlier entity), the same
heuristic used in the project's example knowledge graphs (e.g. "serum lipase
... (850 U/L)" -> the value attaches to the "serum lipase" Exam entity).

Reads  : data/interim/cases_clean.csv
Uses   : data/interim/gazetteer.db (via extract_entities.extract, to find the
         entities a value can attach to)
Writes : data/processed/measurements.csv

Usage:
    clinical-kg extract-measurements
"""

import argparse
import bisect
import collections
import csv
import re
import sqlite3
import sys
from pathlib import Path

from clinical_kg.paths import PROJECT_ROOT

from clinical_kg.extraction.entities import (
    DEFAULT_CASES, DEFAULT_DB, DEFAULT_MAX_N, mask_figure_refs, extract,
)

DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "measurements.csv"

CSV_COLUMNS = ["case_id", "article_id", "start", "end", "surface_text", "is_range",
               "value", "range_low", "range_high", "unit", "unit_norm", "unit_category",
               "linked_entity_start", "linked_entity_end", "linked_entity_surface",
               "linked_entity_type", "link_method", "linked_entity_cui", "linked_entity_id"]

# Closed unit vocabulary -> category. Deliberately excludes bare single-letter
# units ("L", "M", "C") that are ambiguous in this corpus -- "L"/"R" mark
# laterality ("L kidney"), not liters, in case-report style text. Units are
# matched case-insensitively; longer units are tried before their prefixes
# (see _UNIT_LITERALS) so "mg/dL" is not swallowed by "mg".
UNIT_CATEGORY = {
    # lab concentration
    "mg/dl": "lab_concentration", "mg/l": "lab_concentration",
    "ng/ml": "lab_concentration", "pg/ml": "lab_concentration",
    "miu/ml": "lab_concentration", "iu/l": "lab_concentration",
    "mmol/l": "lab_concentration", "meq/l": "lab_concentration",
    "u/l": "lab_concentration", "g/dl": "lab_concentration",
    "mcg/ml": "lab_concentration", "mcg/dl": "lab_concentration",
    "%": "percentage",
    # mass (dosage)
    "mcg": "mass", "µg": "mass", "ug": "mass", "mg": "mass", "kg": "mass", "g": "mass",
    # volume
    "ml": "volume", "dl": "volume",
    # dose unit
    "iu": "dose_unit", "meq": "dose_unit", "mmol": "dose_unit",
    # length / size
    "mm": "length", "cm": "length",
    # pressure
    "mmhg": "pressure",
    # time
    "days": "time", "day": "time", "weeks": "time", "week": "time",
    "months": "time", "month": "time", "years": "time", "year": "time",
    "hours": "time", "hour": "time", "hrs": "time", "hr": "time",
    "mins": "time", "min": "time",
    # rate
    "bpm": "rate",
    # temperature (degree sign required -- bare "c"/"f" are far too ambiguous)
    "°c": "temperature", "°f": "temperature",
}

_UNIT_LITERALS = sorted(UNIT_CATEGORY, key=len, reverse=True)
_UNIT_ALT = "|".join(re.escape(u) for u in _UNIT_LITERALS)

# A number, then optional space/hyphen (covers "5-day", "4 cm"), then a unit,
# then a lookahead that rejects the match if another letter follows -- this
# is what keeps "3 groups" from matching the "g" (mass) unit: after the
# literal "g" the next character is "r", so the lookahead fails.
_NUM = r"\d+(?:[.,]\d+)?"
VALUE_RE = re.compile(
    rf"(?P<value>{_NUM})[\s-]{{0,2}}(?P<unit>{_UNIT_ALT})(?![A-Za-z])",
    re.IGNORECASE,
)
# Reference ranges: "10-140 U/L". Matched first and masked out before VALUE_RE
# runs, so a range is never also reported as one bare trailing value.
RANGE_RE = re.compile(
    rf"(?P<low>{_NUM})\s*[-–]\s*(?P<high>{_NUM})\s*(?P<unit>{_UNIT_ALT})(?![A-Za-z])",
    re.IGNORECASE,
)

# "52-year-old", "5-year-old boy": age phrasing, not a duration measurement --
# age is already a structured column in cases.csv (see data/README.md), and
# without this guard every case's opening sentence emits a bogus "value" row.
_AGE_UNITS = {"year", "years", "month", "months", "week", "weeks", "day", "days"}
_AGE_SUFFIX_RE = re.compile(r"-?\s?old\b", re.IGNORECASE)

# "discharged on day 8" / "day 3 post-op": unit-before-value time phrasing,
# the mirror image of VALUE_RE's value-before-unit order. Restricted to time
# units -- this is specifically for the hospital-course/timeline idiom, not a
# general unit-before-value grammar (e.g. blood pressure "120/80" is a
# different structure, out of scope here).
_TIME_UNITS_BEFORE = ["days", "day", "weeks", "week", "months", "month",
                      "years", "year", "hours", "hour"]
UNIT_BEFORE_VALUE_RE = re.compile(
    rf"\b(?P<unit>{'|'.join(_TIME_UNITS_BEFORE)})\s+(?P<value>\d+)\b",
    re.IGNORECASE,
)

_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

# If no entity precedes the value in the same sentence, fall back to the
# nearest one within this many characters -- covers values placed in the
# sentence right after the one that names the exam/medication.
WINDOW_CHARS = 150


def find_measurements(text: str) -> list[dict]:
    """Find value/range+unit spans, offsets into the original `text`."""
    search_text = mask_figure_refs(text)
    found = []

    range_spans = []
    for m in RANGE_RE.finditer(search_text):
        start, end = m.start(), m.end()
        unit_norm = m.group("unit").lower()
        found.append({
            "start": start, "end": end, "surface_text": text[start:end],
            "is_range": True, "value": "",
            "range_low": m.group("low"), "range_high": m.group("high"),
            "unit": m.group("unit"), "unit_norm": unit_norm,
            "unit_category": UNIT_CATEGORY.get(unit_norm, "other"),
        })
        range_spans.append((start, end))

    # Blank out the range spans (length-preserving, like mask_figure_refs) so
    # VALUE_RE cannot re-match the trailing number+unit of a range as a
    # standalone value.
    masked = search_text
    for start, end in range_spans:
        masked = masked[:start] + (" " * (end - start)) + masked[end:]

    for m in VALUE_RE.finditer(masked):
        start, end = m.start(), m.end()
        unit_norm = m.group("unit").lower()
        if unit_norm in _AGE_UNITS and _AGE_SUFFIX_RE.match(masked, end):
            continue  # "52-year-old" etc. -- age, not a duration
        found.append({
            "start": start, "end": end, "surface_text": text[start:end],
            "is_range": False, "value": m.group("value"),
            "range_low": "", "range_high": "",
            "unit": m.group("unit"), "unit_norm": unit_norm,
            "unit_category": UNIT_CATEGORY.get(unit_norm, "other"),
        })

    claimed = {(d["start"], d["end"]) for d in found}
    for m in UNIT_BEFORE_VALUE_RE.finditer(masked):
        start, end = m.start(), m.end()
        if any(start < c_end and end > c_start for c_start, c_end in claimed):
            continue  # already covered by a value-before-unit match
        unit_norm = m.group("unit").lower()
        found.append({
            "start": start, "end": end, "surface_text": text[start:end],
            "is_range": False, "value": m.group("value"),
            "range_low": "", "range_high": "",
            "unit": m.group("unit"), "unit_norm": unit_norm,
            "unit_category": UNIT_CATEGORY.get(unit_norm, "other"),
        })

    found.sort(key=lambda d: d["start"])
    return found


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Contiguous (start, end) sentence spans covering the whole text."""
    spans, start = [], 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _sentence_index(offset: int, sent_starts: list[int]) -> int:
    i = bisect.bisect_right(sent_starts, offset) - 1
    return max(i, 0)


# "4 cm pseudocyst", "5-day history": the value is an adjectival modifier
# sitting immediately in front of the entity it describes, rather than after
# it. Tried only within this small gap so it doesn't turn into a second,
# looser backward search in disguise.
FORWARD_GAP_CHARS = 25


def link_measurement(meas: dict, entities: list[dict], sent_starts: list[int]) -> dict:
    """Nearest entity: same-sentence backward, then same-sentence forward
    (for adjectival "4 cm pseudocyst" phrasing), then a backward char window."""
    m_sent = _sentence_index(meas["start"], sent_starts)

    def best_backward(candidates):
        chosen = None
        for ent in candidates:
            if ent["end"] > meas["start"]:
                continue
            if chosen is None or ent["end"] > chosen["end"]:
                chosen = ent
        return chosen

    def best_forward(candidates):
        chosen = None
        for ent in candidates:
            gap = ent["start"] - meas["end"]
            if not (0 <= gap <= FORWARD_GAP_CHARS):
                continue
            if chosen is None or ent["start"] < chosen["start"]:
                chosen = ent
        return chosen

    same_sentence = [e for e in entities
                      if _sentence_index(e["start"], sent_starts) == m_sent]
    best = best_backward(same_sentence)
    method = "same_sentence" if best else None

    if best is None:
        best = best_forward(same_sentence)
        method = "same_sentence_forward" if best else None

    if best is None:
        nearby = [e for e in entities if 0 <= meas["start"] - e["end"] <= WINDOW_CHARS]
        best = best_backward(nearby)
        if best is not None:
            method = "window_fallback"

    if best is None:
        return {"linked_entity_start": "", "linked_entity_end": "",
                "linked_entity_surface": "", "linked_entity_type": "",
                "linked_entity_cui": "", "linked_entity_id": "",
                "link_method": "none"}
    return {"linked_entity_start": best["start"], "linked_entity_end": best["end"],
            "linked_entity_surface": best["surface_text"],
            "linked_entity_cui": best.get("cui", ""),
            "linked_entity_id": f'concept:{best["cui"]}' if best.get("cui") else "",
            "linked_entity_type": best["entity_type"], "link_method": method}


def analyze_case(text: str, con: sqlite3.Connection | None,
                 max_n: int = DEFAULT_MAX_N) -> tuple[list[dict], list[dict]]:
    """One shared extraction pass for the app, export CLI and measurement CLI."""
    entities = extract(text, con, max_n) if con is not None else []
    sent_starts = [start for start, _ in split_sentences(text)]
    measurements = find_measurements(text)
    for meas in measurements:
        meas.update(link_measurement(meas, entities, sent_starts))
    return entities, measurements


def main():
    ap = argparse.ArgumentParser(
        description="Extract value/range+unit spans and link them to nearby entities.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
    args = ap.parse_args()

    if not args.cases.exists():
        print(f"ERROR: missing {args.cases}", file=sys.stderr)
        sys.exit(1)

    print("=" * 62)
    print("MEASUREMENT EXTRACTION (value + unit, linked to nearest entity)")
    print("=" * 62)
    print(f"\n  cases : {args.cases}")
    print(f"  db    : {args.db}")
    print(f"  out   : {args.out}\n")

    con = None
    if args.db.exists():
        con = sqlite3.connect(args.db)
    else:
        print(f"  WARNING: {args.db} not found -- running without the UMLS gazetteer.\n"
              "  There is nothing for a value to link to without it, so every\n"
              "  measurement will come out with link_method=none. Build it first:\n"
              "  clinical-kg build-gazetteer\n")

    with open(args.cases, newline="", encoding="utf-8", errors="replace") as f:
        cases = list(csv.DictReader(f))

    fields = CSV_COLUMNS

    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_category = collections.Counter()
    by_link_method = collections.Counter()
    total = 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            text = case.get("case_text") or ""
            _, measurements = analyze_case(text, con, args.max_n)
            for meas in measurements:
                assert text[meas["start"]:meas["end"]] == meas["surface_text"]
                writer.writerow({
                    "case_id": case.get("case_id"),
                    "article_id": case.get("article_id"),
                    **meas,
                })
                total += 1
                by_category[meas["unit_category"]] += 1
                by_link_method[meas["link_method"]] += 1
    if con is not None:
        con.close()

    print(f"  cases processed : {len(cases)}")
    print(f"  measurements    : {total:,}")
    print("\n  By unit category:")
    for name, count in by_category.most_common():
        print(f"    {count:>4}  {name}")
    print("\n  By link method:")
    for name, count in by_link_method.most_common():
        print(f"    {count:>4}  {name}")
    print(f"\n  Written to {args.out}")


if __name__ == "__main__":
    main()
