#!/usr/bin/env python3
"""
Cleans data/raw/cases.csv into data/interim/cases_clean.csv:

1. Merges 2 articles where a single patient's case report was split across
   multiple case_id rows (later rows never reintroduce an age/gender and
   continue with "the patient"/"our case", not a new patient) into one row
   per real patient, concatenating case_text in original order. This also
   improves measurement-to-entity linking for the later fragments:
   extract_measurements.py links to the nearest entity in the same text, and
   a later fragment alone has no visibility into entities named earlier in
   the same patient's report.
2. Drops 1 row that isn't a single-patient case at all (a 115-patient
   retrospective cohort-study summary that upstream metadata.csv mislabels
   case_amount=1) -- forcing it into the one-row-per-patient age/sex model
   would mean inventing a fake patient.
3. Re-derives age and gender FROM THE CASE TEXT ITSELF (extract_age /
   extract_gender below), instead of trusting the upstream cases.parquet
   columns. We have no visibility into whatever process built those columns
   (no source, just the one-line description in data_dictionary.csv), and it
   has confirmed bugs -- see CLAUDE.md's "Known data-quality issues" section
   for how e.g. "39-week-old newborn" got recorded as age=39. Both the
   upstream values and our own are kept in the output (age/gender = ours,
   age_upstream/gender_upstream = original) so the two can be audited
   against each other rather than one silently replacing the other.

Genuinely multi-patient articles (PMC9532611, PMC7086412, PMC11722600) are
left as separate rows -- each case_id there is already a distinct real
patient, confirmed by reading the full text of every multi-case_id article.

Downstream, point extract_entities.py / extract_measurements.py at this file
with --cases; both already support that flag, so no pipeline code changes.

Reads : data/raw/cases.csv
Writes: data/interim/cases_clean.csv

Usage:
    clinical-kg clean-cases
"""

import argparse
import csv
import re
from pathlib import Path

from clinical_kg.paths import PROJECT_ROOT

DEFAULT_IN = PROJECT_ROOT / "data" / "raw" / "cases.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "interim" / "cases_clean.csv"

# Articles where multiple case_id rows are fragments of ONE patient's report,
# not separate patients -- verified by reading the full text of every
# multi-case_id article (see CLAUDE.md). Order matters: fragments are
# concatenated in this order to reconstruct the original narrative.
MERGE_GROUPS = [
    ["PMC6083636_01", "PMC6083636_02", "PMC6083636_03"],
    ["PMC11259348_01", "PMC11259348_02"],
]

# Not single-patient case reports at all -- forcing any of these into the
# one-row-per-patient age/sex model means inventing a fake patient. Found by
# cross-checking age/gender extraction failures against the journal field in
# metadata.csv (see CLAUDE.md):
#   PMC9815523_01  -- a 115-patient retrospective cohort-study summary
#                     (metadata.csv mislabels it case_amount=1)
#   PMC10710131_01 -- a dental-education methodology paper (journal: PeerJ)
#                     describing a teaching database of 26 cases
#   PMC9650410_01  -- a sociology/business-ethics paper about Theranos
#                     (journal: Front Sociol) -- not medical at all
DROP_CASE_IDS = {"PMC9815523_01", "PMC10710131_01", "PMC9650410_01"}

OUT_FIELDS = ["case_id", "article_id", "age", "gender", "age_upstream",
              "gender_upstream", "age_method", "gender_method", "case_text",
              "source_case_ids"]

# ---------------------------------------------------------------------------
# Age / gender extraction from raw text.
#
# Clinical case reports overwhelmingly open with a fixed template -- "A/An
# <N>-year-old <gender word> presented with...", or for infants "A/An
# <N>-day/week/month-old ... newborn/infant ...". Rules are checked in
# priority order and the first one to match wins; each is scoped to a short
# OPEN_CHARS window at the start of the text specifically so that a *second*
# person's age mentioned later in the same opening sentence (most commonly a
# newborn's mother: "...newborn was born to a 45-year-old woman") isn't
# mistaken for the patient's.
# ---------------------------------------------------------------------------

OPEN_CHARS = 70   # covers "<N> years of age" (longer than "<N>-year-old") while
                  # still landing short of a second person's age in the same
                  # opening sentence -- see CLAUDE.md
FALLBACK_CHARS = 700  # empirically: 2 legitimate fallback matches sit at
                      # offset 57/62; the one known bad one sits at 2180 --
                      # see CLAUDE.md for the measurement that picked this cap

# Some reports open with 1-2 sentences of ethics/consent boilerplate before
# ever mentioning the patient (e.g. "This report complies with... Informed
# written consent was obtained from parents."). Stripped before extraction so
# OPEN_CHARS measures from the actual patient description, not the preamble.
BOILERPLATE_RE = re.compile(
    r"(?:this (?:case )?report complies with[^.]*\.\s*)"
    r"|(?:informed (?:written )?consent (?:was|has been) obtained[^.]*\.\s*)"
    r"|(?:(?:institutional review board|ethics committee|irb) approval[^.]*\.\s*)",
    re.I,
)

# Spelled-out numbers ("Seven-month-old", "Twenty-six-year-old") -> digits,
# applied before every other pattern below. Covers 1-99, which is the entire
# realistic range for a spelled-out age.
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
NUMBER_WORDS = {w: i for i, w in enumerate(_ONES)}
for _t, _tens_word in enumerate(_TENS):
    if not _tens_word:
        continue
    NUMBER_WORDS[_tens_word] = _t * 10
    for _o in range(1, 10):
        NUMBER_WORDS[f"{_tens_word}-{_ONES[_o]}"] = _t * 10 + _o
NUMBER_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")\b", re.I
)


def words_to_digits(text: str) -> str:
    return NUMBER_WORD_RE.sub(lambda m: str(NUMBER_WORDS[m.group(1).lower()]), text)


NEWBORN_RE = re.compile(r"\bnewborns?\b|\bneonat", re.I)
BABY_RE = re.compile(r"\bbaby\s+(?:boy|girl)\b", re.I)

# "<N>-year-old" / "<N> years old" / "<N>-years-old" / "<N> years of age" /
# "aged <N>" -- hyphen or space, singular or plural, plus decimals like "2.5".
_NUM = r"\d{1,3}(?:\.\d+)?"
YEAR_OLD_RE = re.compile(
    rf"(?P<n1>{_NUM})[-\s]years?[-\s]old"
    rf"|(?P<n2>{_NUM})\s+years?\s+of\s+age"
    rf"|\baged\s+(?P<n3>{_NUM})\b",
    re.I,
)
# Same shape, for day/week/month units.
SUBYEAR_OLD_RE = re.compile(
    rf"(?P<n1>{_NUM})[-\s](?P<u1>day|week|month)s?[-\s]old"
    rf"|(?P<n2>{_NUM})\s+(?P<u2>day|week|month)s?\s+of\s+age",
    re.I,
)
DAYS_PER_UNIT = {"day": 1, "week": 7, "month": 30.44}

# A range ("45-50 years old") is a signal the text may not be about one
# patient at all (see PMC10710131_01) -- flagged, never averaged into a
# single number.
AGE_RANGE_RE = re.compile(rf"{_NUM}\s*(?:-|to)\s*{_NUM}\s+years?\s+old", re.I)

GENDER_WORD_RE = re.compile(r"\b(boy|girl|male|female|man|woman)\b", re.I)
GENDER_WORD_MAP = {"boy": "Male", "male": "Male", "man": "Male",
                    "girl": "Female", "female": "Female", "woman": "Female"}

# Hand-verified exceptions the general rules can't reach without widening a
# window enough to risk false positives elsewhere. Applied after extraction,
# never silently -- see CLAUDE.md for how each was confirmed.
MANUAL_OVERRIDES = {
    # "the female patient who died" appears at char 2612 of this text --
    # far past any window we're willing to open generally. Its age is not
    # recoverable at all (the "1" a naive full-text search finds belongs to
    # an unrelated 1-year-old mentioned in an epidemiological aside; a
    # postmortem right-lung weight of 1,593 g in the same text rules out age
    # 1 outright -- an infant lung weighs ~150-200 g).
    "PMC3557982_01": {"gender": "Female", "gender_method": "manual-verified"},
}


def _fmt_age(n: float) -> str:
    """Whole-number ages print as '44', not '44.0' -- only genuinely
    fractional ages (infants, decimal mentions like '2.5-year-old') keep a
    decimal point."""
    return str(int(n)) if n == int(n) else str(n)


def extract_age(text: str) -> tuple[str, str]:
    """Returns (age_as_string, method). method documents which rule fired,
    for auditing -- it is not written to the main output, only used by
    run_report() to summarize confidence."""
    text = BOILERPLATE_RE.sub("", text, count=2)
    # Bounded to FALLBACK_CHARS -- no rule below ever looks further than that,
    # so there's no reason to risk a spelled-out-number false positive
    # ("...one of the..." -> "...1 of the...") in text we never search anyway.
    text = words_to_digits(text[:FALLBACK_CHARS]) + text[FALLBACK_CHARS:]
    opening = text[:OPEN_CHARS]

    # 1. "newborn"/"neonate" is definitional: age 0, full stop -- overrides
    #    any week/day figure in the same clause, which describes gestational
    #    age or birth weight, not time since birth.
    if NEWBORN_RE.search(opening):
        return "0", "newborn-keyword"

    # 2. A range ("45-50 years old") -- flag, don't guess a number.
    if AGE_RANGE_RE.search(opening):
        return "", "range-detected"

    # 3. Explicit "<N>-year-old" / "<N> years of age" / "aged <N>" at the
    #    very start -- the report's own patient introduction, the strongest
    #    signal available.
    m = YEAR_OLD_RE.search(opening)
    if m:
        return _fmt_age(float(m.group("n1") or m.group("n2") or m.group("n3"))), "opening-year-old"

    # 4. Explicit "<N>-month/week/day-old" / "<N> months/weeks/days of age"
    #    at the start -- convert to fractional years rather than flattening
    #    to 0 the way upstream did (upstream's own stated rule was
    #    "< 1 y.o. -> 0", but it applied that to e.g. a 20-month-old too; we
    #    compute the real fraction instead).
    m = SUBYEAR_OLD_RE.search(opening)
    if m:
        n = float(m.group("n1") or m.group("n2"))
        unit = (m.group("u1") or m.group("u2")).lower()
        years = round(n * DAYS_PER_UNIT[unit] / 365.25, 2)
        return str(years), f"opening-{unit}-old"

    # 5. "baby boy"/"baby girl" with no explicit number found above -- a
    #    weaker newborn signal than a literal "newborn", checked last among
    #    the opening-window rules so an explicit figure (rule 3/4) always
    #    wins when both are present.
    if BABY_RE.search(opening):
        return "0", "baby-keyword"

    # 6. Fallback: first year-old/aged mention within the first FALLBACK_CHARS
    #    of the full text -- NOT the whole document. Lower confidence -- in a
    #    document that goes on to discuss other people (family history, a
    #    retrospective sub-study, ...) an unbounded search can catch the
    #    wrong person's age, which is exactly what happened before this cap
    #    was added (see CLAUDE.md, PMC3557982_01).
    m = YEAR_OLD_RE.search(text[:FALLBACK_CHARS])
    if m:
        return _fmt_age(float(m.group("n1") or m.group("n2") or m.group("n3"))), "fallback-anywhere"

    return "", "not-found"


def extract_gender(text: str) -> tuple[str, str]:
    """First explicit gender word in the opening; falls back to a he/him vs
    she/her pronoun majority over a larger window when no explicit word
    appears there."""
    text = BOILERPLATE_RE.sub("", text, count=2)
    opening = text[:OPEN_CHARS * 5]
    m = GENDER_WORD_RE.search(opening)
    if m:
        return GENDER_WORD_MAP[m.group(1).lower()], "opening-word"

    window = text[:1000]
    she = len(re.findall(r"\bshe\b|\bher\b", window, re.I))
    he = len(re.findall(r"\bhe\b|\bhis\b|\bhim\b", window, re.I))
    if she > he:
        return "Female", "pronoun-majority"
    if he > she:
        return "Male", "pronoun-majority"
    return "Unknown", "not-found"


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def merge_fragments(rows_by_id: dict[str, dict], group: list[str]) -> dict:
    parts = [rows_by_id[cid] for cid in group]
    first = parts[0]
    return {
        "case_id": f"{first['article_id']}_P1",
        "article_id": first["article_id"],
        "age_upstream": next((p["age"] for p in parts if p["age"].strip()), ""),
        "gender_upstream": next(
            (p["gender"] for p in parts if p["gender"].strip() and p["gender"] != "Unknown"),
            "Unknown",
        ),
        "case_text": "\n\n".join(p["case_text"] for p in parts),
        "source_case_ids": "|".join(group),
    }


def run_report(out_rows: list[dict]) -> None:
    print("\n" + "=" * 62)
    print("AGE/GENDER: self-extracted vs. upstream cases.parquet")
    print("=" * 62)
    disagreements = 0
    for r in out_rows:
        age_disagree = r["age_upstream"].strip() and r["age"] != str(int(float(r["age_upstream"])))\
            if r["age"] and r["age_upstream"].strip() else False
        gender_disagree = r["gender_upstream"].strip() and r["gender"] != r["gender_upstream"]
        if age_disagree or gender_disagree:
            disagreements += 1
            print(f"  {r['case_id']}: "
                  f"age {r['age_upstream'] or '—'} -> {r['age']} ({r['age_method']}), "
                  f"gender {r['gender_upstream'] or '—'} -> {r['gender']} ({r['gender_method']})")
    print(f"\n  {disagreements}/{len(out_rows)} rows disagree with upstream")
    print()


def main():
    # Without argparse this stage rewrote the corpus when handed --help.
    ap = argparse.ArgumentParser(
        description="Clean the raw case corpus into one row per real patient.")
    ap.add_argument("--in", dest="source", type=Path, default=DEFAULT_IN,
                    help="raw cases CSV (default: data/raw/cases.csv)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="cleaned output CSV (default: data/interim/cases_clean.csv)")
    args = ap.parse_args()

    rows = load_rows(args.source)
    rows_by_id = {r["case_id"]: r for r in rows}
    merged_ids = {cid for group in MERGE_GROUPS for cid in group}

    out_rows = []
    for r in rows:
        cid = r["case_id"]
        if cid in DROP_CASE_IDS or cid in merged_ids:
            continue  # dropped, or handled once via MERGE_GROUPS below
        out_rows.append({
            "case_id": cid,
            "article_id": r["article_id"],
            "age_upstream": r["age"],
            "gender_upstream": r["gender"],
            "case_text": r["case_text"],
            "source_case_ids": cid,
        })

    for group in MERGE_GROUPS:
        out_rows.append(merge_fragments(rows_by_id, group))

    for r in out_rows:
        r["age"], r["age_method"] = extract_age(r["case_text"])
        r["gender"], r["gender_method"] = extract_gender(r["case_text"])
        r.update(MANUAL_OVERRIDES.get(r["case_id"], {}))

    out_rows.sort(key=lambda r: r["case_id"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, "") for k in OUT_FIELDS})

    run_report(out_rows)

    print(f"in : {len(rows)} rows  ({args.source})")
    print(f"out: {len(out_rows)} rows  ({args.out})")
    print(f"  dropped: {len(DROP_CASE_IDS)}  ({', '.join(sorted(DROP_CASE_IDS))})")
    print(f"  merged : {len(merged_ids)} fragments -> {len(MERGE_GROUPS)} patient rows")


if __name__ == "__main__":
    main()
