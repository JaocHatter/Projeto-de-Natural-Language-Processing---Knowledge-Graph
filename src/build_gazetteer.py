#!/usr/bin/env python3
"""
Build the SQLite gazetteer from the CSVs produced by
data/external/filter_umls_mrsty.bash

Resulting table: terms
    term_norm    TEXT   <- exact lookup key
    sort_key     TEXT   <- words sorted alphabetically (fallback lookup)
    n_tokens     INT    <- word count (drives maximal/longest matching)
    cui          TEXT
    tui          TEXT
    term_pref    TEXT
    vocabulary   TEXT
    priority     INT    <- lower = better (used to disambiguate)
    is_stopterm  INT    <- 1 if too generic to trust on its own
    tty          TEXT   <- UMLS term type (PT, SY, LA, ...)

Usage:
    python3 src/build_gazetteer.py
    python3 src/build_gazetteer.py --csv-dir data/external/umls_csvs \
                                   --db data/interim/gazetteer.db
"""

import argparse
import csv
import os
import re
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "external" / "umls_csvs"
DEFAULT_DB = PROJECT_ROOT / "data" / "interim" / "gazetteer.db"

# Column headers as written by filter_umls_mrsty.bash. They are in Spanish
# because that script produced the CSVs; change them here (not elsewhere) if
# the CSVs are ever regenerated with different names.
COL_CUI = "cui"
COL_TUI = "tui"
COL_TERM_NORM = "termino_norm"
COL_TERM_ORIGINAL = "termino_original"
COL_VOCABULARY = "vocabulario"
COL_TERM_TYPE = "tipo_termino"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Longest term we keep, in words. Must be >= the sliding-window size used by
# the extractor, otherwise long entries are stored but can never be matched.
MAX_TOKENS = 8

# Only these vocabularies are kept. Matched exactly -- the bash script used an
# unanchored regex, so "MTH" also admitted MTHSPL/MTHICD9/MTHMST/MTHICPC2* and
# "ICD10CM" admitted CCSR_ICD10CM (~22.7k rows). MTHSPL is where "fig" came from.
ALLOWED_VOCABS = {"SNOMEDCT_US", "MSH", "LNC", "RXNORM", "ICD10CM", "MTH"}

# Single-token LOINC entries are dropped wholesale (see load_csvs), which costs
# 4,252 answer-list values -- all junk except these. Verified against the corpus:
# of 155 LNC-only single tokens appearing in the 56 cases, "ct" is the only
# clinically meaningful one (48/48 occurrences uppercase).
LNC_SINGLE_ALLOW = {"ct"}

# LOINC group/section term types: HTML-escaped panel names ("...&#x7c;imp&#x7c;")
# and fragments like "h&p.hx". Never clinical terms, at any length.
EXCLUDED_TTY = {"LG", "HS"}

# Vocabulary priority for disambiguation (lower wins)
VOCAB_PRIORITY = {
    "SNOMEDCT_US": 0,
    "MSH": 1,
    "LNC": 2,
    "RXNORM": 3,
    "ICD10CM": 4,
    "MTH": 5,
}

# Term-type priority (lower wins): PT = preferred term
TTY_PRIORITY = {
    "PT": 0, "PN": 1, "MH": 1, "FN": 2,
    "SY": 3, "SYN": 3, "AB": 6, "IS": 7,
}

# Terms that are too generic: they exist in UMLS but produce false positives.
# They are loaded anyway, just flagged, so extraction can filter them.
STOPTERMS = {
    # generic clinical nouns
    "history", "patient", "case", "day", "time", "level", "levels", "result",
    "results", "study", "studies", "examination", "evaluation", "presence",
    "absence", "normal", "abnormal", "significant", "type", "types", "value",
    "unit", "units", "test", "tests", "procedure", "disease", "disorder",
    "finding", "findings", "sign", "signs", "symptom", "symptoms", "mass",
    "male", "female", "man", "woman", "age", "size", "week", "month", "year",
    "years", "past", "present", "left", "right", "upper", "lower", "total",
    "she", "he", "her", "his", "it", "one", "two", "three", "first", "second",
    # units of measure
    "mg", "mm", "cm", "ml", "kg", "mmhg", "mcg", "iu", "dl", "mmol", "mg/dl",
    # quantifiers / function words that survive as SNOMED PT or MTH PN
    "no", "per", "four", "five", "none", "both", "each", "other", "at", "air",
    # bare modifiers -- qualify an entity, are not one
    "negative", "positive", "severe", "mild", "moderate", "confirmed",
    "improved", "physical", "acute", "chronic", "large", "small",
    # SNOMED qualifier / context phrases (matched as whole phrases)
    "history of", "presence of", "absence of", "negative for", "positive for",
    "treated with", "transferred to", "no evidence of", "evidence of",
    "consistent with", "associated with", "due to", "status post",
}

# British -> American spelling (UMLS uses American)
DIALECT = [
    ("coeliac", "celiac"), ("oesophag", "esophag"), ("haemat", "hemat"),
    ("haemorrh", "hemorrh"), ("paediatr", "pediatr"), ("anaemi", "anemi"),
    ("tumour", "tumor"), ("oedema", "edema"), ("diarrhoea", "diarrhea"),
    ("caecum", "cecum"), ("leucocyt", "leukocyt"), ("foetal", "fetal"),
    ("gynaecolog", "gynecolog"), ("orthopaedic", "orthopedic"),
]

_RE_SPACES = re.compile(r"\s+")
# NOTE: this is deliberately broad and also rewrites words that are not
# British -ise forms ("raised" -> "raized"). That is harmless as long as the
# extractor applies the exact same normalization, since both sides collide the
# same way -- but do not "fix" it on one side only.
_RE_ISED = re.compile(r"is(ed|ation|ing|es)\b")


def normalize(text: str) -> str:
    """Normalization applied both when loading the dictionary and when querying.

    It MUST be identical on both sides or lookups will silently miss.
    """
    s = text.lower().strip()
    for br, us in DIALECT:
        s = s.replace(br, us)
    s = _RE_ISED.sub(r"iz\1", s)
    s = s.replace("‐", "-").replace("‑", "-")
    s = s.replace("‘", "'").replace("’", "'")
    s = _RE_SPACES.sub(" ", s)
    return s.strip()


def sort_key(term_norm: str) -> str:
    """Words sorted alphabetically, so that 'pancreatitis, acute' and
    'acute pancreatitis' collapse to the same key."""
    words = re.findall(r"[a-z0-9]+", term_norm)
    return " ".join(sorted(words))


def priority(vocab: str, tty: str) -> int:
    return VOCAB_PRIORITY.get(vocab, 9) * 10 + TTY_PRIORITY.get(tty, 8)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def create_schema(con: sqlite3.Connection) -> None:
    con.executescript("""
        DROP TABLE IF EXISTS terms;
        CREATE TABLE terms (
            term_norm    TEXT NOT NULL,
            sort_key     TEXT NOT NULL,
            n_tokens     INTEGER NOT NULL,
            cui          TEXT NOT NULL,
            tui          TEXT NOT NULL,
            term_pref    TEXT,
            vocabulary   TEXT,
            priority     INTEGER NOT NULL,
            is_stopterm  INTEGER NOT NULL DEFAULT 0,
            tty          TEXT
        );
    """)
    con.commit()


def load_csvs(con: sqlite3.Connection, csv_dir: Path) -> int:
    """Load the consolidated CSV; fall back to the per-type CSVs."""
    consolidated = csv_dir / "umls_clinico_todos.csv"

    if consolidated.exists():
        files = [consolidated]
    else:
        files = sorted(
            p for p in csv_dir.iterdir()
            if p.name.startswith("umls_T") and p.suffix == ".csv"
        )
        if not files:
            print(f"ERROR: no CSVs found in {csv_dir}", file=sys.stderr)
            print("Run this first: bash data/external/filter_umls_mrsty.bash",
                  file=sys.stderr)
            sys.exit(1)

    # Deduplication: for each (term_norm, cui) keep the best-priority row.
    best: dict[tuple[str, str], tuple] = {}
    read = 0

    for path in files:
        print(f"  reading {path.name}...")
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                read += 1
                if read % 500_000 == 0:
                    print(f"    {read:,} rows...")

                cui = (row.get(COL_CUI) or "").strip()
                tui = (row.get(COL_TUI) or "").strip()
                original = (row.get(COL_TERM_ORIGINAL) or "").strip()
                norm_csv = (row.get(COL_TERM_NORM) or "").strip()
                vocab = (row.get(COL_VOCABULARY) or "").strip()
                tty = (row.get(COL_TERM_TYPE) or "").strip()

                if not cui or not tui:
                    continue
                if vocab not in ALLOWED_VOCABS or tty in EXCLUDED_TTY:
                    continue

                # Re-normalize here so the key is guaranteed consistent with
                # what the extractor will compute at query time.
                norm = normalize(norm_csv or original)
                if not norm or len(norm) < 2:
                    continue
                n_tok = len(norm.split())
                if n_tok > MAX_TOKENS:
                    continue
                # Single-token LOINC entries are answer-list values and axis
                # fragments ("at", "mm", "(+)", "0=none", "$25,000-$49,999"),
                # not clinical terms. Multi-word LOINC names are kept.
                if n_tok == 1 and vocab == "LNC" and norm not in LNC_SINGLE_ALLOW:
                    continue
                # drop purely numeric entries
                if re.fullmatch(r"[\d\s.,\-]+", norm):
                    continue

                p = priority(vocab, tty)
                key = (norm, cui)
                if key not in best or p < best[key][7]:
                    best[key] = (
                        norm,
                        sort_key(norm),
                        n_tok,
                        cui,
                        tui,
                        original,
                        vocab,
                        p,
                        1 if norm in STOPTERMS else 0,
                        tty,
                    )

    print(f"  rows read              : {read:,}")
    print(f"  unique (term,cui) pairs: {len(best):,}")

    con.executemany("INSERT INTO terms VALUES (?,?,?,?,?,?,?,?,?,?)", best.values())
    con.commit()
    return len(best)


def create_indexes(con: sqlite3.Connection) -> None:
    print("  creating indexes...")
    con.executescript("""
        CREATE INDEX idx_norm  ON terms(term_norm);
        CREATE INDEX idx_sort  ON terms(sort_key);
        CREATE INDEX idx_cui   ON terms(cui);
        CREATE INDEX idx_tui   ON terms(tui);
        CREATE INDEX idx_ntok  ON terms(n_tokens);
    """)
    con.execute("ANALYZE")
    con.commit()


def report_stats(con: sqlite3.Connection) -> None:
    print("\n" + "=" * 62)
    print("STATISTICS")
    print("=" * 62)

    total = con.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
    cuis = con.execute("SELECT COUNT(DISTINCT cui) FROM terms").fetchone()[0]
    print(f"\nIndexed terms   : {total:,}")
    print(f"Concepts (CUI)  : {cuis:,}")

    print("\nBy semantic type:")
    for tui, n_cui, n_term in con.execute("""
        SELECT tui, COUNT(DISTINCT cui), COUNT(*)
        FROM terms GROUP BY tui ORDER BY 2 DESC
    """):
        print(f"  {tui:6}  {n_cui:>8,} concepts  {n_term:>10,} terms")

    print("\nBy word count (this is what makes maximal matching work):")
    for n, c in con.execute("""
        SELECT n_tokens, COUNT(*) FROM terms
        GROUP BY n_tokens ORDER BY n_tokens
    """):
        bar = "#" * min(40, c // 25_000)
        print(f"  {n} word(s): {c:>9,}  {bar}")

    multi = con.execute("SELECT COUNT(*) FROM terms WHERE n_tokens > 1").fetchone()[0]
    stop = con.execute("SELECT COUNT(*) FROM terms WHERE is_stopterm = 1").fetchone()[0]
    print(f"\n  Multi-word : {multi:,} ({100 * multi / total:.1f}%)")
    print("  <- this is why the dictionary resolves entity delimitation")
    print(f"  Stopterms  : {stop:,} (flagged, not removed)")


def test_corpus_terms(con: sqlite3.Connection) -> None:
    """Smoke test with terms drawn from the case-report corpus."""
    print("\n" + "=" * 62)
    print("SMOKE TEST WITH CORPUS TERMS")
    print("=" * 62 + "\n")

    cases = [
        # cardiac case report (caso1.txt)
        "ascending aortic aneurysm", "aortic aneurysm", "breast cancer",
        "computed tomography", "coronary artery", "stenosis",
        "internal mammary artery", "systolic function", "cardiac rehabilitation",
        # gastrointestinal / pancreatic
        "acute pancreatitis", "chronic pancreatitis", "pancreatitis",
        "endoscopic ultrasound", "gastric duplication cyst",
        "mucinous cystic neoplasm", "distal pancreatectomy",
        # general
        "hypertension", "pulmonary hypertension", "pancreas", "stomach",
        "epigastric pain",
        # British spellings, to exercise normalization
        "coeliac axis", "oesophagogastroduodenoscopy", "tumour",
    ]

    hits = 0
    for term in cases:
        norm = normalize(term)
        row = con.execute("""
            SELECT cui, tui FROM terms
            WHERE term_norm = ? ORDER BY priority LIMIT 1
        """, (norm,)).fetchone()

        if row:
            hits += 1
            print(f"  OK   {term:<32} -> {row[0]}  {row[1]}")
        else:
            # fallback: word-order-insensitive key
            row = con.execute("""
                SELECT cui, tui FROM terms
                WHERE sort_key = ? ORDER BY priority LIMIT 1
            """, (sort_key(norm),)).fetchone()
            if row:
                hits += 1
                print(f"  ~SRT {term:<32} -> {row[0]}  {row[1]}")
            else:
                print(f"  MISS {term:<32}")

    print(f"\n  {hits}/{len(cases)} matched")

    # Why longest match matters
    print("\n" + "-" * 62)
    print("Why maximal matching matters:")
    print("-" * 62)
    for short, long in [("pancreatitis", "acute pancreatitis"),
                        ("hypertension", "pulmonary hypertension"),
                        ("ultrasound", "endoscopic ultrasound"),
                        ("aneurysm", "ascending aortic aneurysm")]:
        q = "SELECT cui FROM terms WHERE term_norm=? ORDER BY priority LIMIT 1"
        ra = con.execute(q, (normalize(short),)).fetchone()
        rb = con.execute(q, (normalize(long),)).fetchone()
        ca = ra[0] if ra else "---"
        cb = rb[0] if rb else "---"
        mark = "DIFFERENT CUI" if ca != cb else "same"
        print(f"  {short:<14} {ca:<12} | {long:<26} {cb:<12}  {mark}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build the UMLS SQLite gazetteer.")
    ap.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()

    if not args.csv_dir.is_dir():
        print(f"ERROR: {args.csv_dir} does not exist", file=sys.stderr)
        print("Run this first: bash data/external/filter_umls_mrsty.bash",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 62)
    print("GAZETTEER BUILD")
    print("=" * 62)
    print(f"\n  source: {args.csv_dir}")
    print(f"  target: {args.db}\n")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode = OFF")
    con.execute("PRAGMA synchronous = OFF")

    create_schema(con)
    loaded = load_csvs(con, args.csv_dir)
    create_indexes(con)
    report_stats(con)
    test_corpus_terms(con)

    con.close()

    # Reopen and verify. File size alone is not proof: a run that dies after
    # create_schema() leaves a valid-looking but empty database behind.
    check = sqlite3.connect(args.db)
    try:
        rows = check.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        tuis = check.execute("SELECT COUNT(DISTINCT tui) FROM terms").fetchone()[0]
    finally:
        check.close()

    size = os.path.getsize(args.db) / 1e6
    if rows != loaded or tuis < 2:
        print(f"\nERROR: verification failed -- {rows:,} rows / {tuis} TUIs on disk, "
              f"expected {loaded:,} rows.", file=sys.stderr)
        print("The database is incomplete; do not use it.", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 62)
    print(f"Gazetteer ready: {args.db}")
    print(f"  {rows:,} terms across {tuis} semantic types, {size:.0f} MB")
    print("=" * 62)


if __name__ == "__main__":
    main()
