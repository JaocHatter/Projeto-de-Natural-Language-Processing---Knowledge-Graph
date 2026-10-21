#!/usr/bin/env python3
"""Sentence, clause and token segmentation with character offsets.

Every span this module returns indexes back into the *original* case text, so
its output joins to data/processed/entities.csv by position. That constraint is
why masking is length-preserving throughout.

The unit that matters downstream is the **clause**, not the paragraph. The
original relation plan proposed pruning entity pairs that sit in different
paragraphs; measured on this corpus that prunes nothing -- 6 of the 50 cases
contain no newline at all, and in the rest a "paragraph" averages ~5 sentences.
"""

import re

# Tokenization MUST match the entity extractor or relation offsets stop lining
# up with entity offsets. Import rather than copy -- see extract_entities.py.
from clinical_kg.extraction.entities import TOKEN_RE, mask_figure_refs

# Periods that do NOT end a sentence. Counted on this corpus: 205 intra-numeric
# decimals ("0.133 kPa"), 18 "Fig.", plus unit and courtesy abbreviations.
ABBREVIATIONS = {
    "fig", "figs", "figure", "tab", "table", "no", "approx", "vs", "etc",
    "dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr",
    "e.g", "i.e", "eg", "ie", "cf", "al",
    # unit abbreviations that end a token before a following capital
    "mg", "kg", "ml", "dl", "l", "g", "mm", "cm", "mmol", "mol", "iu", "u",
    "hr", "hrs", "min", "sec", "wk", "mo", "yr", "hg",
}

SENT_END_RE = re.compile(r"[.!?]+")
# A sentence starts at a capital, a digit, or an opening bracket/quote.
SENT_START_RE = re.compile(r"[A-Z0-9(\[\"']")
CLOSERS = ")]}\"'\u201d\u2019"

# Clause boundaries inside a sentence. Coordinating conjunctions are included
# because "and"/"but" separate clauses just as reliably as punctuation here.
CLAUSE_PUNCT = frozenset(",;:")
CLAUSE_CONJ = frozenset({"and", "but", "or", "however", "whereas", "while",
                         "although", "though", "because", "since", "then"})

WORD_BEFORE_PERIOD_RE = re.compile(r"([A-Za-z][A-Za-z.\-]*)$")


def _is_sentence_end(text: str, idx: int) -> bool:
    """Is the period at `text[idx]` a real sentence terminator?"""
    # Intra-numeric: "0.133", "1.5 cm". Never a boundary.
    if idx > 0 and text[idx - 1].isdigit():
        after = text[idx + 1:idx + 2]
        if after.isdigit():
            return False
    # A known abbreviation immediately before the period.
    match = WORD_BEFORE_PERIOD_RE.search(text[:idx])
    if match and match.group(1).rstrip(".").lower() in ABBREVIATIONS:
        return False
    # Single initial: "J. Smith".
    if match and len(match.group(1)) == 1 and match.group(1).isupper():
        return False
    return True


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split into sentences, returning (start, end) offsets into `text`.

    Figure/table citations are masked first for the same reason the entity
    extractor masks them: "Fig. 1" would otherwise split a sentence in two.
    """
    search = mask_figure_refs(text)
    spans, start = [], 0
    for match in SENT_END_RE.finditer(search):
        idx = match.start()
        if not _is_sentence_end(search, idx):
            continue
        end = match.end()
        # A closing quote or bracket may sit between the period and the space:
        #   ... worsening for the past 2 h." She reported ...
        while end < len(search) and search[end] in CLOSERS:
            end += 1
        rest = search[end:]
        gap = len(rest) - len(rest.lstrip())
        # Require whitespace then a plausible sentence opener; otherwise the
        # period is internal ("U/L.5" never happens, but "St.Mary" does).
        if gap == 0 and rest:
            continue
        nxt = rest[gap:gap + 1]
        if nxt and not SENT_START_RE.match(nxt):
            continue
        if text[start:end].strip():
            spans.append((start, end))
        start = end + gap
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def token_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[str, int, int]]:
    """Tokens as (surface, start, end), offsets absolute into `text`."""
    end = len(text) if end is None else end
    window = mask_figure_refs(text)[start:end]
    return [(m.group(), start + m.start(), start + m.end())
            for m in TOKEN_RE.finditer(window)]


def clause_spans(text: str, sent_start: int, sent_end: int) -> list[tuple[int, int]]:
    """Split one sentence into clauses on punctuation and coordinators.

    Clauses are the co-occurrence window that replaces "same paragraph": two
    entities in the same clause are far likelier to be related than two merely
    in the same sentence. Separators are removed, never emitted as clauses of
    their own.
    """
    seps = [(i, i + 1) for i in range(sent_start, sent_end) if text[i] in CLAUSE_PUNCT]
    seps += [(s, e) for tok, s, e in token_spans(text, sent_start, sent_end)
             if tok.lower() in CLAUSE_CONJ]
    spans, start = [], sent_start
    for sep_start, sep_end in sorted(seps):
        if sep_start > start:
            spans.append((start, sep_start))
        start = max(start, sep_end)
    if sent_end > start:
        spans.append((start, sent_end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def segment(text: str) -> list[dict]:
    """Full layer: sentences, each with its clauses and tokens."""
    out = []
    for s_start, s_end in sentence_spans(text):
        out.append({
            "start": s_start, "end": s_end,
            "tokens": token_spans(text, s_start, s_end),
            "clauses": clause_spans(text, s_start, s_end),
        })
    return out


def clause_of(clauses: list[tuple[int, int]], start: int, end: int) -> int:
    """Index of the clause containing span (start, end), or -1."""
    for i, (c_start, c_end) in enumerate(clauses):
        if c_start <= start and end <= c_end:
            return i
    return -1
