#!/usr/bin/env python3
"""Turn judged candidates into feature vectors.

Reads  : data/interim/cases_clean.csv, data/processed/entities.csv,
         data/interim/gold_relations.csv
Returns: one `Example` per judged candidate, with its feature dict and labels.

The candidate space is Project 1's own generator (`relations.extract.analyze_case`
at threshold -inf), so this module never decides *which* pairs exist -- only how
each pair is described to a classifier. That keeps the Project 1 / Project 2
comparison controlled.

Features are grouped so the experiment can ablate them (`FEATURE_GROUPS`):

    lex     n-grams of the tokens between the two entities, plus the window
            immediately before the head and after the tail. This is the
            mandatory n-gram component, and the only group that sees raw words.
    pos     the same n-grams over coarse POS tags (relations.core.pos). The
            backoff that lets an unseen "reported with" look like a seen
            "presented with": 81.6% of this corpus's bigrams occur exactly once.
    struct  token distance, clause agreement, entity types and their ordering --
            no lexical content at all.
    assoc   corpus-level association of the inter-entity bigrams (PMI with a
            frequency cutoff, and Dunning's log-likelihood ratio). Measured on
            the corpus, not on the labels, so it adds no leakage.
    p1      what Project 1's rules concluded (trigger category, its score). Kept
            in a group of its own precisely so it can be switched off: with it
            on, the classifier can imitate the rules; with it off, it has to
            learn the patterns itself.
"""

import collections
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

from clinical_kg.extraction.entities import TOKEN_RE, mask_figure_refs
from clinical_kg.paths import DEFAULT_CASES, DEFAULT_ENTITIES, DEFAULT_GOLD_RELATIONS
from clinical_kg.relations.core.pos import tag_sequence
from clinical_kg.relations.extract import analyze_case, load_entities

FEATURE_GROUPS = ("lex", "pos", "struct", "assoc", "p1")
DEFAULT_GROUPS = ("lex", "pos", "struct", "assoc")

# Window of tokens kept on the outer side of each entity.
CONTEXT_WINDOW = 2
# Longest n-gram emitted by the lexical/POS groups.
MAX_NGRAM = 3
# An n-gram seen fewer times than this across the corpus is noise, not evidence:
# 81.6% of this corpus's bigram types occur exactly once (see README).
MIN_NGRAM_COUNT = 2


@dataclass
class Example:
    """One judged candidate: its features, its labels and where it came from."""

    case_id: str
    key: tuple
    features: dict[str, float]
    is_relation: int
    relation: str                      # gold relation, "" when is_relation == 0
    p1_score: float                    # Project 1's score, for the baseline
    p1_relation: str                   # Project 1's proposed relation
    head_text: str = ""
    tail_text: str = ""
    rule_path: str = field(default="")
    # Tokens between the two entities, for ngram.ContextScorer. Kept on the
    # example rather than turned into features here because the language models
    # have to be refitted inside every fold.
    context: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Corpus-level association statistics (the `assoc` group)
# --------------------------------------------------------------------------
class Associations:
    """PMI and log-likelihood ratio over corpus bigrams.

    Counted over the case reports *and* the UMLS definitions of the concepts
    they mention, when those have been extracted. PMI asks whether two words
    co-occur more than chance in clinical language generally, and that estimate
    is what this corpus cannot support on its own: 81.6% of its bigram types
    occur exactly once. The definitions roughly double the text (29,718 words
    against 25,407) and are in-domain by construction. Measured, this is worth
    F1 0.730 -> 0.738.

    The same definitions do *not* help the language models in `ngram.py`, which
    is not a contradiction: those have to tell relation-bearing wording apart
    from the rest, and definitions join concepts with generic connectors
    ("and", "or", "of the") that carry no such signal here.

    Raw PMI is unusable on a corpus this size -- every hapax ties at the maximum
    -- so `pmi` returns 0.0 below MIN_PMI_COUNT rather than a huge score for a
    single co-occurrence. LLR (Dunning 1993) is reported alongside because it
    was designed for exactly this regime, where chi-square's expected-count
    assumption fails.
    """

    MIN_PMI_COUNT = 5

    def __init__(self, texts, background_texts=None):
        if background_texts is None:
            background_texts = load_definition_texts()
        tokens = []
        for text in list(texts) + list(background_texts):
            tokens += [m.group().lower() for m in TOKEN_RE.finditer(mask_figure_refs(text))]
        self.unigram = collections.Counter(tokens)
        self.bigram = collections.Counter(zip(tokens, tokens[1:]))
        self.n = max(sum(self.bigram.values()), 1)

    def pmi(self, left: str, right: str) -> float:
        count = self.bigram.get((left, right), 0)
        if count < self.MIN_PMI_COUNT:
            return 0.0
        joint = count / self.n
        marginal = (self.unigram[left] / self.n) * (self.unigram[right] / self.n)
        return math.log2(joint / marginal) if marginal else 0.0

    def llr(self, left: str, right: str) -> float:
        a = self.bigram.get((left, right), 0)
        if not a:
            return 0.0
        b, d = self.unigram[left] - a, self.unigram[right] - a
        e = self.n - a - b - d
        if min(b, d, e) < 0:
            return 0.0

        def entropy(*counts):
            total = sum(counts)
            return sum(c * math.log(c / total) for c in counts if c > 0) if total else 0.0

        return 2 * (entropy(a, b, d, e) - entropy(a + b, d + e) - entropy(a + d, b + e))


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def _ngrams(items, prefix, out, counts=None):
    """Emit 1..MAX_NGRAM-grams of `items` as binary features."""
    for size in range(1, MAX_NGRAM + 1):
        for i in range(len(items) - size + 1):
            gram = tuple(items[i:i + size])
            if counts is not None and size > 1 and counts.get(gram, 0) < MIN_NGRAM_COUNT:
                continue
            out[f"{prefix}:{'_'.join(gram)}"] = 1.0


def _bucket(gap: int) -> str:
    return "0" if gap == 0 else "1_3" if gap <= 3 else "4_10" if gap <= 10 else "gt_10"


def candidate_features(text, rel, assoc, ngram_counts, groups=DEFAULT_GROUPS):
    """Feature dict for one candidate produced by relations.extract."""
    head, tail = rel["head"], rel["tail"]
    out: dict[str, float] = {}

    start = head["end"] if head else rel["sent_start"]
    between = [m.group() for m in TOKEN_RE.finditer(text[start:tail["start"]])]
    before = [m.group() for m in TOKEN_RE.finditer(
        text[max(rel["sent_start"], start - 60):start])][-CONTEXT_WINDOW:]
    after = [m.group() for m in TOKEN_RE.finditer(
        text[tail["end"]:min(rel["sent_end"], tail["end"] + 60)])][:CONTEXT_WINDOW]

    low = [w.lower() for w in between]
    if "lex" in groups:
        _ngrams(low, "btw", out, ngram_counts)
        _ngrams([w.lower() for w in before], "pre", out)
        _ngrams([w.lower() for w in after], "post", out)
        if low:
            out[f"first:{low[0]}"] = 1.0
            out[f"last:{low[-1]}"] = 1.0

    if "pos" in groups:
        _ngrams(tag_sequence(between), "pos_btw", out)
        _ngrams(tag_sequence(before), "pos_pre", out)
        _ngrams(tag_sequence(after), "pos_post", out)

    if "struct" in groups:
        head_type = head["entity_type"] if head else "Person"
        out[f"pair:{head_type}_{tail['entity_type']}"] = 1.0
        out[f"head_type:{head_type}"] = 1.0
        out[f"tail_type:{tail['entity_type']}"] = 1.0
        out[f"gap:{_bucket(rel['token_gap'])}"] = 1.0
        out["n_tokens_between"] = float(min(len(between), 20)) / 20.0
        out["same_clause"] = 1.0 if rel["same_clause"] else 0.0
        out["person_anchored"] = 1.0 if head is None else 0.0
        out["has_comma"] = 1.0 if "," in text[start:tail["start"]] else 0.0

    if "assoc" in groups and len(low) >= 2:
        pmis = [assoc.pmi(a, b) for a, b in zip(low, low[1:])]
        llrs = [assoc.llr(a, b) for a, b in zip(low, low[1:])]
        out["pmi_max"] = max(pmis) / 12.0
        out["pmi_mean"] = (sum(pmis) / len(pmis)) / 12.0
        out["llr_max"] = min(max(llrs) / 300.0, 1.0)

    if "p1" in groups:
        trigger = rel["trigger"]
        out[f"p1_trig_cat:{trigger['category'] if trigger else 'NONE'}"] = 1.0
        out["p1_has_trigger"] = 1.0 if trigger else 0.0
        out["p1_score"] = max(min(rel["score"], 8.0), -8.0) / 8.0

    return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_definition_texts(path: Path = None) -> list[str]:
    """UMLS definitions of this corpus's own CUIs, or [] when never extracted."""
    from .ngram import DEFAULT_DEFINITIONS
    path = DEFAULT_DEFINITIONS if path is None else path
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [row["definition"] for row in csv.DictReader(f)
                if row.get("definition", "").strip()]


def context_tokens(text: str, start: int, end: int) -> list[str]:
    """The words between two entities, as the language models see them."""
    from .ngram import tokenize
    return tokenize(text[start:end])


def _gold_key(row):
    return (row["case_id"], str(row["head_start"]), str(row["head_end"]),
            str(row["tail_start"]), str(row["tail_end"]))


def load_examples(groups=DEFAULT_GROUPS, cases_path: Path = DEFAULT_CASES,
                  entities_path: Path = DEFAULT_ENTITIES,
                  gold_path: Path = DEFAULT_GOLD_RELATIONS) -> list[Example]:
    """Every judged candidate, as an Example. Order is deterministic."""
    with open(cases_path, newline="", encoding="utf-8", errors="replace") as f:
        cases = list(csv.DictReader(f))
    with open(gold_path, newline="", encoding="utf-8") as f:
        gold = {_gold_key(r): r for r in csv.DictReader(f)}
    entities = load_entities(entities_path)

    assoc = Associations(c["case_text"] for c in cases)
    ngram_counts = collections.Counter()
    for case in cases:
        tokens = [m.group().lower() for m in TOKEN_RE.finditer(mask_figure_refs(case["case_text"]))]
        for size in range(2, MAX_NGRAM + 1):
            ngram_counts.update(tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1))

    gold_cases = {key[0] for key in gold}
    # Keyed, not appended: relations.extract can emit the same (head, tail) pair
    # twice when `orient` swaps an edge into a key another candidate already
    # produced. relations.tools.evaluate collapses those the same way, so both
    # systems are scored over exactly the same 413 candidates.
    collected: dict[tuple, Example] = {}
    for case in cases:
        if case["case_id"] not in gold_cases:
            continue
        text = case["case_text"]
        for rel in analyze_case(text, entities.get(case["case_id"], []), {},
                                threshold=float("-inf")):
            head, tail = rel["head"], rel["tail"]
            key = (case["case_id"], str(head["start"] if head else -1),
                   str(head["end"] if head else -1), str(tail["start"]), str(tail["end"]))
            row = gold.get(key)
            if row is None:          # candidate nobody judged: not evaluable
                continue
            start = head["end"] if head else rel["sent_start"]
            collected[key] = Example(
                case_id=case["case_id"], key=key,
                context=context_tokens(text, start, tail["start"]),
                features=candidate_features(text, rel, assoc, ngram_counts, groups),
                is_relation=int(row["is_relation"]),
                relation=row["gold_relation"],
                p1_score=rel["score"], p1_relation=rel["relation"],
                head_text=head["surface_text"] if head else "PATIENT",
                tail_text=tail["surface_text"], rule_path=rel["rule_path"])
    return list(collected.values())
