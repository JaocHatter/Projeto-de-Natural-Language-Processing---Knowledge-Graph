#!/usr/bin/env python3
"""Extract relations over the whole corpus with the learned model.

Reads  : data/interim/cases_clean.csv, data/processed/entities.csv,
         data/interim/gold_relations.csv (the 413 judged candidates it trains on)
Writes : data/processed/relations_ml.csv, in the same schema Project 1 writes,
         so `graph.model.build_graph` and `export-graph` consume it unchanged.

This is the Project 2 counterpart of `relations.extract`. The candidate
generator, the assertion scoping and the output schema are Project 1's; what
changes is the decision: instead of comparing a hand-set score against a
threshold, a classifier trained on the judged candidates decides both whether
the pair is a relation and which of the eight types it is.

Usage:
    clinical-kg extract-relations-ml
    clinical-kg extract-relations-ml --model svm --out /tmp/relations_ml.csv
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

from clinical_kg.paths import (DEFAULT_CASES, DEFAULT_ENTITIES,
                               DEFAULT_GOLD_RELATIONS, PROJECT_ROOT)
from clinical_kg.relations.extract import (CSV_COLUMNS, analyze_case,
                                           load_entities, to_row)

from .dataset import (Associations, DEFAULT_GROUPS, FEATURE_GROUPS,
                      candidate_features, context_tokens, load_examples)
from .models import LogisticRegression, NaiveBayes, OneVsRest, sklearn_model
from .ngram import ContextScorer

DEFAULT_OUT = PROJECT_ROOT / "data/processed/relations_ml.csv"
NO_RELATION = "NONE"


def build_model(name: str):
    """Model by name, or None when scikit-learn is needed but missing."""
    if name == "logistic":
        return OneVsRest()
    if name == "naive_bayes":
        return NaiveBayes()
    return sklearn_model(name)


def train(model_name: str, groups) -> tuple:
    """Fit on every judged candidate.

    Returns (model, scorer, n), where `scorer` is the fitted pair of n-gram
    language models the classifier expects to see a score from. The training
    values for that feature come from an inner leave-one-case-out pass: a
    language model finds a context it has already read far likelier than a new
    one, so fitting the classifier on self-scored contexts teaches it to trust
    the feature more than it deserves.
    """
    examples = load_examples(groups)
    model = build_model(model_name)
    if model is None:
        raise SystemExit(f"ERROR: model {model_name!r} needs scikit-learn "
                         "(pip install scikit-learn), or use --model logistic")

    fit_scorer = lambda subset: ContextScorer().fit(
        [e.context for e in subset if e.is_relation],
        [e.context for e in subset if not e.is_relation])
    scores = {}
    for held_out in sorted({e.case_id for e in examples}):
        inner = fit_scorer([e for e in examples if e.case_id != held_out])
        scores.update({e.key: inner.score(e.context)
                       for e in examples if e.case_id == held_out})
    scorer = fit_scorer(examples)

    labels = [ex.relation if ex.is_relation else NO_RELATION for ex in examples]
    model.fit([{**ex.features, "lm_score": scores[ex.key]} for ex in examples], labels)
    return model, scorer, len(examples)


def main():
    ap = argparse.ArgumentParser(
        description="Extract relations with the learned classifier (Project 2).")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD_RELATIONS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="logistic",
                    choices=("logistic", "naive_bayes", "svm", "random_forest"))
    ap.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    args = ap.parse_args()

    groups = tuple(g for g in args.groups.split(",") if g)
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        print(f"ERROR: unknown feature group(s): {', '.join(sorted(unknown))}",
              file=sys.stderr)
        sys.exit(1)
    for path in (args.cases, args.entities, args.gold):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            sys.exit(1)

    print("=" * 66)
    print("RELATION EXTRACTION -- LEARNED MODEL (Project 2)")
    print("=" * 66)
    print(f"\n  model     : {args.model}")
    print(f"  features  : {', '.join(groups)}")
    print(f"  out       : {args.out}\n")

    model, scorer, n_train = train(args.model, groups)
    print(f"  trained on {n_train} judged candidates\n")

    with open(args.cases, newline="", encoding="utf-8", errors="replace") as f:
        cases = list(csv.DictReader(f))
    entities = load_entities(args.entities)
    assoc = Associations(c["case_text"] for c in cases)
    ngram_counts = _corpus_ngrams(cases)

    by_relation, by_assertion = collections.Counter(), collections.Counter()
    considered = total = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for case in cases:
            text = case.get("case_text") or ""
            for rel in analyze_case(text, entities.get(case["case_id"], []), {},
                                    threshold=float("-inf")):
                considered += 1
                feats = candidate_features(text, rel, assoc, ngram_counts, groups)
                start = rel["head"]["end"] if rel["head"] else rel["sent_start"]
                feats["lm_score"] = scorer.score(
                    context_tokens(text, start, rel["tail"]["start"]))
                predicted = model.predict(feats)
                if predicted == NO_RELATION:
                    continue
                # Keep Project 1's arguments, offsets, trigger and assertion
                # scoping; replace only the relation the edge asserts.
                rel = {**rel, "relation": predicted}
                row = to_row(case, rel)
                assert text[rel["tail"]["start"]:rel["tail"]["end"]] == row["tail_text"]
                writer.writerow(row)
                total += 1
                by_relation[predicted] += 1
                by_assertion[rel["assertion_status"]] += 1

    print(f"  candidates considered : {considered:,}")
    print(f"  relations kept        : {total:,}  "
          f"({100 * total / max(considered, 1):.0f}% of candidates, "
          f"{total / max(len(cases), 1):.0f} per case)")
    print("\n  By relation:")
    for name, count in by_relation.most_common():
        print(f"    {name:<18} {count:>6,}  {'#' * min(40, count // 15)}")
    print("\n  By assertion status (Project 1's scoping, unchanged):")
    for name, count in by_assertion.most_common():
        print(f"    {name:<18} {count:>6,}")
    print(f"\n  Written to {args.out}")


def _corpus_ngrams(cases):
    """Corpus n-gram counts, as dataset.load_examples builds them."""
    from clinical_kg.extraction.entities import TOKEN_RE, mask_figure_refs
    from .dataset import MAX_NGRAM
    counts = collections.Counter()
    for case in cases:
        tokens = [m.group().lower()
                  for m in TOKEN_RE.finditer(mask_figure_refs(case["case_text"]))]
        for size in range(2, MAX_NGRAM + 1):
            counts.update(tuple(tokens[i:i + size])
                          for i in range(len(tokens) - size + 1))
    return counts


if __name__ == "__main__":
    main()
