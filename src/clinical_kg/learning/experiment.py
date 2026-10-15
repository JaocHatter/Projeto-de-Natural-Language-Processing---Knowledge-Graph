#!/usr/bin/env python3
"""Compare Project 1's hand-set rules with classifiers trained on the gold set.

Both systems score the *same* candidates -- Project 1's generator produces them
and this module never adds or removes one -- so every difference below comes
from the decision function alone.

Validation is leave-one-case-out over the 10 annotated case reports, not a
random split: candidates from one report share entities, sentences and writing
style, so a random split would train and test on the same text and report a
number the model cannot reproduce on a new report. Ten folds also give a spread,
which matters more than a point estimate with 413 examples.

Usage:
    clinical-kg learn-relations                     # full comparison
    clinical-kg learn-relations --ablation          # + feature-group ablation
    clinical-kg learn-relations --task relation     # 8-class instead of binary
"""

import argparse
import collections
import sys

from .dataset import DEFAULT_GROUPS, FEATURE_GROUPS, load_examples
from .models import LogisticRegression, NaiveBayes, OneVsRest, sklearn_model

# Project 1's operating point (relations.core.features.WEIGHTS).
P1_THRESHOLD = 1.0


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def folds(examples):
    """One fold per annotated case report."""
    by_case = collections.defaultdict(list)
    for ex in examples:
        by_case[ex.case_id].append(ex)
    for case_id in sorted(by_case):
        test = by_case[case_id]
        train = [ex for ex in examples if ex.case_id != case_id]
        yield case_id, train, test


def evaluate_binary(predictions):
    """(tp, fp, fn) for 'is there a relation here?'."""
    tp = sum(1 for gold, pred in predictions if gold == 1 and pred == 1)
    fp = sum(1 for gold, pred in predictions if gold == 0 and pred == 1)
    fn = sum(1 for gold, pred in predictions if gold == 1 and pred == 0)
    return tp, fp, fn


def run_p1_baseline(examples, task):
    """Project 1 as it ships: fire when score >= threshold."""
    predictions, typed_hits, fired = [], 0, 0
    for ex in examples:
        pred = 1 if ex.p1_score >= P1_THRESHOLD else 0
        predictions.append((ex.is_relation, pred))
        if pred and ex.is_relation:
            fired += 1
            typed_hits += int(ex.p1_relation == ex.relation)
    typed = typed_hits / fired if fired else 0.0
    return predictions, typed


def run_model(build, examples, task):
    """Leave-one-case-out predictions for one model builder."""
    predictions, per_fold, typed_hits, fired = [], [], 0, 0
    for _, train, test in folds(examples):
        if task == "binary":
            labels = [ex.is_relation for ex in train]
        else:
            labels = [ex.relation or "NONE" for ex in train]
        model = build()
        model.fit([ex.features for ex in train], labels)

        fold_pred = []
        for ex in test:
            raw = model.predict(ex.features)
            if task == "binary":
                pred = int(raw)
            else:
                pred = 0 if raw == "NONE" else 1
                if pred and ex.is_relation:
                    fired += 1
                    typed_hits += int(raw == ex.relation)
            fold_pred.append((ex.is_relation, pred))
        predictions += fold_pred
        per_fold.append(prf(*evaluate_binary(fold_pred))[2])
    typed = typed_hits / fired if fired else None
    return predictions, per_fold, typed


def spread(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance ** 0.5


def report(name, predictions, per_fold=None, typed=None):
    precision, recall, f1 = prf(*evaluate_binary(predictions))
    line = f"  {name:<26} P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}"
    if per_fold:
        mean, std = spread(per_fold)
        line += f"   per-case F1 {mean:.3f}±{std:.3f}"
    if typed is not None:
        line += f"   typed={typed:.3f}"
    print(line)
    return f1


def main():
    ap = argparse.ArgumentParser(description="Learned relation extraction vs Project 1's rules.")
    ap.add_argument("--task", choices=("binary", "relation"), default="binary",
                    help="binary: is there a relation; relation: which of the 8 types")
    ap.add_argument("--ablation", action="store_true", help="evaluate each feature group")
    ap.add_argument("--groups", default=",".join(DEFAULT_GROUPS),
                    help=f"comma-separated subset of {','.join(FEATURE_GROUPS)}")
    args = ap.parse_args()

    groups = tuple(g for g in args.groups.split(",") if g)
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        print(f"ERROR: unknown feature group(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        sys.exit(1)

    print("=" * 74)
    print("RELATION EXTRACTION -- LEARNED vs HAND-SET RULES")
    print("=" * 74)
    examples = load_examples(groups)
    positives = sum(ex.is_relation for ex in examples)
    cases = sorted({ex.case_id for ex in examples})
    features = len({name for ex in examples for name in ex.features})
    print(f"\n  judged candidates : {len(examples)}  ({positives} true, "
          f"{len(examples) - positives} rejected)")
    print(f"  cases (folds)     : {len(cases)}")
    print(f"  feature groups    : {', '.join(groups)}  ->  {features:,} distinct features")
    print(f"  task              : {args.task}")
    print(f"\n  Validation is leave-one-case-out: every number below is measured on a")
    print(f"  case report the model never saw.\n")

    p1_predictions, p1_typed = run_p1_baseline(examples, args.task)
    report("Project 1 (hand-set)", p1_predictions,
           typed=p1_typed if args.task == "relation" else None)

    # Logistic regression is binary; the 8-relation task goes through OneVsRest.
    linear = LogisticRegression if args.task == "binary" else OneVsRest
    builders = [("Naive Bayes (scratch)", NaiveBayes),
                ("Logistic regression", linear)]
    for label, name in (("Linear SVM", "svm"), ("Random forest", "random_forest")):
        if sklearn_model(name) is not None:
            builders.append((label, lambda n=name: sklearn_model(n)))

    for label, build in builders:
        predictions, per_fold, typed = run_model(build, examples, args.task)
        report(label, predictions, per_fold, typed)

    if not any(label.startswith(("Linear", "Random")) for label, _ in builders):
        print("\n  (scikit-learn not installed: SVM and random forest skipped.")
        print("   pip install scikit-learn to include them.)")

    if args.ablation:
        print("\n  Feature-group ablation (logistic regression, binary task):")
        print("  Each row drops one group from the full set; a large drop means the")
        print("  group carries information the others do not.")
        full = None
        for dropped in (None, *groups):
            subset = tuple(g for g in groups if g != dropped)
            if not subset:
                continue
            subset_examples = load_examples(subset)
            predictions, per_fold, _ = run_model(LogisticRegression, subset_examples, "binary")
            _, _, f1 = prf(*evaluate_binary(predictions))
            if dropped is None:
                full = f1
                print(f"    {'all groups':<22} F1={f1:.3f}")
            else:
                print(f"    {'without ' + dropped:<22} F1={f1:.3f}  ({f1 - full:+.3f})")

        print("\n  What the model learned (logistic regression on all data):")
        model = LogisticRegression().fit([ex.features for ex in examples],
                                         [ex.is_relation for ex in examples])
        for name, weight in model.top_features(12):
            print(f"    {weight:+7.3f}  {name}")


if __name__ == "__main__":
    main()
