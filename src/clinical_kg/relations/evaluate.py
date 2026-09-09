#!/usr/bin/env python3
"""Score the relation extractor against the held-out gold set.

Reads  : data/interim/gold_relations.csv (from annotate_relations.py)
Prints : precision/recall/F1, a threshold sweep, per-relation breakdown and an
         ablation table.

The weights in features.WEIGHTS are set by hand, so no claim that they
are optimal is available. The defensible claims are the ones this script
produces: how the model scores on cases never used for tuning, and how much
each design component actually contributes.

Two ceilings are reported honestly:
  * candidate recall -- relations outside the generator's window (more than
    pair.max_token_gap tokens apart, or cross-sentence without a Person anchor)
    can never be found, whatever the threshold.
  * gold coverage -- only candidates a human actually judged are scored.

Usage:
    clinical-kg evaluate-relations
    clinical-kg evaluate-relations --no-ablation
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

from clinical_kg.paths import DEFAULT_CASES, DEFAULT_ENTITIES
from . import features as rf
from .annotate import DEFAULT_GOLD
from .extract import DEFAULT_FREQ, analyze_case, load_entities, load_frequencies

SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def load_gold(path):
    with open(path, newline="", encoding="utf-8") as f:
        gold = {}
        for row in csv.DictReader(f):
            key = (row["case_id"], int(row["head_start"]), int(row["head_end"]),
                   int(row["tail_start"]), int(row["tail_end"]))
            gold[key] = row
    return gold


def predictions(cases, entities, freq, gold_cases):
    """Every candidate with its score, for the gold cases only."""
    out = {}
    for case in cases:
        if case["case_id"] not in gold_cases:
            continue
        text = case["case_text"]
        for rel in analyze_case(text, entities.get(case["case_id"], []), freq,
                                threshold=float("-inf")):
            head, tail = rel["head"], rel["tail"]
            key = (case["case_id"], head["start"] if head else -1,
                   head["end"] if head else -1, tail["start"], tail["end"])
            out[key] = rel
    return out


def score(gold, preds, threshold):
    """Detection, typed and assertion scores at one threshold."""
    tp = fp = fn = typed_tp = assert_ok = 0
    per_relation = collections.defaultdict(lambda: [0, 0, 0])
    for key, row in gold.items():
        is_gold = row["is_relation"] == "1"
        rel = preds.get(key)
        fired = rel is not None and rel["score"] >= threshold
        if fired and is_gold:
            tp += 1
            per_relation[row["gold_relation"]][0] += 1
            if rel["relation"] == row["gold_relation"]:
                typed_tp += 1
            if rel["assertion_status"] == (row["gold_assertion"] or "affirmed"):
                assert_ok += 1
        elif fired and not is_gold:
            fp += 1
            per_relation[rel["relation"]][1] += 1
        elif not fired and is_gold:
            fn += 1
            per_relation[row["gold_relation"]][2] += 1
    return {"tp": tp, "fp": fp, "fn": fn, "typed_tp": typed_tp,
            "assert_ok": assert_ok, "per_relation": per_relation}


def report(gold, preds, threshold, label=""):
    result = score(gold, preds, threshold)
    precision, recall, f1 = prf(result["tp"], result["fp"], result["fn"])
    typed_p = result["typed_tp"] / result["tp"] if result["tp"] else 0.0
    if label:
        print(f"  {label:<28} P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
              f"typed={typed_p:.3f}")
    return precision, recall, f1, result


def main():
    ap = argparse.ArgumentParser(description="Evaluate relation extraction.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    if not args.gold.exists():
        print(f"ERROR: no gold set at {args.gold}\n"
              f"Create it first:  clinical-kg annotate-relations",
              file=sys.stderr)
        sys.exit(1)

    gold = load_gold(args.gold)
    judged = sum(1 for r in gold.values() if r["is_relation"] == "1")
    gold_cases = {key[0] for key in gold}
    freq = load_frequencies(DEFAULT_FREQ)
    entities = load_entities(args.entities)
    with open(args.cases, newline="", encoding="utf-8", errors="replace") as f:
        cases = list(csv.DictReader(f))
    preds = predictions(cases, entities, freq, gold_cases)

    print("=" * 68)
    print("RELATION EXTRACTION -- EVALUATION")
    print("=" * 68)
    print(f"\n  gold cases      : {len(gold_cases)}  ({', '.join(sorted(gold_cases))})")
    print(f"  candidates judged: {len(gold)}  ({judged} true relations, "
          f"{len(gold) - judged} rejected)")
    missing = [k for k in gold if k not in preds]
    if missing:
        print(f"  WARNING: {len(missing)} gold rows have no matching candidate "
              f"(entities.csv changed since annotation?)")

    print("\n  Threshold sweep (detection):")
    print(f"    {'thresh':>7}  {'P':>6}  {'R':>6}  {'F1':>6}  {'typed':>6}  "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    best = None
    for threshold in SWEEP:
        precision, recall, f1, result = report(gold, preds, threshold)
        typed_p = result["typed_tp"] / result["tp"] if result["tp"] else 0.0
        marker = " *" if best is None or f1 > best[1] else ""
        if best is None or f1 > best[1]:
            best = (threshold, f1)
        print(f"    {threshold:>7.1f}  {precision:>6.3f}  {recall:>6.3f}  "
              f"{f1:>6.3f}  {typed_p:>6.3f}  {result['tp']:>5} "
              f"{result['fp']:>5} {result['fn']:>5}{marker}")

    operating = rf.WEIGHTS["decide.threshold"]
    print(f"\n  At the configured threshold ({operating}):")
    precision, recall, f1, result = report(gold, preds, operating)
    print(f"    detection        P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
    if result["tp"]:
        print(f"    relation type    {result['typed_tp'] / result['tp']:.3f} "
              f"correct on detected edges")
        print(f"    assertion status {result['assert_ok'] / result['tp']:.3f} "
              f"correct on detected edges")

    print("\n  Per relation (at the configured threshold):")
    print(f"    {'relation':<18} {'TP':>5} {'FP':>5} {'FN':>5}   {'P':>6} {'R':>6} {'F1':>6}")
    for name, (tp, fp, fn) in sorted(result["per_relation"].items()):
        if not name:
            continue
        p_, r_, f_ = prf(tp, fp, fn)
        print(f"    {name:<18} {tp:>5} {fp:>5} {fn:>5}   {p_:>6.3f} {r_:>6.3f} {f_:>6.3f}")

    if not args.no_ablation:
        print("\n  Ablation -- what each component buys (at the configured threshold):")
        base = f1
        print(f"    {'full model':<28} F1={base:.3f}")
        for switch, description in [
            ("pos_backoff", "without POS backoff"),
            ("coordination", "without coordination"),
            ("negation", "without assertion status"),
            ("transitions", "without CRF transitions"),
        ]:
            rf.ABLATE[switch] = True
            ablated = predictions(cases, entities, freq, gold_cases)
            _, _, f1_a, res_a = report(gold, ablated, operating)
            rf.ABLATE[switch] = False
            delta = f1_a - base
            extra = ""
            if switch == "negation" and res_a["tp"]:
                extra = (f"   assertion acc "
                         f"{res_a['assert_ok'] / res_a['tp']:.3f}")
            print(f"    {description:<28} F1={f1_a:.3f}  ({delta:+.3f}){extra}")

    print("\n  Ceilings: recall is bounded by the candidate generator "
          f"(max {rf.WEIGHTS['pair.max_token_gap']} tokens apart,")
    print("  intra-sentence or Person-anchored) and by what was actually judged.")


if __name__ == "__main__":
    main()
