#!/usr/bin/env python3
"""Interactive annotation of the held-out gold relation set.

Reads  : data/interim/cases_clean.csv, data/processed/entities.csv
Writes : data/interim/gold_relations.csv  (append-only, resumable)

This enumerates every *candidate* pair the extractor considers -- not the edges
it accepted -- because judging only the model's own output measures precision
and leaves recall unmeasurable. Each candidate is judged independently of the
score, so the same annotations support a full precision-recall sweep.

The recall ceiling is therefore the candidate generator's, not 100%: relations
between entities more than pair.max_token_gap tokens apart, or spanning
sentences other than through the Person anchor, are outside the candidate space
by construction. Report that ceiling alongside the numbers.

Gold cases are chosen deterministically, spread across the corpus by entity
count so the sample is not all short cases.

Usage:
    clinical-kg annotate-relations            # start or resume
    clinical-kg annotate-relations --list     # show the gold case IDs
"""

import argparse
import csv
from pathlib import Path

from clinical_kg.paths import (DEFAULT_CASES, DEFAULT_ENTITIES,
                           DEFAULT_GOLD_RELATIONS)

from . import features as rf
from .extract import DEFAULT_FREQ, analyze_case, load_entities, load_frequencies

DEFAULT_GOLD = DEFAULT_GOLD_RELATIONS
DEFAULT_N_CASES = 9

GOLD_COLUMNS = ["case_id", "head_start", "head_end", "tail_start", "tail_end",
                "head_text", "tail_text", "proposed_relation", "gold_relation",
                "gold_assertion", "is_relation"]

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
HEAD_C, TAIL_C = "\033[36m", "\033[33m"


def gold_case_ids(cases, entities, n):
    """Deterministic, spread by entity count so gold is not all short cases."""
    ranked = sorted((len(entities.get(c["case_id"], [])), c["case_id"])
                    for c in cases)
    if n >= len(ranked):
        return [cid for _, cid in ranked]
    step = len(ranked) / n
    return [ranked[int(i * step)][1] for i in range(n)]


def key_of(row):
    return (row["case_id"], str(row["head_start"]), str(row["head_end"]),
            str(row["tail_start"]), str(row["tail_end"]))


def load_done(path):
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {key_of(r) for r in csv.DictReader(f)}


def render(text, rel):
    """Print the sentence with the two arguments highlighted in place."""
    start, end = rel["sent_start"], rel["sent_end"]
    head, tail = rel["head"], rel["tail"]
    marks = []
    if head:
        marks.append((head["start"], head["end"], HEAD_C))
    marks.append((tail["start"], tail["end"], TAIL_C))
    out, last = "", start
    for m_start, m_end, colour in sorted(marks):
        out += text[last:m_start] + colour + BOLD + text[m_start:m_end] + RESET
        last = m_end
    out += text[last:end]
    print("\n" + " ".join(out.split()))


def main():
    ap = argparse.ArgumentParser(description="Annotate the gold relation set.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--n-cases", type=int, default=DEFAULT_N_CASES)
    ap.add_argument("--list", action="store_true", help="print gold case IDs and exit")
    args = ap.parse_args()

    freq = load_frequencies(DEFAULT_FREQ)
    entities = load_entities(args.entities)
    with open(args.cases, newline="", encoding="utf-8", errors="replace") as f:
        cases = list(csv.DictReader(f))
    chosen = gold_case_ids(cases, entities, args.n_cases)

    if args.list:
        for case_id in chosen:
            print(f"{case_id}\t{len(entities.get(case_id, []))} entities")
        return

    done = load_done(args.gold)
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.gold.exists()
    menu = list(rf.RELATIONS)

    print(f"\n{BOLD}Gold annotation{RESET} -- {len(chosen)} cases, "
          f"{len(done)} candidates already judged.")
    print(f"  {HEAD_C}{BOLD}head{RESET}   {TAIL_C}{BOLD}tail{RESET}")
    print("  [y] accept as proposed   [n] not a relation   [1-8] pick relation")
    print("  [a] mark assertion       [s] skip             [q] save and quit\n")

    with open(args.gold, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GOLD_COLUMNS)
        if new_file:
            writer.writeheader()
        for case in cases:
            if case["case_id"] not in chosen:
                continue
            text = case["case_text"]
            candidates = analyze_case(text, entities.get(case["case_id"], []),
                                      freq, threshold=float("-inf"))
            for rel in candidates:
                head, tail = rel["head"], rel["tail"]
                row = {
                    "case_id": case["case_id"],
                    "head_start": head["start"] if head else -1,
                    "head_end": head["end"] if head else -1,
                    "tail_start": tail["start"], "tail_end": tail["end"],
                    "head_text": head["surface_text"] if head else "PATIENT",
                    "tail_text": tail["surface_text"],
                    "proposed_relation": rel["relation"],
                }
                if key_of(row) in done:
                    continue
                render(text, rel)
                print(f"  {HEAD_C}{row['head_text']}{RESET} "
                      f"--[{BOLD}{rel['relation']}{RESET}/{rel['assertion_status']}]--> "
                      f"{TAIL_C}{row['tail_text']}{RESET}")
                print(f"  {DIM}score={rel['score']:.2f}  {rel['rule_path']}{RESET}")
                for i, name in enumerate(menu, 1):
                    print(f"    {DIM}{i}{RESET} {name}", end="")
                print()

                answer = input("  > ").strip().lower()
                if answer == "q":
                    print(f"\nSaved to {args.gold}. Rerun to resume.")
                    return
                if answer == "s":
                    continue
                assertion = rel["assertion_status"]
                if answer.startswith("a"):
                    print(f"    assertions: {', '.join(rf.ASSERTIONS)}")
                    typed = input("    assertion > ").strip().lower()
                    if typed in rf.ASSERTIONS:
                        assertion = typed
                    answer = "y"
                if answer.isdigit() and 1 <= int(answer) <= len(menu):
                    row.update(gold_relation=menu[int(answer) - 1],
                               gold_assertion=assertion, is_relation="1")
                elif answer == "y":
                    row.update(gold_relation=rel["relation"],
                               gold_assertion=assertion, is_relation="1")
                else:
                    row.update(gold_relation="", gold_assertion="", is_relation="0")
                writer.writerow(row)
                f.flush()
                done.add(key_of(row))

    print(f"\nDone. {len(done)} candidates judged -> {args.gold}")


if __name__ == "__main__":
    main()
