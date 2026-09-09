#!/usr/bin/env python3
"""Relation extraction: candidate generation, attachment, typing and output.

Reads  : data/interim/cases_clean.csv, data/processed/entities.csv
Writes : data/processed/relations.csv

Per sentence:
  1. `crf.viterbi` tags the token sequence and `crf.decode_triggers` groups the
     B/I runs into trigger phrases.
  2. Each trigger attaches to its nearest entity on each side within the clause,
     falling back to the Person root when the clause subject is the patient and
     no other-subject cue is present.
  3. Coordinated lists expand: siblings inherit the head relation of the list's
     first member, which turns "presented with fatigue, swollen abdomen,
     decreased appetite, and weight loss" into four symptom edges rather than
     one. The original plan discarded any pair with a comma between it, which
     would have thrown away 34% of adjacent pairs.
  4. Assertion status is assigned by NegEx-style forward scope.

Usage:
    clinical-kg extract-relations
    clinical-kg extract-relations --threshold 2.0 --explain PMC10106591_01
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

from clinical_kg.paths import (DEFAULT_CASES, DEFAULT_ENTITIES, DEFAULT_RELATIONS,
                           DEFAULT_TOKEN_FREQ)

from . import features as rf
from .crf import decode_triggers, emissions, viterbi
from .pos import tag_token
from .text_layer import clause_of, segment

DEFAULT_OUT = DEFAULT_RELATIONS
DEFAULT_FREQ = DEFAULT_TOKEN_FREQ

CSV_COLUMNS = ["case_id", "article_id", "relation", "assertion_status",
               "head_kind", "head_start", "head_end", "head_cui",
               "head_entity_type", "head_text",
               "tail_start", "tail_end", "tail_cui", "tail_entity_type", "tail_text",
               "trigger_text", "trigger_start", "trigger_end", "trigger_category",
               "sent_start", "sent_end", "token_gap", "same_clause",
               "score", "rule_path"]


def load_frequencies(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["token"]: int(r["frequency"]) for r in csv.DictReader(f)}

# --------------------------------------------------------------------------
# Assertion status (NegEx/ConText-style forward scope)
# --------------------------------------------------------------------------
def assertion_states(tokens, spans, clauses, text):
    """Assertion status active at each token index.

    A cue opens a scope that runs forward until a clause boundary or a
    termination cue ("but", "however"). Without this the corpus's 293 negation
    cues are silently dropped and the graph asserts findings the text denies:
    "revealed abdominal pain in RIF ... but no rebound tenderness".
    """
    if rf.ABLATE["negation"]:
        return ["affirmed"] * len(tokens)
    PRECEDENCE = {"affirmed": 0, "hedged": 1, "historical": 2,
                  "negated": 3, "family": 4}
    states, state, clause_idx = [], "affirmed", -1
    for i, token in enumerate(tokens):
        here = clause_of(clauses, spans[i][1], spans[i][2])
        if here != clause_idx:
            clause_idx, state = here, "affirmed"
        low = token.lower()
        if low in rf.SCOPE_TERMINATORS:
            state = "affirmed"
        new = None
        if low in rf.FAMILY_CUES:
            new = "family"
        elif low in rf.NEGATION_CUES:
            new = "negated"
        elif low in rf.HISTORICAL_CUES:
            new = "historical"
        elif low in rf.HEDGE_CUES:
            new = "hedged"
        if new and PRECEDENCE[new] >= PRECEDENCE[state]:
            state = new
        states.append(state)
    return states


def assertion_for(states, spans, start, end):
    """Status of the token that opens an entity span."""
    for i, (_, t_start, t_end) in enumerate(spans):
        if t_start >= start and t_end <= end:
            return states[i]
    return "affirmed"


# --------------------------------------------------------------------------
# Coordination
# --------------------------------------------------------------------------
COORD_FILLERS = frozenset({",", ";", "and", "or", "as", "well", "with", "plus", "&"})


def coordinate_siblings(entities, text):
    """Map entity index -> index of the list head it coordinates with.

    The original plan discarded any pair with a comma between it. Measured on
    this corpus that throws away 34% of adjacent pairs (1,055 of 3,084) and
    with them the single most productive pattern in clinical narrative:
    "fatigue, swollen abdomen, decreased appetite, and weight loss".
    """
    head_of = {}
    if rf.ABLATE["coordination"]:
        return head_of
    for i in range(len(entities) - 1):
        left, right = entities[i], entities[i + 1]
        between = text[left["end"]:right["start"]]
        words = [w.strip(",;") for w in between.replace(",", " , ").split()]
        if any(w and w.lower() not in COORD_FILLERS for w in words):
            continue
        if len(between) > 30:
            continue
        # Compatible arguments: same project type, or both clinical concepts.
        if left["entity_type"] != right["entity_type"]:
            continue
        head_of[i + 1] = head_of.get(i, i)
    return head_of


# --------------------------------------------------------------------------
# Attachment
# --------------------------------------------------------------------------
def patient_subject(tokens, upto):
    """Is the clause subject the patient, with no other-subject cue?"""
    window = [t.lower() for t in tokens[:upto]]
    if any(w in rf.FAMILY_CUES for w in window):
        return False
    return any(w in rf.PATIENT_CUES for w in window)


def score_pair(head, tail, gap, same_clause, intervening, trigger_strength, has_trigger):
    """Pair-level potential added to the trigger score. Returns (score, names)."""
    W = rf.WEIGHTS
    total, names = W["score.trigger_weight"] * trigger_strength, []
    if has_trigger:
        names.append("score.trigger_weight")
    else:
        total += W["pair.no_trigger"]; names.append("pair.no_trigger")
    bucket = ("pair.dist_0" if gap == 0 else "pair.dist_1_3" if gap <= 3
              else "pair.dist_4_10" if gap <= 10 else "pair.dist_gt_10")
    total += W[bucket]; names.append(bucket)
    key = "pair.same_clause" if same_clause else "pair.cross_clause"
    total += W[key]; names.append(key)
    if intervening:
        total += W["pair.per_intervening"] * intervening
        names.append("pair.per_intervening")
    head_type = head["entity_type"] if head else "Person"
    prior = rf.pair_prior(head_type, tail["entity_type"])
    if prior:
        total += W["pair.type_prior"] * prior
        names.append("pair.type_prior")
    if head is None:
        total += W["pair.patient_anchor"]; names.append("pair.patient_anchor")
    return total, names


def analyze_sentence(text, sent, entities, freq, threshold=None):
    """Extract relations from one sentence. Returns a list of relation dicts.

    `threshold` overrides decide.threshold; pass float("-inf") to get every
    candidate the generator considered, which is what the annotation tool and
    the precision-recall sweep need -- scoring only the edges the model already
    accepted would make recall unmeasurable.
    """
    W = rf.WEIGHTS
    cutoff = W["decide.threshold"] if threshold is None else threshold
    spans = sent["tokens"]
    if not spans:
        return []
    tokens = [s[0] for s in spans]
    covered = [any(e["start"] <= s < e["end"] or e["start"] < t <= e["end"]
                   for e in entities) for _, s, t in spans]
    pos, prev = [], ""
    for token in tokens:
        prev = tag_token(token, prev)
        pos.append(prev)

    emit, fired = emissions(tokens, pos, covered, freq)
    tags = viterbi(emit)
    triggers = decode_triggers(tokens, spans, tags)
    states = assertion_states(tokens, spans, sent["clauses"], text)
    head_of = coordinate_siblings(entities, text)

    def clause_idx(ent):
        return clause_of(sent["clauses"], ent["start"], ent["end"])

    def tokens_between(a_end, b_start):
        return sum(1 for _, s, e in spans if a_end <= s and e <= b_start)

    out = []
    seen = set()

    def emit_relation(head, tail, trigger, score, names, inherited=False):
        relation = rf.relation_for(trigger["category"] if trigger else "",
                                   tail["entity_type"])
        head, tail, relation, swapped = rf.orient(head, tail, relation)
        key = (head["start"] if head else -1, head["end"] if head else -1,
               tail["start"], tail["end"])
        if key in seen:
            return
        seen.add(key)
        # A BodyPart behind a locative preposition is a location, whatever the
        # trigger was: "lesion involving body and ramus of right mandible".
        if tail["entity_type"] == "BodyPart":
            before = text[(head["end"] if head else tail["start"]):tail["start"]].lower()
            if any(f" {p} " in f" {before} " for p in rf.LOCATIVE_PREPS):
                relation = "LOCATED_IN"
        path = list(names) + (["pair.coordinate"] if inherited else [])
        if swapped:
            path.append("orient.swapped")
        out.append({
            "relation": relation,
            "assertion_status": assertion_for(states, spans, tail["start"], tail["end"]),
            "head": head, "tail": tail, "trigger": trigger,
            "sent_start": sent["start"], "sent_end": sent["end"],
            "token_gap": tokens_between(head["end"] if head else sent["start"],
                                        tail["start"]),
            "same_clause": head is not None and clause_idx(head) == clause_idx(tail),
            "score": score, "rule_path": "|".join(path),
        })

    for trigger in triggers:
        strength = sum(emit[k]["I-TRIG" if k > trigger["i"] else "B-TRIG"]
                       for k in range(trigger["i"], trigger["j"]))
        strength /= (trigger["j"] - trigger["i"])
        left = [e for e in entities if e["end"] <= trigger["start"]]
        right = [e for e in entities if e["start"] >= trigger["end"]]
        if not right:
            continue
        tail = min(right, key=lambda e: e["start"])
        head = max(left, key=lambda e: e["end"]) if left else None
        if head is None and not patient_subject(tokens, trigger["i"]):
            continue
        intervening = sum(1 for e in entities
                          if head and head["end"] <= e["start"] < tail["start"]
                          and e is not head)
        if intervening > W["pair.max_intervening"]:
            continue
        gap = tokens_between(head["end"] if head else sent["start"], tail["start"])
        if gap > W["pair.max_token_gap"]:
            continue
        same = head is not None and clause_idx(head) == clause_idx(tail)
        score, names = score_pair(head, tail, gap, same, intervening, strength, True)
        if score < cutoff:
            continue
        emit_relation(head, tail, trigger, score, names)
        # Coordinated siblings inherit this relation.
        tail_idx = entities.index(tail)
        for idx, list_head in head_of.items():
            if list_head == tail_idx and idx != tail_idx:
                emit_relation(head, entities[idx], trigger,
                              score + W["pair.coordinate"], names, inherited=True)

    # Adjacent pairs with no trigger: modifier and coordinate links.
    for i in range(len(entities) - 1):
        head, tail = entities[i], entities[i + 1]
        gap = tokens_between(head["end"], tail["start"])
        if gap > W["pair.max_token_gap"]:
            continue
        same = clause_idx(head) == clause_idx(tail)
        is_coord = head_of.get(i + 1) is not None
        score, names = score_pair(head, tail, gap, same, 0, 0.0, False)
        if is_coord:
            score += W["pair.coordinate"]; names.append("pair.coordinate")
        if score < cutoff:
            continue
        key = (head["start"], head["end"], tail["start"], tail["end"])
        if key in seen:
            continue
        seen.add(key)
        relation = ("COORDINATE_WITH" if is_coord
                    else "LOCATED_IN" if tail["entity_type"] == "BodyPart"
                    else rf.relation_for("", tail["entity_type"]))
        head, tail, relation, swapped = rf.orient(head, tail, relation)
        if swapped:
            names.append("orient.swapped")
        out.append({
            "relation": relation,
            "assertion_status": assertion_for(states, spans, tail["start"], tail["end"]),
            "head": head, "tail": tail, "trigger": None,
            "sent_start": sent["start"], "sent_end": sent["end"],
            "token_gap": gap, "same_clause": same,
            "score": score, "rule_path": "|".join(names),
        })
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def load_entities(path: Path) -> dict[str, list[dict]]:
    by_case = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["start"], row["end"] = int(row["start"]), int(row["end"])
            by_case[row["case_id"]].append(row)
    for rows in by_case.values():
        rows.sort(key=lambda r: (r["start"], r["end"]))
    return by_case


def analyze_case(text: str, entities: list[dict], freq: dict[str, int],
                 threshold=None) -> list[dict]:
    out = []
    for sent in segment(text):
        inside = [e for e in entities
                  if e["start"] >= sent["start"] and e["end"] <= sent["end"]]
        if len(inside) < 1:
            continue
        out.extend(analyze_sentence(text, sent, inside, freq, threshold))
    return out


def to_row(case: dict, rel: dict) -> dict:
    head, tail, trigger = rel["head"], rel["tail"], rel["trigger"]
    return {
        "case_id": case["case_id"], "article_id": case.get("article_id", ""),
        "relation": rel["relation"], "assertion_status": rel["assertion_status"],
        "head_kind": "Concept" if head else "Person",
        "head_start": head["start"] if head else "",
        "head_end": head["end"] if head else "",
        "head_cui": head["cui"] if head else "",
        "head_entity_type": head["entity_type"] if head else "Person",
        "head_text": head["surface_text"] if head else "",
        "tail_start": tail["start"], "tail_end": tail["end"],
        "tail_cui": tail["cui"], "tail_entity_type": tail["entity_type"],
        "tail_text": tail["surface_text"],
        "trigger_text": " ".join(trigger["words"]) if trigger else "",
        "trigger_start": trigger["start"] if trigger else "",
        "trigger_end": trigger["end"] if trigger else "",
        "trigger_category": trigger["category"] if trigger else "",
        "sent_start": rel["sent_start"], "sent_end": rel["sent_end"],
        "token_gap": rel["token_gap"], "same_clause": rel["same_clause"],
        "score": round(rel["score"], 3), "rule_path": rel["rule_path"],
    }


def main():
    ap = argparse.ArgumentParser(description="Extract relations from case reports.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--freq", type=Path, default=DEFAULT_FREQ)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override decide.threshold from the weight table")
    ap.add_argument("--explain", metavar="CASE_ID",
                    help="print scored relations for one case instead of writing")
    args = ap.parse_args()

    for path in (args.cases, args.entities):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            sys.exit(1)
    if args.threshold is not None:
        rf.WEIGHTS["decide.threshold"] = args.threshold

    freq = load_frequencies(args.freq)
    by_case = load_entities(args.entities)
    with open(args.cases, newline="", encoding="utf-8", errors="replace") as f:
        cases = list(csv.DictReader(f))

    if args.explain:
        case = next((c for c in cases if c["case_id"] == args.explain), None)
        if case is None:
            print(f"ERROR: no case {args.explain}", file=sys.stderr)
            sys.exit(1)
        text = case["case_text"]
        for rel in analyze_case(text, by_case.get(args.explain, []), freq):
            head = rel["head"]["surface_text"] if rel["head"] else "PATIENT"
            trig = " ".join(rel["trigger"]["words"]) if rel["trigger"] else "-"
            print(f"  {rel['score']:>6.2f}  {head:<28} --[{rel['relation']}"
                  f"/{rel['assertion_status']}]--> {rel['tail']['surface_text']:<28}"
                  f"  trig={trig!r}")
            print(f"          {rel['rule_path']}")
        return

    print("=" * 62)
    print("RELATION EXTRACTION")
    print("=" * 62)
    print(f"\n  cases     : {args.cases}")
    print(f"  entities  : {args.entities}")
    print(f"  threshold : {rf.WEIGHTS['decide.threshold']}")
    print(f"  out       : {args.out}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_relation, by_assertion, by_trigger = (collections.Counter() for _ in range(3))
    total = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for case in cases:
            text = case.get("case_text") or ""
            for rel in analyze_case(text, by_case.get(case["case_id"], []), freq):
                row = to_row(case, rel)
                # Offsets must index back into the untouched case_text.
                assert text[rel["tail"]["start"]:rel["tail"]["end"]] == row["tail_text"]
                if rel["head"]:
                    assert text[rel["head"]["start"]:rel["head"]["end"]] == row["head_text"]
                writer.writerow(row)
                total += 1
                by_relation[rel["relation"]] += 1
                by_assertion[rel["assertion_status"]] += 1
                if rel["trigger"]:
                    by_trigger[" ".join(rel["trigger"]["words"])] += 1

    print(f"  cases processed : {len(cases)}")
    print(f"  relations       : {total:,}  ({total / max(len(cases), 1):.0f} per case)")
    print("\n  By relation:")
    for name, count in by_relation.most_common():
        print(f"    {name:<18} {count:>6,}  {'#' * min(40, count // 15)}")
    print("\n  By assertion status:")
    for name, count in by_assertion.most_common():
        print(f"    {name:<18} {count:>6,}")
    print("\n  Most frequent triggers:")
    for name, count in by_trigger.most_common(15):
        print(f"    {count:>4}  {name}")
    print(f"\n  Written to {args.out}")


if __name__ == "__main__":
    main()
