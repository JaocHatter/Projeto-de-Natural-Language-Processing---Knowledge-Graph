#!/usr/bin/env python3
"""UMLS's own relations (MRREL.RRF) between concepts this corpus actually uses.

This is a *second*, independent source of relation evidence, alongside the
text-derived CRF in extract.py: instead of inferring "gastritis TREATED_WITH
aspirin" from how a sentence is worded, UMLS may already assert
`gastritis may_treat aspirin` as domain knowledge, regardless of what any one
case report says. It feeds an extra, explicitly labelled graph layer
(`graph/model.py::build_graph(..., umls_relations=...)`): ontological edges
between concepts, distinct from `MENTIONS_*`/text-derived typed relations. A
patient having "aspirin" and "gastritis" both mentioned does not mean the
*text* asserts the drug treated that finding for that patient -- this layer
says only that UMLS relates the two concepts in general, and it is tagged
`evidence_source="umls_ontology"` so it is never confused with a text-derived
edge of the same name.

RELA_DIRECTION deliberately does not try to cover every MRREL relation type,
only the ones with an unambiguous, high-precision match to one of this
project's 8 relation types. `may_treat`/`may_be_treated_by` -- the best
source for TREATED_WITH -- come only from the MED-RT vocabulary, not the six
this project's gazetteer restricts itself to (SNOMEDCT_US, MSH, LNC, RXNORM,
ICD10CM, MTH); the gazetteer's vocabulary whitelist exists to keep *entity
names* manageable and is not reapplied here, since this step adds no new
entities, only edges between CUIs the corpus already extracted. Each row's
source vocabulary is still carried through to the output CSV for
provenance/auditability.

Build with:
    clinical-kg build-umls-relations
Reads  : data/processed/entities.csv (which CUIs this corpus actually uses),
         data/external/umls-2026AA-metathesaurus-full.zip (MRREL.RRF member)
Writes : data/interim/umls_relations.csv
"""

import argparse
import collections
import csv
import io
import math
import sys
import zipfile
from pathlib import Path

from clinical_kg.paths import (DEFAULT_ENTITIES, DEFAULT_UMLS_RELATIONS,
                           DEFAULT_UMLS_ZIP)

DEFAULT_OUT = DEFAULT_UMLS_RELATIONS

# RELA (col 8 of MRREL.RRF) -> (this project's relation name, swap?).
# `swap` says whether MRREL's (CUI1, CUI2) must be flipped to line up with
# this project's own head/tail convention (features.TAIL_TYPES): head is the
# clinical subject (disease/patient/agent), tail is TREATED_WITH's drug,
# LOCATED_IN's body part, CAUSED_BY's effect.
#
# This is NOT derived from the RELA name's grammar -- checked empirically
# against this corpus's own resolved CUIs instead: MED-RT's
# may_treat/may_be_treated_by pair does not mirror each other the way
# finding_site_of/has_finding_site do, so guessing direction from the name
# alone would have silently produced backwards TREATED_WITH edges.
#
#   rela                  observed (cui1, cui2)         swap needed?
#   may_treat              (disease, drug)               no
#   may_be_treated_by      (drug, disease)                yes
#   finding_site_of        (finding, bodypart)            no
#   has_finding_site       (bodypart, finding)            yes
#   has_causative_agent    (agent, effect)                no
#   causative_agent_of     (effect, agent)                yes
#
# Verified present against a live 2026AA MRREL.RRF: has_finding_site/
# finding_site_of and has_causative_agent/causative_agent_of occur under the
# project's own six gazetteer vocabularies; may_treat/may_be_treated_by occur
# only under MED-RT (not one of those six -- see the module docstring).
RELA_DIRECTION = {
    "may_treat":            ("TREATED_WITH", False),
    "may_be_treated_by":    ("TREATED_WITH", True),
    "finding_site_of":      ("LOCATED_IN", False),
    "has_finding_site":     ("LOCATED_IN", True),
    "has_causative_agent":  ("CAUSED_BY", False),
    "causative_agent_of":   ("CAUSED_BY", True),
}

OUT_COLUMNS = ["head_cui", "tail_cui", "rela", "relation", "sab"]


def corpus_cuis(entities_path: Path) -> set[str]:
    """Every CUI this corpus's extracted entities actually use."""
    with open(entities_path, newline="", encoding="utf-8") as f:
        return {row["cui"] for row in csv.DictReader(f) if row["cui"]}


def _mrrel_lines(zip_path: Path):
    """Stream MRREL.RRF's lines out of the UMLS zip without extracting it.

    MRREL.RRF is ~6 GB uncompressed; zipfile reads it as a stream so the
    whole thing is never held in memory or written back to disk.
    """
    with zipfile.ZipFile(zip_path) as zf:
        name = next((n for n in zf.namelist() if n.endswith("META/MRREL.RRF")),
                    None)
        if name is None:
            raise FileNotFoundError(f"no META/MRREL.RRF member in {zip_path}")
        with zf.open(name) as raw:
            yield from io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def resolve_row(fields: list[str], cuis: set[str]) -> dict | None:
    """Apply the RELA_DIRECTION filter/reorientation to one parsed MRREL.RRF
    line's fields. Returns a row dict, or None if this line should be
    dropped. Split out from build() so the direction logic is testable
    without a real UMLS zip.
    """
    if len(fields) < 15:
        return None
    cui1, rela, cui2, sab, suppress = (fields[0], fields[7], fields[4],
                                       fields[10], fields[14])
    direction = RELA_DIRECTION.get(rela)
    if direction is None or suppress != "N":
        return None
    if cui1 not in cuis or cui2 not in cuis or cui1 == cui2:
        return None
    relation, swap = direction
    head_cui, tail_cui = (cui2, cui1) if swap else (cui1, cui2)
    return {"head_cui": head_cui, "tail_cui": tail_cui, "rela": rela,
            "relation": relation, "sab": sab}


def build(entities_path: Path, zip_path: Path, out_path: Path) -> int:
    """Filter MRREL.RRF down to relations between this corpus's own CUIs.

    Keeps a row only if: its RELA is in RELA_DIRECTION, it is not suppressed,
    and *both* endpoints are CUIs this corpus's entities.csv actually contains
    -- an edge whose endpoint never resolves to a node in this corpus's graph
    is not useful here, matching the rest of the project ("an edge whose
    endpoints do not resolve is reported in warnings rather than attached to
    an arbitrary node"). Each row is reoriented to (head_cui, tail_cui) per
    RELA_DIRECTION before being written.

    Returns the number of rows written.
    """
    cuis = corpus_cuis(entities_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for line in _mrrel_lines(zip_path):
            row = resolve_row(line.rstrip("\n").split("|"), cuis)
            if row is None:
                continue
            key = (row["head_cui"], row["tail_cui"], row["relation"])
            if key in seen:
                continue
            seen.add(key)
            writer.writerow(row)
            n += 1
    return n


def load(path: Path) -> dict[tuple[str, str], list[dict]]:
    """(head_cui, tail_cui) -> list of {relation, rela, sab} rows."""
    by_pair = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_pair.setdefault((row["head_cui"], row["tail_cui"]), []).append(row)
    return by_pair


def type_prior(entities_path: Path, umls_relations_path: Path
              ) -> dict[tuple[str, str], float]:
    """Data-derived prior per (head_entity_type, tail_entity_type), replacing a
    hand-picked features.ENTITY_PAIR_PRIOR guess with a real density: how many
    UMLS relations connect concepts of these two project types, among CUIs
    this corpus's own entities actually use.

    Counts are sqrt-compressed (so a thin count like 3 doesn't collapse to
    near-zero next to a well-supported one like 92) and scaled so the
    best-supported pair lands at 1.0 -- the same ceiling ENTITY_PAIR_PRIOR
    already used for its strongest hand-set guess. Measured on this corpus:

        (Diagnosis, Treatment)  1.00  n=92     (Symptom, Treatment)   0.36  n=12
        (Diagnosis, BodyPart)   0.81  n=61     (Finding, Treatment)   0.26  n=6
        (Symptom, BodyPart)     0.49  n=22     (Treatment, Diagnosis) 0.18  n=3
        (Finding, BodyPart)     0.43  n=17

    Returns {} if either file is missing (e.g. build-umls-relations has not
    been run yet) rather than raising -- callers fall back to the hand-set
    table, same as if UMLS coverage simply didn't exist.

    Deliberately partial: only type pairs whose relation is one
    RELA_DIRECTION maps (TREATED_WITH/LOCATED_IN/CAUSED_BY-eligible) get a
    count here. Pairs behind HAS_SYMPTOM/HAS_DIAGNOSIS/HAS_FINDING/
    REVEALED_BY have no UMLS equivalent to derive a density from, so they are
    simply absent from the returned dict -- not zeroed out -- and the caller
    keeps using its hand-set value for those.
    """
    if not entities_path.exists() or not umls_relations_path.exists():
        return {}
    cui_types = collections.defaultdict(collections.Counter)
    with open(entities_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cui_types[row["cui"]][row["entity_type"]] += 1
    cui_type = {cui: counter.most_common(1)[0][0] for cui, counter in cui_types.items()}

    counts = collections.Counter()
    with open(umls_relations_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            head_type = cui_type.get(row["head_cui"])
            tail_type = cui_type.get(row["tail_cui"])
            if head_type and tail_type:
                counts[(head_type, tail_type)] += 1
    if not counts:
        return {}
    max_count = max(counts.values())
    return {pair: round(math.sqrt(n / max_count), 2) for pair, n in counts.items()}


def main():
    ap = argparse.ArgumentParser(
        description="Filter UMLS's own MRREL.RRF relations down to CUI pairs "
                    "this corpus's entities actually use.")
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--umls-zip", type=Path, default=DEFAULT_UMLS_ZIP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    for path in (args.entities, args.umls_zip):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            sys.exit(1)

    print("=" * 62)
    print("UMLS RELATIONS (MRREL.RRF -> this corpus's CUIs)")
    print("=" * 62)
    print(f"\n  entities : {args.entities}")
    print(f"  umls zip : {args.umls_zip}  (this is the slow part -- ~6 GB, "
          "streamed, not extracted to disk)")
    print(f"  out      : {args.out}\n")

    n = build(args.entities, args.umls_zip, args.out)
    print(f"  {n} relations written")
    if n == 0:
        print("  (0 is possible on a small corpus -- not every relevant CUI "
              "pair has a matching UMLS relation)")


if __name__ == "__main__":
    main()
