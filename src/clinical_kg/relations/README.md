# Relation extraction

> **Step-by-step walkthrough** with real traces, per-token scores and two fully
> worked examples: [English](../../../docs/relation-extraction.en.md) ·
> [Castellano](../../../docs/relation-extraction.es.md).
> This file covers the design rationale; those cover the mechanics.

How the clinical relations in `data/processed/relations.csv` are produced, and
why the pipeline is built the way it is. Run from the repository root:

```bash
clinical-kg extract-relations                            # -> data/processed/relations.csv
clinical-kg extract-relations --explain PMC10106591_01   # scored edges for one case
```

Without installing the package, every command also works as
`python3 -m clinical_kg extract-relations ...`, or via `make relations`.

Output: **849 relations across 50 patients**, in eight types, each carrying an
assertion status. The package is stdlib-only, like the rest of the pipeline.

## The problem

Entity extraction gives 3,134 typed entities but no links between them. The
knowledge graph could therefore only say *this case mentions `chest pain` and
`hypertension`* — never that one was revealed by an exam, treated with a drug,
or explicitly denied. Every clinical edge was a generic `MENTIONS_*` from the
Person node.

There is **no labeled relation data** for this corpus, and the project is
restricted to symbolic and classic statistical methods. Both constraints shape
everything below.

## Data flow

```text
interim/cases_clean.csv ──┐
processed/entities.csv ───┤
interim/token_frequency.csv ──┐
                              ▼
                    text_layer.segment()          sentences → clauses → tokens
                              │                   (character offsets throughout)
                              ▼
                    pos.tag_sequence()            6 coarse tags
                              │
                              ▼
                    crf.emissions() ──► crf.viterbi() ──► crf.decode_triggers()
                              │                   O / B-TRIG / I-TRIG / NEG
                              ▼
                    extract.analyze_sentence()    attach · type · coordinate · assert
                              │
                              ▼
                    processed/relations.csv ──► clinical_kg.graph.model.build_graph()
```

```text
relations/
├── text_layer.py   # sentence, clause and token segmentation with offsets
├── pos.py          # coarse POS: hand lexicon + suffix backoff
├── features.py     # trigger lexicon, cue lists, and the WEIGHTS table
├── crf.py          # emission/transition potentials and Viterbi decoding
├── extract.py      # candidate generation, attachment, typing, assertion, CLI
├── annotate.py     # interactive gold-set annotation (evaluation only)
└── evaluate.py     # precision/recall, threshold sweep, ablation
```

Every stage is reached through the package CLI (`clinical_kg/cli.py`), which
dispatches to each module's own `main()`; the modules keep their own argparse
definitions, so the dispatcher cannot drift from the flags they accept.

## Step 1 — Segmentation (`text_layer.py`)

The unit of co-occurrence is the **clause**, not the paragraph. The original
design pruned entity pairs that sat in different paragraphs; measured on this
corpus that prunes nothing, because **6 of the 50 cases contain no newline at
all** and in the rest a "paragraph" averages ~5 sentences.

Sentences split on `[.!?]` followed by whitespace and a plausible opener,
suppressed for the hazards this corpus actually contains: **205 intra-numeric
decimals** (`0.133 kPa`), **18 `Fig.` citations**, unit abbreviations (`g/dL.`)
and single initials. A closing quote or bracket may sit between the period and
the space (`... for the past 2 h." She reported ...`). Verified at **zero bad
splits across all 223 hazards**.

Clauses then split on `,;:` and coordinating conjunctions, and separators are
dropped rather than emitted as clauses of their own.

The tokenizer is **imported from `extract_entities.py`, not copied**. A second
copy that drifts would silently misalign relation offsets against entity
offsets, with no error — only missing links.

## Step 2 — Coarse POS (`pos.py`)

POS tagging is not a goal here. It is the **backoff that makes the bigram
features generalize**, and the corpus shows why it is needed:

- **87% of the bigrams occurring between two entities are seen exactly once**
  (2,156 of 2,489 distinct types); only 130 occur three or more times. A purely
  lexical bigram model memorises ~130 strings and is blind to the rest.
- `presented by` occurs **once** in the entire corpus; `reported with` occurs
  **zero times**. Neither is learnable as a string, yet both are VERB+PREP, and
  crude suffix matching finds 125 distinct VERB+PREP inter-entity bigrams
  occurring 186 times. The *pattern* is frequent where each *string* is not.

Six coarse tags come from a **419-entry hand-built lexicon plus suffix backoff**.
The lexicon exists to fix one error class that suffix rules cannot: they cannot
tell a plural noun from a verb, so they read `areas of`, `episodes of`,
`months of`, `doses of`, `features of` as VERB+PREP. Measured against the crude
rule, the tagger drops exactly that class (126 → 89 bigram types) while
*recovering* irregular pasts the suffix rule missed — `found to`, `found in`,
`developed during`.

`-ing` is deliberately **not** a verb suffix here: its top corpus hits are
`during` 41, `following` 23, `using` 14, `including` 11, `according` 11 — plus
the nominalizations `bleeding` 15, `swelling` 6, `vomiting` 6. An `-ing` word is
tagged VERB only under an auxiliary. `-ify` never fires anywhere in the corpus.

## Step 3 — The CRF (`crf.py`)

A **linear-chain CRF** BIO-tags relation triggers over each sentence's token
sequence, with tags `O`, `B-TRIG`, `I-TRIG`, `NEG`, decoded by **Viterbi**.

The **transition potentials are the bigram model**:

| transition | effect |
|---|---|
| `B → I` (+1.5) | keeps multi-token triggers together: `was treated with`, `presented with`, `followed by` |
| `I → I` (+0.4) | continues, capped at 3 tokens (`-inf` beyond) |
| `B → B` (−3.0) | forbids two adjacent independent triggers |
| `O → I` (`-inf`) | BIO validity |

Viterbi state is `(tag, run_length)`, so the trigger-length cap is enforced
inside the lattice rather than by trimming afterwards.

`I-TRIG` means **continuation**, and is scored separately from `B-TRIG` rather
than derived from it. In English the continuation of a trigger phrase is a
preposition or particle, so `I` rewards PREP and penalises a content verb, which
should open its own trigger. A structural constraint completes it: **a trigger
phrase must be headed by a verb** (`B-TRIG` is `-inf` otherwise).

Both rules were added after observing the same failure twice: without them the
lattice pulls a *noun* into `B` purely to collect the `B→I` bonus, decoding
`examination revealed` and — in "presented to the emergency **department with**
abdominal pain" — `department with` as relation triggers. Both are regression
tests.

### The weights are set by hand

There is no labeled data, so the potentials in `features.WEIGHTS` are hand-set,
not fitted. This is a coherent CRF, not a compromise: a linear-chain CRF is a
log-linear model over sequences, so hand-setting the potentials keeps Viterbi
inference — and with it the sequence-level constraints a per-pair score cannot
express. What is given up is any claim that the values are *optimal*.

Two consequences are designed in:

1. Every weight lives in **one auditable dict**, so the whole model can be read
   and re-tuned in one place.
2. Every edge records which weights fired, in the **`rule_path`** column. An
   explanation is the only defence a hand-weighted model has. `--explain`
   prints them per case.

If labels ever exist, these become the initialization and the same feature code
fits properly.

## Step 4 — Attachment, typing and coordination (`extract.py`)

Each decoded trigger attaches to its **nearest entity on each side within the
clause**. With no entity to the left it anchors to the **Person** root — but
only when the clause subject is patient-referring (`the patient` 263, `she` 93,
`he` 62 …) **and** no other-subject cue is present, so `family` (20) and
`mother` (3) prevent family history becoming the patient's history.

Pair-level potentials add token distance, bucketed to the observed distribution
(gap 0 → 142 pairs, 1–3 → 1,078, 4–10 → 1,133, >10 → 731), clause agreement,
intervening-entity count, and an entity-type prior scored by clinical
meaningfulness rather than raw frequency — `BodyPart+BodyPart` is the third most
common adjacent pair (180) but is usually anatomical modification, so its prior
is **negative**.

### Coordination: the inverted rule

The original design discarded any pair with a comma between it. A comma sits
between **34% of adjacent entity pairs** (1,055 of 3,084), and comma-separated
lists are the single most productive relation pattern in clinical narrative, so
that rule deletes the best signal in the corpus.

Instead, coordinated entities become `COORDINATE_WITH` siblings that **inherit
the head relation of the list's first member**, provided only fillers separate
them and **they share the same `entity_type`**.

That last condition is restrictive in practice. Coordination fires **54 times**
across the corpus and always between identical types: Treatment+Treatment (31),
Finding+Finding (10), Exam+Exam (6), BodyPart+BodyPart (4),
Diagnosis+Diagnosis (3).

> An earlier version of this file claimed that "presented with fatigue, swollen
> abdomen, decreased appetite, and weight loss" yields four symptom edges. That
> holds in the unit test, where all four entities are constructed as `Symptom`
> by hand, but **not on real corpus data**: the gazetteer types that list as
> Symptom / Finding / Symptom / Finding, so no pair coordinates and the sentence
> produces no coordinate edges at all. See
> [`docs/relation-extraction.en.md`](../../../docs/relation-extraction.en.md)
> for the full trace.

### Orientation

Eight relations are emitted — `HAS_SYMPTOM`, `HAS_DIAGNOSIS`, `HAS_FINDING`,
`TREATED_WITH`, `REVEALED_BY`, `LOCATED_IN`, `CAUSED_BY`, `COORDINATE_WITH` —
chosen from the trigger's category crossed with the argument types. Each edge is
then checked against its relation's **type signature** and flipped if it points
backwards, so `right ovary → REVEALED_BY → laparoscopic exploration` becomes
`laparoscopic exploration → REVEALED_BY → right ovary`.

## Step 5 — Assertion status

The corpus contains **293 negation-cue tokens** (`no` 134, `not` 73, `negative`
47, `without` 39, `denied` 11) plus hedges (`consistent` 22, `may` 20,
`possible` 12). Ignoring them makes the graph assert findings the text denies.

A NegEx/ConText-style **forward scope** labels every edge `affirmed`, `negated`,
`hedged`, `historical` or `family`. A cue opens scope; scope closes at a clause
boundary or a termination cue (`but`, `however`). That is what keeps the first
mention affirmed and negates only the second in:

> "Physical examination revealed abdominal pain in RIF, with localized
> tenderness in the RIF **but no** rebound tenderness was found."

Negated relations are **labelled, never dropped** — "no rebound tenderness" is a
clinical fact about the patient too.

## Output

`data/processed/relations.csv`, one row per edge:

| column | meaning |
|---|---|
| `relation`, `assertion_status` | the typed edge and its status |
| `head_kind`, `head_*`, `tail_*` | arguments by offset, CUI, type and surface text (`head_kind=Person` for a root-anchored edge) |
| `trigger_text`, `trigger_start/end`, `trigger_category` | the decoded trigger |
| `sent_start/end`, `token_gap`, `same_clause` | context |
| `score`, `rule_path` | the score and exactly which weights produced it |

Offsets index the untouched `case_text` and are asserted on every write.
`clinical_kg.graph.model.build_graph(..., relations=...)` turns these into typed
directed edges; an edge whose endpoints do not resolve is reported in
`warnings` rather than attached to an arbitrary node.

## Evaluation

With nothing to train on, the gold set is **test-only**: 9 cases chosen
deterministically and spread by entity count, never used to tune weights.

```bash
clinical-kg annotate-relations     # build the gold set (resumable)
clinical-kg evaluate-relations     # P/R/F1, sweep, per-relation, ablation
```

The annotator enumerates every **candidate** pair, not the edges the model
accepted — judging only the model's own output measures precision and leaves
recall unmeasurable. Because the weights are hand-set, the defensible claim is a
component one, so the evaluator prints an **ablation table**: what the POS
backoff, coordination inheritance, assertion scoping and CRF transitions each
contribute.

Two ceilings are reported alongside the numbers, and both are real: recall is
bounded by the candidate generator's window (`pair.max_token_gap` tokens,
intra-sentence or Person-anchored) and by how many candidates were judged.

> The gold set has **not been annotated yet**. Until it is, the pipeline has
> counts but no quality numbers.

## Tests

```bash
python3 -m unittest discover -s tests
```

`tests/test_relations.py` covers segmentation hazards, the POS minimal pairs,
BIO validity and the trigger-length cap, negation scope termination, coordination
inheritance, edge orientation, offset round-trips, the graph integration, and a
subprocess smoke test of each CLI entry point.
