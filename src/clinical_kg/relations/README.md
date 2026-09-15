# Relation extraction

How the clinical relations in `data/processed/relations.csv` are produced, and
why the pipeline is built the way it is. Run from the repository root:

```bash
clinical-kg extract-relations                            # -> data/processed/relations.csv
clinical-kg extract-relations --explain PMC10106591_01   # scored edges for one case
```

Without installing the package, every command also works as
`python3 -m clinical_kg extract-relations ...`, or via `make relations`.

Output: **871 relations across 50 patients** (over the committed
`entities.csv`, 2,957 entities), in eight types, each carrying an assertion
status. The package is stdlib-only, like the rest of the pipeline.

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
                    extract.analyze_sentence()    attach (coreference.CorefState) ·
                              │                   type · coordinate · assert
                              ▼
                    processed/relations.csv ──► clinical_kg.graph.model.build_graph()
```

```text
relations/
├── core/               # the algorithm itself -- no CLI, imported by extract.py
│   ├── text_layer.py   #   sentence, clause and token segmentation with offsets
│   ├── pos.py          #   coarse POS: hand lexicon + suffix backoff
│   ├── features.py     #   trigger lexicon, cue lists, and the hand-set WEIGHTS table
│   ├── crf.py          #   emission/transition potentials and Viterbi decoding
│   └── coreference.py  #   cross-sentence patient anchoring (recency rule, no ML)
├── extract.py          # candidate generation, attachment, typing, assertion, CLI --
│                       #   the one module that runs in production
├── ontology.py         # UMLS's own relations (MRREL.RRF), a second evidence source
└── tools/              # developer-facing only, not part of extract-relations
    ├── annotate.py     #   interactive gold-set annotation
    └── evaluate.py     #   precision/recall, threshold sweep, ablation
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

## Cross-sentence patient coreference (`coreference.py`)

`extract.py`'s attachment already anchored a trigger with no left-hand entity
to the **Person** root when the clause subject was patient-referring
(`the patient`, `she`, `he`, ...) — but only within that one clause. A
one-bit `CorefState` (`patient_active`) is now threaded across a case's
sentences in `analyze_case`, so an elided subject in a later sentence still
resolves correctly:

> The patient was diagnosed with pneumonia. **Subsequently treated with
> ceftriaxone.**

The second sentence has no `PATIENT_CUES` word at all, so before this it
could never anchor to Person and the edge was silently unreachable. The rule
is recency, not general entity coreference (every case here has exactly one
patient to refer to): a sentence's own `PATIENT_CUES`/`FAMILY_CUES` always
take priority over the inherited state, so a fresh "his mother" correctly
blocks anchoring in its own sentence without leaking into the next one, and a
fresh "he" in the next sentence overrides a stale family-history state from
the one before. A sentence with no clinical entity at all is skipped by
`analyze_case` entirely and therefore never updates the state — a known gap,
same spirit as this project's other documented heuristic limits.

## Step 4 — Attachment, typing and coordination (`extract.py`)

Each decoded trigger attaches to its **nearest entity on each side within the
clause**. With no entity to the left it anchors to the **Person** root — but
only when the clause subject is patient-referring (`the patient` 263, `she` 93,
`he` 62 …) **and** no other-subject cue is present, so `family` (20) and
`mother` (3) prevent family history becoming the patient's history. When the
clause has no subject cue of its own at all (an elided subject: "Subsequently
treated with ceftriaxone."), this falls back to `coreference.CorefState`,
which carries the previous sentence's patient/family verdict forward — see
[Cross-sentence patient coreference](#cross-sentence-patient-coreference-coreferencepy)
below.

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
> produces no coordinate edges at all.

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
`warnings` rather than attached to an arbitrary node. Both consumers feed it:
`graph/export.py` reads this file in `--from-csv` mode and otherwise extracts
live (`extract_relations`, the same `analyze_case` + `to_row` path over the
live entity spans), and the Streamlit viewer calls that same helper, so the app
and the CSV can never disagree on how a relation was produced.

## A second evidence source: UMLS's own relations (`ontology.py`)

Everything above infers relations from how a sentence is *worded*. UMLS
already asserts many of the same relations as domain knowledge, independent
of any one case report -- e.g. `may_treat`/`may_be_treated_by` between a drug
and a disease in `MRREL.RRF`. `ontology.py` filters that file down to
relations between CUIs this corpus's own entities actually resolved to:

```bash
clinical-kg build-umls-relations     # -> data/interim/umls_relations.csv
```

`MRREL.RRF` is read as a stream directly out of the UMLS metathesaurus zip
(`zipfile` + `io.TextIOWrapper`) and never extracted to disk -- it is ~6 GB
uncompressed. Only three RELA pairs are mapped, chosen for an unambiguous,
high-precision match to one of this project's 8 relation types:
`may_treat`/`may_be_treated_by` → `TREATED_WITH`,
`has_finding_site`/`finding_site_of` → `LOCATED_IN`,
`has_causative_agent`/`causative_agent_of` → `CAUSED_BY`. Direction
(`RELA_DIRECTION`'s `swap` flag) was **verified empirically** against a live
2026AA `MRREL.RRF`, not assumed from the RELA name's grammar --
`may_treat`/`may_be_treated_by` turned out not to mirror each other the way
`finding_site_of`/`has_finding_site` do, so guessing would have silently
produced backwards `TREATED_WITH` edges. The gazetteer's six-vocabulary
whitelist (`SNOMEDCT_US, MSH, LNC, RXNORM, ICD10CM, MTH`) is deliberately
**not** reapplied here: `may_treat`/`may_be_treated_by` only exist under
`MED-RT`, and this step adds no new entities, only edges between CUIs the
corpus already extracted, so the vocabulary restriction that keeps entity
*names* manageable does not need to apply to relation *evidence*. Each row's
source vocabulary is still kept in the output for provenance.

This feeds **a distinct graph layer** (`graph.model.build_graph(...,
umls_relations=...)`): an edge between two concepts *this case's own
entities* resolved to, tagged `evidence_source="umls_ontology"` and never
sharing an edge id with a text-derived edge of the same relation -- "UMLS
relates these two concepts in general" is not a claim that this patient's
text asserts it for this case.

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

### Results

`data/interim/gold_relations.csv` holds **385 judged candidates over the 9 gold
cases: 207 true relations, 178 rejected** (every candidate the generator
produced at threshold −∞, judged from its sentence; entity-extraction
artefacts such as `2O`, `back`, `A 7`, `4, 5, 6` are rejected outright). Gold
labels follow the project's own schema conventions -- an Exam whose finding is
localised in a body part is `LOCATED_IN` (`RELATION_BY_CATEGORY`'s
`REVEAL+BodyPart`), a patient undergoing a procedure or exam is `TREATED_WITH`
(`TREAT+Exam`), list members are `COORDINATE_WITH` even across entity types --
and 76 of the 207 accepted pairs carry a *corrected* relation type, 19 a
corrected assertion status. Distribution: `COORDINATE_WITH` 70, `LOCATED_IN`
41, `TREATED_WITH` 31, `HAS_DIAGNOSIS` 21, `REVEALED_BY` 19, `HAS_FINDING` 12,
`HAS_SYMPTOM` 9, `CAUSED_BY` 4; 178 affirmed, 13 negated, 11 hedged, 3
historical, 2 family.

`clinical-kg evaluate-relations` at the configured threshold (1.5):

| | P | R | F1 |
|---|---|---|---|
| detection | 0.618 | 0.454 | 0.524 |
| relation type, on detected edges | 0.734 | | |
| assertion status, on detected edges | 0.883 | | |

Per relation: `HAS_SYMPTOM` F1 0.95, `REVEALED_BY` 0.76, `HAS_DIAGNOSIS` 0.74,
`TREATED_WITH` 0.64, `LOCATED_IN` 0.53, `CAUSED_BY` 0.44, `HAS_FINDING` 0.44,
`COORDINATE_WITH` 0.18 (8 TP / 62 FN -- most gold lists are cross-type or
span an intervening non-entity, which `coordinate_siblings` rejects). The
threshold sweep peaks at **F1 0.675 at threshold 0.5** (P 0.605, R 0.763): the
configured 1.5 trades ~30 points of recall for ~1 of precision. Ablation at
1.5: POS backoff −0.043 F1 when removed, coordination −0.079, CRF transitions
−0.004, assertion scoping ±0 on detection but −0.05 assertion accuracy.

Because the gold set is test-only, none of this has been fed back into
`WEIGHTS`; the threshold observation is recorded here for whoever tunes next,
not applied.

## Tests

```bash
python3 -m unittest discover -s tests
```

`tests/test_relations.py` covers segmentation hazards, the POS minimal pairs,
BIO validity and the trigger-length cap, negation scope termination, coordination
inheritance, edge orientation, offset round-trips, the graph integration, a
subprocess smoke test of each CLI entry point, cross-sentence coreference (and
that a family cue never leaks into the next sentence), and `ontology.py`'s RELA
direction table (pinned against real examples, not just internal consistency).
`tests/test_graph.py` covers the UMLS ontology edge layer staying distinct
from a text-derived edge of the same relation and pair.
