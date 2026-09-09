# Natural Language Processing - Knowledge Graph Project

## Quickstart

The pipeline is stdlib-only and runs from a plain checkout — the `Makefile`
puts `src` on `PYTHONPATH`, so nothing needs installing:

```bash
make test      # run the test suite
make all       # clean-cases -> entities -> measurements -> relations
make app       # launch the Streamlit viewer
make help      # list every target
```

Installing the package adds a `clinical-kg` command (equivalent to
`python3 -m clinical_kg`):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
clinical-kg                                    # list every command
clinical-kg extract-relations --explain PMC10106591_01
```

Only the Streamlit viewer needs third-party packages; every extraction stage is
standard library only.

## Project Structure

```
...
│
└── ProjetoNLP
    │
    ├── README.md  <- Project documentation
    │
    ├── data
    │   ├── external       <- Third-party data in input format for transformation
    │   ├── interim        <- Intermediate data, e.g., transformation results
    │   ├── processed      <- Final data used for publication
    │   └── raw            <- Original data without modifications
    │
    ├── pipelines
    │   ├── notebooks      <- Jupyter notebooks or equivalent
    │   └── workflows      <- Orange workflows or equivalent 
    │
    ├── pyproject.toml     <- Packaging, dependencies and the clinical-kg command
    ├── Makefile           <- Zero-install entry points (make test / all / app)
    │
    ├── src                <- Source code (src-layout: one installable package)
    │   └── clinical_kg
    │       ├── README.md      <- Architecture, layering and data flow
    │       ├── paths.py       <- Single source of truth for project paths
    │       ├── cli.py         <- One dispatcher over every pipeline stage
    │       ├── corpus/        <- Case cleaning and patient records
    │       ├── gazetteer/     <- UMLS term dictionary (SQLite)
    │       ├── extraction/    <- Entities and measurements
    │       ├── relations/     <- Typed clinical relations (CRF)
    │       ├── graph/         <- Graph model, export and viewer
    │       └── app/           <- Streamlit front end
    │
    ├── tests              <- Test suite (python3 -m unittest discover -s tests)
    │
    └── assets             <- Media used in the project
        ├── images         <- Images used in README.md text
        └── slides         <- PDF slides
```

## External Data (UMLS 2026AA)

The project uses **UMLS semantic classification** for medical entity extraction. Data is located in `data/external/umls_csvs/` organized by **14 semantic types (TUI)**:

| CSV File | TUI | Semantic Type | Concept | Examples |
|---------|-----|----------------|---------|----------|
| `umls_T047_Disease_or_Syndrome.csv` | T047 | Disease or Syndrome | Clinical conditions | acute pancreatitis, diabetes |
| `umls_T200_Clinical_Drug.csv` | T200 | Clinical Drug | Medications | aspirin, metformin, ibuprofen |
| `umls_T023_Body_Part_Organ.csv` | T023 | Body Part or Organ | Anatomy | pancreas, stomach, liver |
| `umls_T191_Neoplastic_Process.csv` | T191 | Neoplastic Process | Cancers/Tumors | gastric neoplasia, leukemia |
| `umls_T033_Finding.csv` | T033 | Finding | Clinical findings | elevated glucose, fever |
| `umls_T037_Injury_or_Poisoning.csv` | T037 | Injury or Poisoning | Trauma | fracture, burn, intoxication |
| `umls_T046_Pathologic_Function.csv` | T046 | Pathologic Function | Dysfunctions | hemorrhage, hypertension |
| `umls_T059_Laboratory_Procedure.csv` | T059 | Laboratory Procedure | Lab tests/analyses | blood test, urinalysis |
| `umls_T060_Diagnostic_Procedure.csv` | T060 | Diagnostic Procedure | Diagnostic procedures | computed tomography, endoscopy |
| `umls_T061_Therapeutic_Procedure.csv` | T061 | Therapeutic Procedure | Surgical treatments | surgery, transplant |
| `umls_T121_Pharmacologic_Substance.csv` | T121 | Pharmacologic Substance | Chemical components | penicillin, morphine |
| `umls_T029_Body_Location_or_Region.csv` | T029 | Body Location or Region | Regional anatomy | abdomen, thorax |
| `umls_T184_Sign_or_Symptom.csv` | T184 | Sign or Symptom | Clinical manifestations | cough, headache |
| `umls_T034_Laboratory_or_Test_Result.csv` | T034 | Laboratory or Test Result | Test values | glucose level, blood pressure |

**Consolidated file:** `umls_clinico_todos.csv` (~190 MB, 1,666,483 rows covering 644,516 unique CUIs across all 14 types)

> Note: the CSVs quote fields that contain commas (e.g. `"muscle, abdominal"`),
> so they must be parsed with a real CSV reader — `cut -d,` / `awk -F,` will misparse them.

## Gazetteer

`src/clinical_kg/gazetteer/build.py` loads these CSVs into an indexed SQLite database used for
longest-match entity extraction:

```bash
bash data/external/filter_umls_mrsty.bash   # produces data/external/umls_csvs/
clinical-kg build-gazetteer              # produces data/interim/gazetteer.db
```

## Data Cleaning

`data/raw/cases.csv` comes from an upstream dataset build we don't control and has no source
code of its own -- just a one-line field description in `data/raw/data_dictionary.csv`. It has
confirmed issues: rows that are chapters of one patient's report split across several `case_id`s,
rows that aren't single-patient case reports at all, and an `age`/`gender` pair that's occasionally
wrong (e.g. a newborn's *39-week gestational age* recorded as `age=39`).

`src/clinical_kg/corpus/clean.py` addresses this before anything else runs:

```bash
clinical-kg clean-cases                  # data/raw/cases.csv -> data/interim/cases_clean.csv
```

- **Merges** 2 articles whose case report was split across multiple `case_id` rows into one row
  per real patient (`PMC6083636`: 3 fragments -> 1; `PMC11259348`: 2 -> 1).
- **Drops** 3 rows that aren't single-patient cases: a 115-patient retrospective cohort summary,
  a dental-education methodology paper, and an unrelated sociology paper about Theranos -- all
  three slipped in under `case_amount=1` upstream.
- **Re-derives `age`/`gender` from the case text itself** (regex against the patient-introducing
  clause -- `"A/An <N>-year-old..."`, newborn/infant language, spelled-out numbers, explicit
  gender words, pronoun-majority fallback) instead of trusting the upstream columns. Both the
  upstream and self-extracted values are kept side by side (`age`/`gender` vs.
  `age_upstream`/`gender_upstream`) for auditing.

Corpus goes from **56 rows to 50 real patients**. Every downstream script takes `--cases`, so
they all point at the cleaned file instead of the raw one from here on.

## Entity Extraction

`src/clinical_kg/extraction/entities.py` runs longest-match extraction over the cleaned corpus:

```bash
clinical-kg extract-entities --cases data/interim/cases_clean.csv \
                                 --out data/processed/entities.csv
```

Yields **3,134 entities across 50 patients**, typed as Treatment / Finding / Diagnosis / BodyPart /
Exam / Symptom. Each row carries character offsets, CUI, TUI and source vocabulary. See `PLAN.md`
for the TUI-to-entity mapping.

## Measurements

Medications are not extracted as a separate category: RXNORM/MSH already resolve most drug names
inside the gazetteer's `Treatment` type (alongside `T061` therapeutic procedures), and the team
decided a suffix-pattern fallback for the rest wasn't worth the added complexity for this stage.

`src/clinical_kg/extraction/measurements.py` finds value(+range)+unit spans (`850 U/L`, `10-140 U/L`, `4 cm`,
`40mg`, `day 8`) over a closed clinical unit vocabulary, then links each one to the nearest gazetteer
entity that precedes it in the same sentence -- checking just after the value too, for the common
adjectival phrasing ("4 cm pseudocyst", "5-day history") -- falling back to a character window when
the sentence has no candidate:

```bash
clinical-kg extract-measurements --cases data/interim/cases_clean.csv \
                                     --out data/processed/measurements.csv
```

Against the full gazetteer: **535 measurements across 50 patients**, and 516 of them (96%) resolve
to a linked entity -- 382 `same_sentence` (high-confidence), 55 `same_sentence_forward` (adjectival,
e.g. "8 cm spleen"), 79 `window_fallback` (lower-confidence), 19 `none`.

Stdlib-only, following the same offset/normalization conventions as `extract_entities.py` so its
output lines up with `entities.csv` by character position. Known limitation: the nearest-entity
heuristic has no notion of hospital-outcome spans ("discharged on day 8"), so a duration mentioned
late in a case can link to the wrong entity instead -- downstream users can filter on `link_method`
(`same_sentence` is high-confidence, `window_fallback` is not) or discard unlinked rows.

### About `data/external/filter_umls_mrsty.bash`

This file was missing from the repository (not in git history, and this clone has no configured
remote to re-fetch it from), so it was reconstructed from this README's own spec -- English,
non-suppressed, the same six vocabularies and 14 semantic types, same column names. Real run against
the two UMLS zips: **1,617,808 consolidated rows** and a **1,073,291-term / 296 MB gazetteer**,
close to but not identical to the numbers quoted elsewhere in this document (1,666,483 / 1,097,541 /
289 MB) -- expected, since this is a reconstruction, not the original byte-for-byte script. If the
original teammate's version turns up, prefer it and diff the two rather than assuming they match;
whichever one is kept should be the one actually committed to git going forward.

## Relations

`src/extract_relations.py` extracts typed, directed clinical relations between the entities,
using a **linear-chain CRF** that BIO-tags relation triggers over each sentence and decodes with
Viterbi:

```bash
clinical-kg extract-relations --cases data/interim/cases_clean.csv \
                                 --entities data/processed/entities.csv \
                                 --out data/processed/relations.csv
```

Yields **849 relations across 50 patients** in eight types -- `TREATED_WITH`, `LOCATED_IN`,
`HAS_DIAGNOSIS`, `REVEALED_BY`, `HAS_FINDING`, `COORDINATE_WITH`, `HAS_SYMPTOM`, `CAUSED_BY` --
each carrying an **assertion status** (`affirmed` / `negated` / `hedged` / `historical` /
`family`), so "examination revealed tenderness ... but **no** rebound tenderness" does not become
an affirmed symptom.

The CRF's weights are **set by hand** (`src/clinical_kg/relations/features.py`, one auditable `WEIGHTS` dict),
not learned: this project has no labeled relation data. A linear-chain CRF is a log-linear model
over sequences, so hand-set potentials keep Viterbi inference and the sequence constraints that a
per-pair score cannot express -- BIO validity, one trigger per clause, a trigger-length cap -- while
giving up any claim the weights are optimal. Every edge therefore records which weights fired in a
`rule_path` column, and `--explain CASE_ID` prints scored relations for one case.

The implementation is isolated in `src/clinical_kg/relations/` (segmentation, coarse POS, the weight table,
the CRF, extraction, annotation and evaluation), stdlib-only like the rest of the pipeline;
`src/extract_relations.py` is a compatibility entry point, following the `graph_export.py`
convention.

**[`src/clinical_kg/relations/README.md`](src/clinical_kg/relations/README.md) documents the full process** -- each stage,
what the corpus measurements showed, and why each design choice was made. Two highlights: POS
exists because 87% of the bigrams between two entities occur exactly once (`presented by` appears
once in the corpus, `reported with` never), and the "discard any pair with a comma between them"
rule had to be **inverted**, since a comma sits between 34% of adjacent entity pairs and
coordinated lists are the most productive relation pattern in the text.

See [`MEMORY_BANK/PLAN_TO_GET_RELATIONS.md`](MEMORY_BANK/PLAN_TO_GET_RELATIONS.md) for the
original seven heuristics and what measuring each one against the corpus showed.

### Evaluating relations

With no labels to train on, the gold set is **test-only**: 9 cases chosen deterministically and
spread by entity count, never used to tune the weights.

```bash
clinical-kg annotate-relations     # build the gold set (resumable, accept/reject)
clinical-kg evaluate-relations     # P/R/F1, threshold sweep, per-relation, ablation
```

The annotator enumerates every *candidate* pair rather than the edges the model accepted -- judging
only the model's own output would measure precision and leave recall unmeasurable. Because the
weights are hand-set, the defensible claim is a component one, so the evaluator prints an ablation
table showing what the POS backoff, coordination inheritance, assertion scoping and CRF transitions
each contribute. Recall ceilings (the candidate window, and how many candidates were judged) are
printed alongside.

## Visualization

`src/clinical_kg/app/main.py` is a Streamlit viewer that highlights entities inline, for either a corpus case or new
text you paste or upload. Extraction runs live (~35 ms per case) through the same `extract()`
function the CLI uses, so the app and the pipeline can never disagree. The corpus source is
`data/interim/cases_clean.csv` (50 patients, see Data Cleaning above), not the raw file -- the
sidebar shows each selected patient's `case_id`, self-extracted age and sex alongside the text.

The **Knowledge Graph** tab uses that same cleaned patient and live extraction.
It represents Person, Age, Sex, the six clinical entity categories and their
linked measurements. Select nodes or edges to inspect source text and methods;
switch between concepts aggregated by CUI and individual occurrences; filter
types and association methods; and export JSON, GraphML or CSV ZIP. Relation
labels and an explicit Relations table make the graph's meaning inspectable.

Measurements are hidden initially: click a concept to reveal its associated
values, or Person to reveal unlinked measurements. Clicking the background hides
them again. The Measurements control also offers Hide all; there is no automatic
Show all mode. Changing cases or filters clears previous expansions.
The canvas provides search by name/CUI, focus selection, zoom buttons and Escape
to clear selection. Circular nodes and zoom-sensitive labels reduce clutter.
Enable **Group by IS_A category** to arrange concepts around Treatment, Diagnosis,
Exam, Finding, Symptom and BodyPart class nodes. Multi-type concepts keep every
IS_A relation (their visual placement uses one class). These are project semantic
categories, not inferred clinical assertions or the full UMLS hierarchy.

Extracted relations from `data/processed/relations.csv` are passed to `build_graph(...,
relations=...)` and become typed directed edges alongside the `MENTIONS_*` scaffolding, each
retaining its assertion status, trigger text and score. A relation whose endpoints do not resolve
to an entity occurrence is reported in `warnings` rather than attached to an arbitrary node, and a
negated relation is labelled, never silently dropped.

`Person` is identified by the cleaned `case_id` (including merged `_P1` IDs).
`HAS_AGE` and `HAS_SEX` use the cleaning output and retain extraction methods,
upstream values and `source_case_ids` for auditing. Missing age produces no Age
node; newborn age **0** is retained. Clinical links are typed `MENTIONS_*`
relations, and measurement links remain `ASSOCIATED_WITH_MEASUREMENT` with their
original occurrence offsets and `link_method`. A text mention does not establish
a confirmed diagnosis, treatment administration or causality.

All downstream defaults now use `data/interim/cases_clean.csv` via
`src/clinical_kg/paths.py`. The app and graph export CLI extract directly from the
selected cleaned text, so stale processed CSVs cannot introduce dropped patients
or lose the merged fragments. Optional `--from-csv` export validates corpus IDs
and offsets and rejects incompatible annotations; `--cases` still supports an
explicit alternative corpus. See [source documentation](src/clinical_kg/README.md) for the
graph schema, CLI, provenance and tests.

```bash
make app                                  # no install needed
```

or, to get the `clinical-kg` command on your PATH:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
streamlit run src/clinical_kg/app/main.py     # opens http://localhost:8501
```

Hover any highlight for its CUI, semantic type and source vocabulary. The page also shows per-type
counts, a sortable entity table, and a CSV download in the same schema as
`data/processed/entities.csv`.

> The pipeline modules (`build_gazetteer.py`, `extract_entities.py`) are stdlib-only and need no
> dependencies; only the viewer requires `streamlit` and `pandas`.

## Project Entities

Extracted from text via the gazetteer:

- Symptom, Finding, Diagnosis, Exam, Treatment, BodyPart

Per-patient, not extracted from the entity gazetteer:

- **Age, Sex** -- self-extracted from the case text by `clean_cases.py` (see Data Cleaning), not
  carried through from `cases.csv` unchanged; the upstream columns had confirmed errors.
- **Person** -- identity is the row itself (`case_id` in `cases_clean.csv`, one row per real
  patient); there's no separate name/ID field beyond that.

## AI-assisted development

Claude and OpenAI Codex were used as development assistants during implementation,
refactoring, debugging and documentation. Their suggestions and generated changes
were reviewed and validated with the project's automated and browser tests before
being incorporated.
