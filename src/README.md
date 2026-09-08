# Patient knowledge graph

Run from the repository root with Python 3.11+:

```bash
pip install -r requirements.txt
python3 src/clean_cases.py
# Build this only if the SQLite gazetteer is missing (requires UMLS CSVs):
python3 src/build_gazetteer.py
streamlit run src/app.py
```

`clean_cases.py` writes `data/interim/cases_clean.csv`. If that file already
contains reviewed changes, use it directly rather than regenerating it.
The application never rewrites input or processed CSVs. Its corpus is the
50 cleaned patient rows, not the 56 upstream fragments/cohort rows. The
`source_case_ids`, extraction methods and original demographics are preserved.

## Data flow

```text
raw/cases.csv → clean_cases.py → interim/cases_clean.csv
                                      │
                     gazetteer.db ────┤
                                      ▼
                   extract_measurements.analyze_case()
                       entities + linked measurements
                                      │
                                      ▼
                       graph_builder.build_graph()
                           ├── graph_panel.py → graph_view.py → Cytoscape.js
                           └── graph_export.py → JSON / GraphML / CSV ZIP
```

`project_paths.py` is the shared source for the cleaned-corpus, gazetteer and
article-metadata paths. Entity extraction, measurement extraction, the app and
graph export all default to the cleaned corpus. The SQLite gazetteer remains a
dictionary, not a patient graph database. The viewer and graph exporter run
fresh extraction; processed CSVs are used only by explicit CSV export mode.

## Graph schema (version 2)

| Source | Relation | Target | Meaning |
|---|---|---|---|
| Article | `REPORTS_PERSON` | Person | Article reports this cleaned patient |
| Person | `HAS_AGE` | Age | Age in years from the cleaning output |
| Person | `HAS_SEX` | Sex | Sex from the cleaning output (`gender` column) |
| Person | `MENTIONS_DIAGNOSIS` | Diagnosis concept/mention | Diagnosis term occurs in the patient's report |
| Person | `MENTIONS_TREATMENT` | Treatment concept/mention | Treatment term occurs in the report |
| Person | `MENTIONS_EXAM` | Exam concept/mention | Exam term occurs in the report |
| Person | `MENTIONS_SYMPTOM` | Symptom concept/mention | Symptom term occurs in the report |
| Person | `MENTIONS_FINDING` | Finding concept/mention | Finding term occurs in the report |
| Person | `MENTIONS_BODY_PART` | BodyPart concept/mention | Anatomy term occurs in the report |
| Concept/mention | `ASSOCIATED_WITH_MEASUREMENT` | Measurement | Heuristic association with an exact text occurrence |
| Person | `CONTAINS_UNLINKED_MEASUREMENT` | Measurement | Value was extracted without a resolved association |

These clinical relation names express **text mentions**, with
`assertion_status=not_assessed`. No negation, temporal context, experiencer,
diagnostic confirmation or causal relation is inferred. Even `same_sentence`
is a heuristic, not a calibrated confidence score. All measurement methods are
visible initially; `window_fallback` edges are dashed and can be filtered out.

Identity and provenance:

- Person: `person:{case_id}`. Merged `_P1` rows have one Person and preserve all
  original fragment IDs in `source_case_ids`.
- Age/Sex: `age:{case_id}` and `sex:{case_id}`. Their provenance records the
  field, extraction method, upstream value and source. Missing/unknown values
  produce no demographic node. Age `0` is a valid known value.
- Concept: `concept:{cui}`. Repeated mentions are aggregated by CUI, with all
  offsets and surface forms retained. Concept nodes carry the clinical
  `entity_type`, TUI set and vocabularies.
- Occurrence mode: `mention:{case_id}:{start}:{end}:{cui}`. Measurement edges
  attach to the exact occurrence, including later merged fragments.
- Measurement: `measurement:{case_id}:{start}:{end}`. Equal values at different
  positions remain distinct; entity/measurement spans may overlap.
- Article: `article:{article_id}`, optionally shown from `data/raw/metadata.csv`.

ID components are URL-escaped. Offsets index Python characters in the untouched
**cleaned, concatenated text**, with an exclusive end. Graph metadata includes
`text_sha256`, `case_id`, `source_case_ids`, `data_source`, `mode` and warnings.
The builder validates evidence and edge endpoints. Raw-fragment offsets cannot
be substituted for offsets in the merged text.

`preferred_term` is the matched gazetteer term, not a guaranteed canonical name
for the whole CUI. Aggregated labels prefer an observed PT term deterministically.
`linked_entity_cui` and `linked_entity_id` enrich measurement output without
replacing `linked_entity_start/end`, which retain the actual occurrence.

## Viewer

The Annotated text, Knowledge Graph and Tables tabs share the selected patient
and extraction. Select a node or edge in the graph (or use the keyboard-accessible
node selector) to inspect metadata, source excerpts and links. Measurements show
both their own span and their linked entity occurrence. The Tables tab contains
entities, measurements and an explicit relation table, with a selection filter.

Filters control categories, Age/Sex, measurements, link methods and article
visibility. Relation labels are shown on selected edges and can be enabled for
the whole graph. Selecting a concept, measurement or edge emphasizes its direct
neighbors. Measurements default to **On node selection**: concepts reveal only
their associated values, Person reveals unlinked values, and selecting a value
keeps its owner's values open. Clicking the background hides measurements again.
**Hide all** disables measurements entirely; automatic Show all is no longer
available. Revisiting a case/filter view does not restore old expansions.
Search by name/CUI and press Enter to focus, use zoom controls, or clear the
selection with the button/Escape. Node labels adapt to zoom and appear on hover.
Visible exports and the node cap
follow the actual expanded view; complete exports retain all measurements.
**Group by IS_A category** adds SemanticType nodes and explicit IS_A edges and
arranges concepts around their categories. Multi-type concepts retain every
classification but are positioned with their first sorted category. Grouping
overrides the layout selector; disable it to restore the usual layouts.
These classes describe project entity types, not the complete UMLS hierarchy.
Force-directed,
hierarchical and circular layouts are available, with zoom, pan, drag and fit.
The node cap defaults to 100; the page reports truncation and never leaves
dangling edges or relabels filtered associations as unlinked. Exports explicitly
distinguish the visible graph from the complete patient graph.

New pasted/uploaded cases use the age/sex extraction functions from
`clean_cases.py`, and a content-derived patient ID. Unknown demographics stay
unknown. Corpus/database/metadata revision keys invalidate caches after changes.

Cytoscape.js 3.33.1 is bundled locally with its MIT license in
`src/assets/cytoscape/`. No CDN, Node build step, NetworkX or Neo4j is required.
The component uses Streamlit v2 with `isolate_styles=False` for Cytoscape pointer
hit testing. Component HTML/JS are fixed; external labels are passed as data and
text excerpts are HTML-escaped. CSS classes use the `kg-` prefix.

## Export CLI

By default the CLI uses the same cleaned-text extraction as the viewer:

```bash
python3 src/graph_export.py --case-id PMC6083636_P1 --include-article --out /tmp/patient.json
python3 src/graph_export.py --case-id PMC6083636_P1 --occurrences --out /tmp/patient.graphml
python3 src/graph_export.py --case-id PMC4630775_01 --out /tmp/newborn.zip
```

Use `--cases`, `--db`, or `--metadata` to select other inputs. Existing exports
are protected unless `--force` is given. JSON retains nested data. GraphML uses
directed edges and string attributes; lists/dictionaries are JSON-encoded.
CSV ZIP contains `nodes.csv`, `edges.csv` and `metadata.json`, with nested fields
JSON-encoded. Use a CSV parser to preserve quoted commas.

For reproducible offline export from aligned CSVs, explicitly choose `--from-csv`
or provide CSV paths. This mode does not require the gazetteer or Streamlit:

```bash
# Generate aligned annotations without overwriting reviewed processed files:
python3 src/extract_entities.py --out /tmp/entities-clean.csv
python3 src/extract_measurements.py --out /tmp/measurements-clean.csv
python3 src/graph_export.py --case-id PMC6083636_P1 --entities /tmp/entities-clean.csv --measurements /tmp/measurements-clean.csv --out /tmp/patient-offline.json
```

CSV mode validates IDs against the selected corpus, text offsets and resolved
measurement links. A CSV from the upstream 56-row corpus is rejected against the
cleaned 50-patient corpus with instructions to regenerate it. Legacy schemas
without `preferred_term` or linked CUI still work when their IDs/spans align.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Core tests use the standard library. Integration tests also run when Streamlit
and the local gazetteer/corpus are available. Coverage includes the six typed
clinical relations, corrected neonatal age, missing demographics, merged patient
identity/offsets, stale raw annotations, filters and exports. Corpus integration
builds graphs in both modes for all 50 cleaned patients.

For the browser smoke test, install Playwright in a temporary directory, start
Streamlit on port 8517, then run the test in another terminal:

```bash
npm install --prefix /tmp/kg-browser-test --ignore-scripts playwright@1.55.0
streamlit run src/app.py --server.port=8517 --server.headless=true
node tests/browser_smoke.cjs /tmp/kg-browser-test/node_modules/playwright
```

The test uses `/usr/bin/google-chrome` by default (`KG_CHROME` can override it),
blocks external HTTP requests, and checks canvas selection, linked text, table
filtering, download, demographics and occurrence mode. Screenshot:
`/tmp/knowledge-graph-smoke.png`.
