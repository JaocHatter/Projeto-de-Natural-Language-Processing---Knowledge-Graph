# src

Pipeline source: a single installable Python package, `clinical_kg`.

## Basic installation/run

Runs from a clean checkout with nothing installed — the `Makefile` (at the
project root) puts `src` on `PYTHONPATH` for you:

```bash
make test      # run the test suite
make all       # clean-cases -> entities -> measurements -> relations
make app       # open the Streamlit viewer
make help      # list every target
```

Installing the package adds the `clinical-kg` command (equivalent to
`python3 -m clinical_kg`):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
clinical-kg                       # lists every command
```

The whole extraction pipeline is stdlib-only; only the Streamlit viewer
needs third-party dependencies (`streamlit`, `pandas`).

## Running the full pipeline

```bash
clinical-kg clean-cases                    # data/raw/cases.csv -> cases_clean.csv
clinical-kg build-gazetteer                 # requires local UMLS, see data/README.md
clinical-kg extract-entities
clinical-kg extract-measurements
clinical-kg extract-relations
clinical-kg build-umls-relations            # optional ontological layer (requires UMLS)
```

UMLS is licensed content and is not redistributed in this repository — each
person downloads it and builds it locally (`data/README.md` has the
step-by-step instructions and download links).

## `clinical_kg` structure

```text
clinical_kg/
├── paths.py            # single source of truth for project paths
├── cli.py               # one dispatcher over every pipeline stage
├── corpus/               # case cleaning and patient records
├── gazetteer/            # UMLS term dictionary (SQLite)
├── extraction/            # entities and measurements
├── relations/             # typed clinical relations (CRF)
│   ├── core/               # the algorithm (segmentation, POS, weights, CRF, coreference)
│   ├── extract.py          # production pipeline
│   ├── ontology.py         # second evidence source (UMLS relations)
│   └── tools/              # annotation and evaluation (not run in production)
├── graph/                 # graph model, export and viewer
└── app/                   # Streamlit front end
```

Full architecture, layers and graph schema:
[`clinical_kg/README.md`](clinical_kg/README.md). Relation extraction process
in detail: [`clinical_kg/relations/README.md`](clinical_kg/relations/README.md).
