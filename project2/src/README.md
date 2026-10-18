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

## Project 2: n-grams, PMI and logistic regression

An extension of Project 1 that replaces the hand-written rules' decision with a statistical model
(no LLMs, no embeddings). The new code lives in `clinical_kg/learning/`:

```text
learning/
├── ngramas.py     # n-grams, PMI, LLR and a bigram language model
├── modelo.py      # per-pair features, hand-written logistic regression, stacking, ModeloAprendido
└── pipeline.py    # subcommands: tabelas, treinar, extrair, comparar
```

Everything runs with one command, in the right order (the UMLS gazetteer is not needed):

```bash
bash run_all.sh                 # tabelas -> treinar -> extrair -> comparar (~1 min)
bash run_all.sh --teste         # run the tests first
bash run_all.sh --notebook      # also execute pipelines/notebooks/projeto2.ipynb
make p2                         # the same, without bash
```

On Windows, `run_all.sh` runs in Git Bash or WSL. Each stage also runs on its own:

```bash
PYTHONPATH=src python -m clinical_kg.learning.pipeline tabelas   # data/interim/tabela_*.csv
PYTHONPATH=src python -m clinical_kg.learning.pipeline treinar   # metrics, weights and modelo.json
PYTHONPATH=src python -m clinical_kg.learning.pipeline extrair   # data/processed/relacoes_ml.csv
PYTHONPATH=src python -m clinical_kg.learning.pipeline comparar  # comparacao_*.csv and figures
```

Optional dependencies (the core uses only the standard library; the viewer uses `streamlit` and `pandas`):

```bash
pip install -e ".[verificacao]"   # scikit-learn: cross-checks the hand-written logistic regression
pip install -e ".[figuras]"       # matplotlib: draws the PNGs written by `comparar`
```

**P1 × P2 viewer** (graphs coloured by method, metrics, PMI and model weights):

```bash
make app-p2        # or: PYTHONPATH=src python -m streamlit run src/clinical_kg/app/projeto2.py
```

Corpus cases use the files already extracted. The *Caso novo (ao vivo)* option needs the UMLS
gazetteer: keep it at `data/interim/gazetteer.db`, or point to another location with the
`CLINICAL_KG_GAZETTEER` environment variable. The gazetteer is licensed content and is **not tracked by git**.
