# Natural Language Processing - Knowledge Graph Project

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
    ├── src                <- Source code in programming language or system (e.g., Cytoscape)
    │   └── README.md      <- Basic installation/execution instructions
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

`src/build_gazetteer.py` loads these CSVs into an indexed SQLite database used for
longest-match entity extraction:

```bash
bash data/external/filter_umls_mrsty.bash   # produces data/external/umls_csvs/
python3 src/build_gazetteer.py              # produces data/interim/gazetteer.db
```

## Entity Extraction

`src/extract_entities.py` runs longest-match extraction over `data/raw/cases.csv`:

```bash
python3 src/extract_entities.py             # produces data/processed/entities.csv
```

Yields **3,287 entities across 56 cases**, typed as Treatment / Finding / Diagnosis / BodyPart /
Exam / Symptom. Each row carries character offsets, CUI, TUI and source vocabulary. See `PLAN.md`
for the TUI-to-entity mapping.

## Visualization

`src/app.py` is a Streamlit viewer that highlights entities inline, for either a corpus case or new
text you paste or upload. Extraction runs live (~35 ms per case) through the same `extract()`
function the CLI uses, so the app and the pipeline can never disagree.

```bash
pip install -r requirements.txt
streamlit run src/app.py            # opens http://localhost:8501
```

Hover any highlight for its CUI, semantic type and source vocabulary. The page also shows per-type
counts, a sortable entity table, and a CSV download in the same schema as
`data/processed/entities.csv`.

> The pipeline modules (`build_gazetteer.py`, `extract_entities.py`) are stdlib-only and need no
> dependencies; only the viewer requires `streamlit` and `pandas`.

## Project Entities

Extracted from text via the gazetteer:

- Symptom, Finding, Diagnosis, Exam, Treatment, BodyPart

Carried through from the `cases.csv` columns (not extracted):

- Person, Age, Sex
