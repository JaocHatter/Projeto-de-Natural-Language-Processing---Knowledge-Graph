# Data directory

Follows the project layout: `raw` (untouched inputs) -> `external` (third-party sources and the
transformation that consumes them) -> `interim` (intermediate build products) -> `processed`
(final output).

Total on disk when fully built: **~9.7 GB**, almost all of it UMLS. Only `raw/` and
`processed/entities.csv` are tracked by git — see [Licensing and git](#licensing-and-git).

---

## What you must download

`external/filter_umls_mrsty.bash` does **not** download anything. Two UMLS release files must be
placed in `data/external/` first, under these exact names:

| File | Size | Used for |
|---|---|---|
| `umls-2026AA-mrconso.zip` | 490 MB | `MRCONSO.RRF` — every concept name, its CUI, source vocabulary and term type |
| `umls-2026AA-metathesaurus-full.zip` | 5.5 GB | `MRSTY.RRF` only — maps each CUI to its semantic type (TUI) |

The script extracts just those two `.RRF` members; the rest of the 5.5 GB archive is never read.
If `umls_work/MRCONSO.RRF` or `umls_work/MRSTY.RRF` already exist, the corresponding zip is not
needed and the extraction step is skipped.

### Where to get them

UMLS is free but **licensed** — you need an account and must accept the UMLS Metathesaurus License
Agreement before downloading.

1. Create a UMLS Terminology Services (UTS) account and request a license:
   <https://uts.nlm.nih.gov/uts/signup-login>
   (approval is generally quick, but it is not instant — request it before you need the data)
2. Sign in, then download both files from the **"UMLS Metathesaurus Precomputed Subsets"** section of
   the UMLS Knowledge Sources release page:
   <https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html>

   | Link on the page | Direct URL |
   |---|---|
   | *MRCONSO.RRF* | <https://download.nlm.nih.gov/umls/kss/2026AA/umls-2026AA-mrconso.zip> |
   | *UMLS Metathesaurus Full Subset* | <https://download.nlm.nih.gov/umls/kss/2026AA/umls-2026AA-metathesaurus-full.zip> |

   The direct URLs require an authenticated session — opening one while signed out redirects to the
   UTS login page rather than downloading. Sign in first, or click through from the release page.

   Do **not** download *Level 0 Subset* or *Full Release* instead: Level 0 omits SNOMED CT (the
   largest source here), and the Full Release is a much bigger archive containing the same `.RRF`
   files plus tooling this pipeline never uses.
3. Put both `.zip` files in `data/external/` — do not unzip them by hand. The names NLM gives them
   already match what the script expects, so no renaming is needed.

General UMLS documentation: <https://www.nlm.nih.gov/research/umls/index.html>

### Using a different UMLS release

`2026AA` is hardcoded at the top of the script. For any other release (`2026AB`, `2027AA`, …),
either rename your downloads to match, or edit these two lines in
`external/filter_umls_mrsty.bash`:

```bash
ZIP_MRCONSO="umls-2026AA-mrconso.zip"
ZIP_FULL="umls-2026AA-metathesaurus-full.zip"
```

NLM publishes each release under a predictable path, so for release `<REL>` the two files are:

```
https://download.nlm.nih.gov/umls/kss/<REL>/umls-<REL>-mrconso.zip
https://download.nlm.nih.gov/umls/kss/<REL>/umls-<REL>-metathesaurus-full.zip
```

Changing release also means rebuilding everything downstream — `umls_csvs/`, `gazetteer.db` and
`entities.csv` are all derived from these two files, and CUIs can change between releases.

---

## Running the transformation

The script uses **relative paths**, so it must be run from inside `data/external/`:

```bash
cd data/external
bash filter_umls_mrsty.bash
```

**Requirements:** `bash`, `unzip`, `awk`, `sort`, `join` (all standard on Linux/macOS) and roughly
**10 GB of free disk space** beyond the two zips.

It is a long job: it scans the 2.2 GB `MRCONSO.RRF` line by line, then sorts and joins a 353 MB
intermediate once per semantic type. Progress is printed per step.

**What it filters:** English, non-suppressed terms from `SNOMEDCT_US`, `MSH`, `LNC`, `RXNORM`,
`ICD10CM` and `MTH`, restricted to the 14 semantic types listed in the `TIPOS` array. Set
`VOCABS=""` to keep all vocabularies.

> Note: the vocabulary filter is an unanchored regex, so `MTH` also admits `MTHSPL`/`MTHICD9` and
> `ICD10CM` admits `CCSR_ICD10CM`. `src/clinical_kg/gazetteer/build.py` compensates by matching vocabularies
> exactly (`ALLOWED_VOCABS`), so the leak is filtered out downstream rather than here.

### Then, from the project root

```bash
clinical-kg build-gazetteer     # umls_csvs/ -> interim/gazetteer.db
clinical-kg extract-entities    # raw/cases.csv -> processed/entities.csv
```

---

## Contents

| Path | Origin | Size |
|---|---|---|
| `raw/` | **Input.** PMC open-access clinical case reports (see below) | 224 KB |
| `external/*.zip` | **Downloaded by you** (above) | 6.0 GB |
| `external/filter_umls_mrsty.bash` | **Tracked in git.** The transformation | 8 KB |
| `external/umls_work/` | *Generated.* Extracted `.RRF` files and intermediates | 3.1 GB |
| `external/umls_csvs/` | *Generated.* One CSV per semantic type + consolidated | 380 MB |
| `interim/gazetteer.db` | *Generated* by `src/clinical_kg/gazetteer/build.py`. 1,097,541 indexed terms | 289 MB |
| `processed/entities.csv` | *Generated* by `src/clinical_kg/extraction/entities.py`. 3,287 entity spans | 352 KB |

### `raw/` — the case corpus

| File | Rows | Contents |
|---|---|---|
| `cases.csv` | 56 | `article_id`, `case_id`, `case_text`, `age`, `gender` — the text the NER runs on |
| `metadata.csv` | 50 | Article-level: DOI, journal, PMCID, year, title, license, MeSH terms |
| `data_dictionary.csv` | 45 | Field definitions for the upstream dataset |

Articles are PubMed Central open-access case reports under CC BY (see the `license` and `link`
columns in `metadata.csv`). `data_dictionary.csv` documents the upstream distribution's parquet
files (`cases.parquet`, `metadata.parquet`, `case_images.parquet`, `abstracts.parquet`,
`captions_and_labels.csv`); the CSVs here are a subset extracted from it.

> **TODO:** record the exact upstream dataset name, version and DOI/URL here — it is not
> recoverable from the files in this repository, and reproducibility depends on it.

Note that `age` and `gender` are already structured columns, so the pipeline carries them through
as case-level metadata rather than extracting them from text.

---

## Licensing and git

`.gitignore` excludes `external/umls_work/`, `external/umls_csvs/`, `external/*.zip` and `*.db`.
This is deliberate on two grounds: size (~9.5 GB), and the fact that **UMLS content may not be
redistributed** — each user must accept the license and download it themselves. Everything
excluded is reproducible from the two zips by re-running the commands above.

Tracked: `raw/`, `external/filter_umls_mrsty.bash`, `processed/entities.csv`, and this README.

---

## Directory tree

Files marked *(generated)* are created by the pipeline — you do not download or create them.

```
data
├── README.md
├── raw/                                     <- input, tracked in git
│   ├── cases.csv
│   ├── data_dictionary.csv
│   └── metadata.csv
├── external/
│   ├── filter_umls_mrsty.bash               <- the transformation, tracked in git
│   ├── umls-2026AA-mrconso.zip              <- YOU DOWNLOAD (490 MB)
│   ├── umls-2026AA-metathesaurus-full.zip   <- YOU DOWNLOAD (5.5 GB)
│   ├── umls_work/                           (generated)
│   │   ├── MRCONSO.RRF                      extracted from the mrconso zip (2.2 GB)
│   │   ├── MRSTY.RRF                        extracted from the full zip (203 MB)
│   │   ├── conso_filtrado.tsv               filtered MRCONSO (353 MB)
│   │   ├── conso_sorted.tsv                 same, sorted by CUI for the join (353 MB)
│   │   └── cuis_T0*.txt                     one CUI list per semantic type (14 files)
│   └── umls_csvs/                           (generated)
│       ├── umls_clinico_todos.csv           consolidated, 1,666,483 rows
│       └── umls_T0*.csv                     one per semantic type (14 files)
├── interim/
│   └── gazetteer.db                         (generated) indexed SQLite gazetteer
└── processed/
    └── entities.csv                         (generated) final entity annotations
```

The semantic-type CSVs and their TUI meanings are tabulated in the project [`README.md`](../README.md).

> If you see a `conso_filtered.tsv` in `umls_work/`, it is a manually renamed leftover — the script
> writes `conso_filtrado.tsv` and will recreate that name on the next run.
