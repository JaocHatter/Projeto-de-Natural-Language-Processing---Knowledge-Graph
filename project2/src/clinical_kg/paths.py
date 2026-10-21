"""Shared pipeline locations: extraction and visualization use real patient rows.

Single source of truth for every path in the package -- modules import
PROJECT_ROOT from here rather than re-deriving it from their own __file__,
which used to break whenever a module moved.
"""

import os
from pathlib import Path

# src/clinical_kg/paths.py -> src/clinical_kg -> src -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = PROJECT_ROOT / "data/interim/cases_clean.csv"
# O gazetteer é grande e licenciado (não vai para o git). Cada pessoa pode deixá-lo em
# data/interim/gazetteer.db ou apontar outro lugar com a variável CLINICAL_KG_GAZETTEER.
DEFAULT_DB = Path(os.environ.get("CLINICAL_KG_GAZETTEER") or PROJECT_ROOT / "data/interim/gazetteer.db")
DEFAULT_METADATA = PROJECT_ROOT / "data/raw/metadata.csv"
DEFAULT_ENTITIES = PROJECT_ROOT / "data/processed/entities.csv"
DEFAULT_MEASUREMENTS = PROJECT_ROOT / "data/processed/measurements.csv"
DEFAULT_RELATIONS = PROJECT_ROOT / "data/processed/relations.csv"
DEFAULT_TOKEN_FREQ = PROJECT_ROOT / "data/interim/token_frequency.csv"
DEFAULT_POS_LEXICON = PROJECT_ROOT / "data/interim/pos_lexicon.csv"
DEFAULT_GOLD_RELATIONS = PROJECT_ROOT / "data/interim/gold_relations.csv"
DEFAULT_UMLS_RELATIONS = PROJECT_ROOT / "data/interim/umls_relations.csv"
DEFAULT_UMLS_ZIP = PROJECT_ROOT / "data/external/umls-2026AA-metathesaurus-full.zip"
