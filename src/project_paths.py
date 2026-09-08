"""Shared pipeline inputs: extraction and visualization use real patient rows."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "data/interim/cases_clean.csv"
DEFAULT_DB = PROJECT_ROOT / "data/interim/gazetteer.db"
DEFAULT_METADATA = PROJECT_ROOT / "data/raw/metadata.csv"
