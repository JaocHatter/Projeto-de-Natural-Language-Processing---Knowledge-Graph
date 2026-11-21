"""Shared pipeline locations: extraction and visualization use real patient rows.

Single source of truth for every path in the package -- modules import
PROJECT_ROOT from here rather than re-deriving it from their own __file__,
which used to break whenever a module moved.

Organização de data/ (Projeto 3), por fonte e por tipo de arquivo:

    raw/multicare/              os 50 casos e os metadados originais (nunca modificados)
    raw/maccrobat/              o zip do MACCROBAT2020
    external/                   UMLS (zips, script de filtro) e MultiCaRe completo (Parquet)
    interim/multicare/          casos limpos, gold de relações, relações do UMLS
    interim/maccrobat/          o MACCROBAT convertido para o esquema do P1
    interim/cache/              embeddings e respostas do LLM
    interim/gazetteer.db        gazetteer UMLS (licenciado, local de cada pessoa)
    processed/extracao/         entidades, medidas e relações do P1, P2 e P3
    processed/modelos/          modelos treinados (P2, P3)
    processed/resultados/       tabelas dos experimentos: gold/, maccrobat/, ida_volta/, p2/
    processed/app/              corpora da aplicação Paciente Virtual
"""

import os
from pathlib import Path

# src/clinical_kg/paths.py -> src/clinical_kg -> src -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"

# raw/
RAW_MULTICARE = DATA / "raw/multicare"
DEFAULT_RAW_CASES = RAW_MULTICARE / "cases.csv"
DEFAULT_METADATA = RAW_MULTICARE / "metadata.csv"
MACCROBAT_ZIP = DATA / "raw/maccrobat/MACCROBAT2020.zip"

# external/ (o UMLS fica onde cada pessoa já o tem: data/external/ e data/interim/gazetteer.db)
DEFAULT_UMLS_ZIP = DATA / "external/umls-2026AA-metathesaurus-full.zip"
MULTICARE_METADATA_PARQUET = DATA / "external/multicare/metadata.parquet"
MULTICARE_CASES_PARQUET = DATA / "external/multicare/cases.parquet"

# interim/
INTERIM_MULTICARE = DATA / "interim/multicare"
DEFAULT_CASES = INTERIM_MULTICARE / "cases_clean.csv"
DEFAULT_GOLD_RELATIONS = INTERIM_MULTICARE / "gold_relations.csv"
DEFAULT_UMLS_RELATIONS = INTERIM_MULTICARE / "umls_relations.csv"
DEFAULT_TOKEN_FREQ = INTERIM_MULTICARE / "token_frequency.csv"
DEFAULT_POS_LEXICON = INTERIM_MULTICARE / "pos_lexicon.csv"
TABELA_NGRAMAS = INTERIM_MULTICARE / "tabela_ngramas.csv"
TABELA_PMI = INTERIM_MULTICARE / "tabela_pmi.csv"
INTERIM_MACCROBAT = DATA / "interim/maccrobat"
MACCROBAT_ENTIDADES = INTERIM_MACCROBAT / "maccrobat_entidades.csv"
MACCROBAT_RELACOES = INTERIM_MACCROBAT / "maccrobat_relacoes.csv"
MACCROBAT_MULTICARE = INTERIM_MACCROBAT / "maccrobat_multicare.csv"
CACHE = DATA / "interim/cache"
EMBEDDINGS_CACHE = CACHE / "embeddings"
LLM_CACHE = CACHE / "llm_cache.jsonl"
LLM_VOTOS = CACHE / "llm_votos.csv"
# O gazetteer é grande e licenciado (não vai para o git). Cada pessoa pode deixá-lo em
# data/interim/gazetteer.db ou apontar outro lugar com a variável CLINICAL_KG_GAZETTEER.
DEFAULT_DB = Path(os.environ.get("CLINICAL_KG_GAZETTEER") or DATA / "interim/gazetteer.db")

# processed/
PASTA_EXTRACAO = DATA / "processed/extracao"
DEFAULT_ENTITIES = PASTA_EXTRACAO / "entities.csv"
DEFAULT_MEASUREMENTS = PASTA_EXTRACAO / "measurements.csv"
DEFAULT_RELATIONS = PASTA_EXTRACAO / "relations.csv"          # P1 (regras)
RELACOES_ML = PASTA_EXTRACAO / "relacoes_ml.csv"              # P2 (regressão logística)
RELACOES_P3 = PASTA_EXTRACAO / "relacoes_p3.csv"              # P3 (P3-G)
PASTA_MODELOS = DATA / "processed/modelos"
MODELO_P2_JSON = PASTA_MODELOS / "modelo.json"
MODELO_P3_JSON = PASTA_MODELOS / "modelo_p3.json"
PASTA_RESULTADOS = DATA / "processed/resultados"
RESULTADOS_GOLD = PASTA_RESULTADOS / "gold"                   # variantes no gold do MultiCaRe, custo do LLM
RESULTADOS_MACCROBAT = PASTA_RESULTADOS / "maccrobat"         # E0 e avaliação externa
RESULTADOS_IDA_VOLTA = PASTA_RESULTADOS / "ida_volta"         # pacientes virtuais reextraídos
RESULTADOS_P2 = PASTA_RESULTADOS / "p2"                       # saídas do pipeline do P2, se for rodado
PASTA_APP = DATA / "processed/app"
