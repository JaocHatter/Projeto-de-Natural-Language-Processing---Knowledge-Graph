#!/usr/bin/env bash
# Regenera todo o Projeto 2, na ordem certa, a partir dos dados já extraídos pelo Projeto 1.
#
#   bash run_all.sh               # tabelas -> treinar -> extrair -> comparar
#   bash run_all.sh --teste       # roda antes a suíte de testes
#   bash run_all.sh --rapido      # pula a supervisão fraca (a parte lenta): ~1 min em vez de ~6
#   bash run_all.sh --notebook    # executa também pipelines/notebooks/projeto2.ipynb
#
# Entradas (já no repositório):  data/interim/cases_clean.csv, data/interim/gold_relations.csv,
#                                data/processed/entities.csv, data/processed/relations.csv
# Saídas:                        data/interim/tabela_*.csv, data/processed/{metricas_*,pesos_modelo,
#                                modelo.json, relacoes_ml, comparacao_*}.csv e assets/images/*.png
#
# Não precisa do gazetteer UMLS (só o caso novo ao vivo do visualizador precisa).
# No Windows, rode com Git Bash ou WSL. Alternativa sem bash: `make p2`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # acentos corretos no terminal do Windows

# Escolhe um Python que funcione (no Windows, "python3" pode ser só um atalho da Microsoft Store).
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidato in python3 python py; do
    if "$candidato" -c "import sys" >/dev/null 2>&1; then PYTHON="$candidato"; break; fi
  done
fi
[ -n "$PYTHON" ] || { echo "ERRO: não encontrei o Python (defina PYTHON=/caminho/do/python)." >&2; exit 1; }

TESTE=0; NOTEBOOK=0; RAPIDO=0
for argumento in "$@"; do
  case "$argumento" in
    --teste) TESTE=1 ;;
    --notebook) NOTEBOOK=1 ;;
    --rapido) RAPIDO=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "opção desconhecida: $argumento (use --teste, --rapido, --notebook ou --help)" >&2; exit 2 ;;
  esac
done

for arquivo in data/interim/cases_clean.csv data/interim/gold_relations.csv \
               data/processed/entities.csv data/processed/relations.csv; do
  [ -f "$arquivo" ] || { echo "ERRO: falta $arquivo (saída do Projeto 1)." >&2; exit 1; }
done

passo() { printf '\n== %s ==\n' "$1"; }
inicio=$SECONDS

if [ "$TESTE" = 1 ]; then
  passo "0. Testes"
  "$PYTHON" -m unittest discover -s tests
fi

passo "1/4  Tabelas de n-gramas, PMI e LLR"
"$PYTHON" -m clinical_kg.learning.pipeline tabelas

passo "2/4  Treinar: validação deixando um caso de fora, ablações e modelo final"
if [ "$RAPIDO" = 1 ]; then
  "$PYTHON" -m clinical_kg.learning.pipeline treinar --sem-fraco
else
  "$PYTHON" -m clinical_kg.learning.pipeline treinar
fi

passo "3/4  Extrair: relações dos 50 casos com o modelo"
"$PYTHON" -m clinical_kg.learning.pipeline extrair

passo "4/4  Comparar: P1 x P2 (grafo, exemplos, gatilhos e figuras)"
"$PYTHON" -m clinical_kg.learning.pipeline comparar

if [ "$NOTEBOOK" = 1 ]; then
  passo "Notebook: executando projeto2.ipynb"
  "$PYTHON" - <<'FIM'
import nbformat
from nbclient import NotebookClient
caminho = "pipelines/notebooks/projeto2.ipynb"
nb = nbformat.read(caminho, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3").execute()
nbformat.write(nb, caminho)
print("notebook executado e salvo")
FIM
fi

printf '\nPronto em %ss.\n' "$((SECONDS - inicio))"
echo "Para ver os grafos:  $PYTHON -m streamlit run src/clinical_kg/app/projeto2.py"
echo "(caso novo ao vivo: aponte CLINICAL_KG_GAZETTEER para o gazetteer.db)"
