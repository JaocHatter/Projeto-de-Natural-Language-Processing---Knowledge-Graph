# src

Fonte do pipeline: um único pacote Python instalável, `clinical_kg`.

## Instalação/execução básica

Roda de um checkout limpo, sem instalar nada — o `Makefile` (na raiz do
projeto) coloca `src` no `PYTHONPATH`:

```bash
make test      # roda a suíte de testes
make all       # clean-cases -> entities -> measurements -> relations
make app       # abre o visualizador Streamlit
make help      # lista todos os alvos
```

Instalar o pacote adiciona o comando `clinical-kg` (equivalente a
`python3 -m clinical_kg`):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
clinical-kg                       # lista todos os comandos
```

Todo o pipeline de extração é stdlib-only; apenas o visualizador Streamlit
precisa de dependências de terceiros (`streamlit`, `pandas`).

## Rodando o pipeline completo

```bash
clinical-kg clean-cases                    # data/raw/cases.csv -> cases_clean.csv
clinical-kg build-gazetteer                 # requer UMLS local, ver data/README.md
clinical-kg extract-entities
clinical-kg extract-measurements
clinical-kg extract-relations
clinical-kg build-umls-relations            # camada ontológica opcional (requer UMLS)
```

A UMLS é conteúdo licenciado e não é redistribuída neste repositório — cada
pessoa baixa e gera localmente (`data/README.md` tem o passo a passo e os
links de download).

## Estrutura de `clinical_kg`

```text
clinical_kg/
├── paths.py            # caminhos únicos do projeto
├── cli.py               # um dispatcher sobre cada etapa do pipeline
├── corpus/               # limpeza de casos e registros de paciente
├── gazetteer/            # dicionário de termos UMLS (SQLite)
├── extraction/            # entidades e medições
├── relations/             # relações clínicas tipadas (CRF)
│   ├── core/               # o algoritmo (segmentação, POS, pesos, CRF, correferência)
│   ├── extract.py          # pipeline de produção
│   ├── ontology.py         # segunda fonte de evidência (relações da UMLS)
│   └── tools/              # anotação e avaliação (não roda em produção)
├── graph/                 # modelo do grafo, export e visualizador
└── app/                   # front-end Streamlit
```

Arquitetura completa, camadas e schema do grafo:
[`clinical_kg/README.md`](clinical_kg/README.md). Processo de extração de
relações em detalhe: [`clinical_kg/relations/README.md`](clinical_kg/relations/README.md).
