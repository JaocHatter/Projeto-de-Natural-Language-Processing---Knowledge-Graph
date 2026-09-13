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

Arquitetura completa, camadas e schema do grafo:
[`clinical_kg/README.md`](clinical_kg/README.md).
