#!/usr/bin/env python3
"""Pipeline do Projeto 2, um subcomando por etapa, na ordem em que rodam.

    python -m clinical_kg.learning.pipeline tabelas     # n-gramas, PMI e LLR

Lê  : data/interim/cases_clean.csv, data/processed/entities.csv
Grava: data/interim/tabela_ngramas.csv, data/interim/tabela_pmi.csv

Os candidatos de relação são os do Projeto 1 (`relations.extract.analyze_case`
sem limiar), de modo que os dois sistemas sempre vejam os mesmos pares.
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

from clinical_kg.paths import DEFAULT_CASES, DEFAULT_ENTITIES, PROJECT_ROOT
from clinical_kg.relations.extract import analyze_case, load_entities

from .ngramas import Associacoes, contar_ngramas, tokenizar

TABELA_NGRAMAS = PROJECT_ROOT / "data/interim/tabela_ngramas.csv"
TABELA_PMI = PROJECT_ROOT / "data/interim/tabela_pmi.csv"
N_MAX = 3


def ler_casos(caminho: Path = DEFAULT_CASES) -> list[dict]:
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def texto_entre(texto: str, cand: dict) -> str:
    """Trecho entre a cabeça e a cauda; sem cabeça (paciente), desde a frase."""
    inicio = cand["head"]["end"] if cand["head"] else cand["sent_start"]
    return texto[inicio:cand["tail"]["start"]]


def textos_entre_candidatos(casos, entidades) -> list[str]:
    """O texto entre as duas entidades de cada candidato do Projeto 1."""
    saida = []
    for caso in casos:
        ents = entidades.get(caso["case_id"], [])
        for cand in analyze_case(caso["case_text"], ents, {}, threshold=float("-inf")):
            saida.append(texto_entre(caso["case_text"], cand))
    return saida


def gravar_csv(caminho: Path, colunas, linhas):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(colunas)
        w.writerows(linhas)


def cmd_tabelas(args):
    casos = ler_casos(args.casos)
    entidades = load_entities(args.entidades)
    corpus = [c["case_text"] for c in casos]
    entre = textos_entre_candidatos(casos, entidades)

    n_corpus, n_entre = collections.Counter(), collections.Counter()
    for t in corpus:
        n_corpus.update(contar_ngramas(tokenizar(t), N_MAX))
    for t in entre:
        n_entre.update(contar_ngramas(tokenizar(t), N_MAX))

    linhas = [(" ".join(g), len(g), c, n_entre.get(g, 0))
              for g, c in n_corpus.items() if len(g) >= 2 and c >= 2]
    linhas.sort(key=lambda r: (-r[2], r[0]))
    gravar_csv(TABELA_NGRAMAS, ["ngrama", "n", "contagem_corpus", "contagem_entre"], linhas)

    assoc = Associacoes(corpus)
    ranking = assoc.ranking("pmi")
    gravar_csv(TABELA_PMI, ["palavra_1", "palavra_2", "contagem", "pmi", "llr"],
               [(a, b, c, f"{pmi:.3f}", f"{assoc.llr(a, b):.1f}") for a, b, c, pmi in ranking])

    print(f"{len(casos)} casos, {len(entre)} trechos entre entidades")
    print(f"{TABELA_NGRAMAS.relative_to(PROJECT_ROOT)}: {len(linhas)} n-gramas (n>=2, contagem>=2)")
    print(f"{TABELA_PMI.relative_to(PROJECT_ROOT)}: {len(ranking)} bigramas (contagem>={assoc.contagem_minima})")
    print("\nTop 10 por PMI:")
    for a, b, c, pmi in ranking[:10]:
        print(f"  {a} {b:<20} contagem={c:<4} PMI={pmi:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="comando", required=True)
    t = sub.add_parser("tabelas", help="n-gramas, PMI e LLR do corpus")
    t.add_argument("--casos", type=Path, default=DEFAULT_CASES)
    t.add_argument("--entidades", type=Path, default=DEFAULT_ENTITIES)
    t.set_defaults(func=cmd_tabelas)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
