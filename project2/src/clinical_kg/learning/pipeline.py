#!/usr/bin/env python3
"""Pipeline do Projeto 2, um subcomando por etapa, na ordem em que rodam.

    python -m clinical_kg.learning.pipeline tabelas    # n-gramas, PMI e LLR do corpus
    python -m clinical_kg.learning.pipeline treinar    # validação, ablações e modelo final
    python -m clinical_kg.learning.pipeline extrair    # relações dos 50 casos com o modelo

tabelas
    Lê  : data/interim/cases_clean.csv, data/processed/entities.csv
    Grava: data/interim/tabela_ngramas.csv, data/interim/tabela_pmi.csv

treinar
    Lê  : os mesmos + data/interim/gold_relations.csv (413 pares julgados à mão)
    Grava: data/processed/metricas_validacao.csv, metricas_por_caso.csv,
           metricas_por_relacao.csv, pesos_modelo.csv e modelo.json

extrair
    Lê  : data/processed/modelo.json + casos e entidades
    Grava: data/processed/relacoes_ml.csv, no esquema de relations.csv do Projeto 1,
           de modo que `graph.model.build_graph` o consome sem mudanças

Os candidatos de relação são os do Projeto 1 (`relations.extract.analyze_case`
sem limiar), de modo que os dois sistemas sempre vejam os mesmos pares.

Validação: deixa um caso fora (leave-one-case-out). Em cada rodada o modelo, as
contagens de PMI e os modelos de linguagem são refeitos SEM o caso de teste, então
nenhum número é medido sobre texto que o modelo já leu.
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

from clinical_kg.paths import (DEFAULT_CASES, DEFAULT_ENTITIES, DEFAULT_MEASUREMENTS,
                               DEFAULT_RELATIONS, PROJECT_ROOT)
from clinical_kg.relations.core import features as regras_p1
from clinical_kg.relations.extract import (CSV_COLUMNS, DEFAULT_FREQ, analyze_case,
                                           load_entities, load_frequencies, to_row)

from .modelo import (NENHUMA, ModeloAprendido, UmContraTodosSklearn,
                     atributos_estruturais, carregar_exemplos, sklearn_disponivel)
from .ngramas import Associacoes, contar_ngramas, tokenizar

TABELA_NGRAMAS = PROJECT_ROOT / "data/interim/tabela_ngramas.csv"
TABELA_PMI = PROJECT_ROOT / "data/interim/tabela_pmi.csv"
SAIDA = PROJECT_ROOT / "data/processed"
MODELO_JSON = SAIDA / "modelo.json"
RELACOES_ML = SAIDA / "relacoes_ml.csv"
N_MAX = 3

# Cada variante liga uma combinação de peças. "completo" é o modelo proposto.
VARIANTES = {
    "estrutura": dict(usar_pmi=False, usar_lm=False),
    "pmi":       dict(usar_pmi=True,  usar_lm=False),
    "lm":        dict(usar_pmi=False, usar_lm=True),
    "completo":  dict(usar_pmi=True,  usar_lm=True),
}
DESCRICAO = {
    "p1": "Projeto 1 (regras à mão)",
    "estrutura": "só tipos, distância e pontuação",
    "pmi": "estrutura + PMI/LLR",
    "lm": "estrutura + modelo de linguagem (stacking)",
    "completo": "estrutura + PMI/LLR + modelo de linguagem",
    "completo_sklearn": "completo, com a regressão do scikit-learn",
}


# --------------------------------------------------------------------------
# Leitura e utilidades
# --------------------------------------------------------------------------
def ler_casos(caminho: Path = DEFAULT_CASES) -> list[dict]:
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def textos_dos_casos(caminho: Path = DEFAULT_CASES) -> dict[str, str]:
    # Sem newline="": os deslocamentos indexam o texto com quebras de linha LF.
    with open(caminho, encoding="utf-8") as f:
        return {r["case_id"]: r["case_text"] for r in csv.DictReader(f)}


def entidades_lf(caminho: Path = DEFAULT_ENTITIES) -> dict[str, list[dict]]:
    """Entidades por caso, com o texto de cada uma em quebras de linha LF.

    `entities.csv` guarda algumas entidades que cruzam uma quebra de linha com
    CR+LF, mas os deslocamentos (e o texto do caso) usam só LF. Sem normalizar, o
    grafo recusa a evidência porque o trecho do texto não bate com a entidade.
    """
    por_caso = load_entities(caminho)
    for linhas in por_caso.values():
        for e in linhas:
            e["surface_text"] = e["surface_text"].replace("\r\n", "\n")
    return por_caso


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


def prf(vp, fp, fn):
    p = vp / (vp + fp) if vp + fp else 0.0
    r = vp / (vp + fn) if vp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# --------------------------------------------------------------------------
# tabelas
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# treinar: validação leave-one-case-out, ablações e modelo final
# --------------------------------------------------------------------------
def prever_p1(exemplos):
    """A linha de base: o Projeto 1 como foi entregue (dispara se score >= limiar)."""
    limiar = regras_p1.WEIGHTS["decide.threshold"]
    return [(e, e.extras["p1_relacao"] if e.extras["p1_score"] >= limiar else NENHUMA)
            for e in exemplos]


def prever_validando(exemplos, textos, variante):
    """Previsões fora de amostra: um caso por vez fica de fora de TUDO (modelo,
    contagens de PMI e modelos de linguagem)."""
    if variante == "p1":
        return prever_p1(exemplos)
    flags = VARIANTES["completo" if variante == "completo_sklearn" else variante]
    saida = []
    for caso in sorted({e.caso for e in exemplos}):
        treino = [e for e in exemplos if e.caso != caso]
        teste = [e for e in exemplos if e.caso == caso]
        corpus = [t for c, t in textos.items() if c != caso]
        classificador = UmContraTodosSklearn() if variante == "completo_sklearn" else None
        modelo = ModeloAprendido(**flags).ajustar(treino, corpus, classificador)
        saida += [(e, modelo.prever_exemplo(e)) for e in teste]
    return saida


def medir(previsoes):
    """Detecção (há relação?), tipo certo entre as detectadas, F1 por caso e por relação."""
    def contar(itens):
        vp = sum(1 for e, p in itens if p != NENHUMA and e.eh_relacao)
        fp = sum(1 for e, p in itens if p != NENHUMA and not e.eh_relacao)
        fn = sum(1 for e, p in itens if p == NENHUMA and e.eh_relacao)
        return vp, fp, fn

    vp, fp, fn = contar(previsoes)
    p, r, f1 = prf(vp, fp, fn)
    detectadas = [(e, p_) for e, p_ in previsoes if p_ != NENHUMA and e.eh_relacao]
    tipo = (sum(1 for e, p_ in detectadas if p_ == e.relacao) / len(detectadas)) if detectadas else 0.0

    por_caso = {}
    for caso in sorted({e.caso for e, _ in previsoes}):
        por_caso[caso] = prf(*contar([(e, p_) for e, p_ in previsoes if e.caso == caso]))[2]

    por_relacao = {}
    for rel in sorted({e.relacao for e, _ in previsoes} - {NENHUMA}):
        rvp = sum(1 for e, p_ in previsoes if p_ == rel and e.relacao == rel)
        rfp = sum(1 for e, p_ in previsoes if p_ == rel and e.relacao != rel)
        rfn = sum(1 for e, p_ in previsoes if e.relacao == rel and p_ != rel)
        por_relacao[rel] = (rvp, rfp, rfn)
    return dict(vp=vp, fp=fp, fn=fn, p=p, r=r, f1=f1, tipo=tipo,
                por_caso=por_caso, por_relacao=por_relacao)


def confronto(por_caso, referencia, tol=1e-9):
    """Em quantos casos a variante vence, empata e perde da referência."""
    v = sum(1 for c in por_caso if por_caso[c] > referencia[c] + tol)
    d = sum(1 for c in por_caso if por_caso[c] < referencia[c] - tol)
    return v, len(por_caso) - v - d, d


def media_dp(valores):
    media = sum(valores) / len(valores)
    return media, (sum((x - media) ** 2 for x in valores) / len(valores)) ** 0.5


def cmd_treinar(args):
    exemplos = carregar_exemplos(args.casos, args.entidades, args.gold)
    textos = textos_dos_casos(args.casos)
    nomes = ["p1", *VARIANTES]
    if sklearn_disponivel():
        nomes.append("completo_sklearn")

    resultados = {}
    for nome in nomes:
        resultados[nome] = medir(prever_validando(exemplos, textos, nome))

    n_casos = len({e.caso for e in exemplos})
    print(f"{len(exemplos)} pares julgados, {n_casos} casos (deixa-um-caso-fora)\n")
    print(f"{'variante':<18}{'prec':>7}{'cob':>7}{'F1':>7}{'tipo':>7}   F1 por caso     vs P1 (V-E-D)")
    linhas, linhas_caso = [], []
    for nome, m in resultados.items():
        media, dp = media_dp(list(m["por_caso"].values()))
        v, e, d = confronto(m["por_caso"], resultados["p1"]["por_caso"])
        vs_p1 = "-" if nome == "p1" else f"{v}-{e}-{d}"
        vs_est = "-" if nome in ("p1", "estrutura") else "{}-{}-{}".format(
            *confronto(m["por_caso"], resultados["estrutura"]["por_caso"]))
        print(f"{nome:<18}{m['p']:>7.3f}{m['r']:>7.3f}{m['f1']:>7.3f}{m['tipo']:>7.3f}"
              f"   {media:.3f} ± {dp:.3f}   {vs_p1}")
        linhas.append((nome, DESCRICAO[nome], f"{m['p']:.4f}", f"{m['r']:.4f}", f"{m['f1']:.4f}",
                       f"{m['tipo']:.4f}", f"{media:.4f}", f"{dp:.4f}", vs_p1, vs_est))
        linhas_caso += [(nome, c, f"{f:.4f}") for c, f in m["por_caso"].items()]
    gravar_csv(SAIDA / "metricas_validacao.csv",
               ["variante", "descricao", "precisao", "cobertura", "f1", "tipo_acertado",
                "f1_caso_media", "f1_caso_dp", "casos_vs_p1_V-E-D", "casos_vs_estrutura_V-E-D"], linhas)
    gravar_csv(SAIDA / "metricas_por_caso.csv", ["variante", "caso", "f1"], linhas_caso)

    linhas_rel = []
    for nome in ("p1", "completo"):
        for rel, (vp, fp, fn) in resultados[nome]["por_relacao"].items():
            p, r, f1 = prf(vp, fp, fn)
            linhas_rel.append((nome, rel, vp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"))
    gravar_csv(SAIDA / "metricas_por_relacao.csv",
               ["variante", "relacao", "vp", "fp", "fn", "precisao", "cobertura", "f1"], linhas_rel)

    # Modelo final: todos os 413 pares e o texto de todos os casos.
    final = ModeloAprendido(**VARIANTES["completo"]).ajustar(exemplos, list(textos.values()))
    MODELO_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELO_JSON, "w", encoding="utf-8") as f:
        json.dump(final.para_dict(), f, ensure_ascii=False, separators=(",", ":"))
    recarregado = ModeloAprendido.de_dict(json.load(open(MODELO_JSON, encoding="utf-8")))
    iguais = sum(final.prever_exemplo(e) == recarregado.prever_exemplo(e) for e in exemplos)

    pesos = [(c, nome, f"{w:.4f}") for c, m in final.classificador.modelos.items()
             for nome, w in sorted(m.pesos.items(), key=lambda kv: -abs(kv[1]))]
    gravar_csv(SAIDA / "pesos_modelo.csv", ["classe", "atributo", "peso"], pesos)

    tamanho = MODELO_JSON.stat().st_size / 1024
    print(f"\n{MODELO_JSON.relative_to(PROJECT_ROOT)}: {tamanho:.0f} KB; "
          f"recarregado reproduz {iguais}/{len(exemplos)} previsões")
    print("Maiores pesos de TREATED_WITH:")
    for nome, w in final.classificador.modelos["TREATED_WITH"].pesos_principais(6):
        print(f"  {nome:<34}{w:>8.2f}")


# --------------------------------------------------------------------------
# extrair: o modelo decide todos os candidatos dos 50 casos
# --------------------------------------------------------------------------
def linha_ml(caso: dict, cand: dict, relacao: str, probabilidade: float) -> dict:
    """Uma linha no esquema de relations.csv, decidida pelo modelo aprendido.

    O que vem do Projeto 1 e não é decisão de relação é mantido: os pares, as
    posições e o `assertion_status` (negada, hipotética, histórica...), que é uma
    etapa à parte de escopo. O que era decisão das regras é trocado: o tipo vem do
    modelo, `score` passa a ser a probabilidade dele e o gatilho do P1 é apagado,
    para a linha não sugerir que uma regra a produziu.
    """
    linha = to_row(caso, cand)
    linha.update(relation=relacao, score=round(probabilidade, 3),
                 trigger_text="", trigger_start="", trigger_end="", trigger_category="",
                 rule_path="p2.regressao_logistica")
    return linha


def chave_relacao(linha: dict) -> tuple:
    return (linha["case_id"], str(linha["head_start"]), str(linha["tail_start"]), linha["relation"])


def cmd_extrair(args):
    modelo = ModeloAprendido.de_dict(json.load(open(args.modelo, encoding="utf-8")))
    casos = ler_casos(args.casos)
    textos = textos_dos_casos(args.casos)
    entidades = entidades_lf(args.entidades)
    freq = load_frequencies(DEFAULT_FREQ)

    linhas, candidatos = [], 0
    for caso in casos:
        texto = textos[caso["case_id"]]
        for cand in analyze_case(texto, entidades.get(caso["case_id"], []), freq,
                                 threshold=float("-inf")):
            candidatos += 1
            estruturais, entre = atributos_estruturais(texto, cand)
            relacao, probabilidade = modelo.prever(estruturais, entre)
            if relacao == NENHUMA:
                continue
            linha = linha_ml({**caso, "case_text": texto}, cand, relacao, probabilidade)
            assert texto[cand["tail"]["start"]:cand["tail"]["end"]] == linha["tail_text"]
            linhas.append(linha)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(linhas)

    por_tipo = collections.Counter(l["relation"] for l in linhas)
    print(f"{len(casos)} casos, {candidatos} candidatos, {len(linhas)} relações aceitas "
          f"({len(linhas) / max(candidatos, 1):.0%})")
    print(f"{args.saida.relative_to(PROJECT_ROOT)}\n")

    # O grafo do Projeto 1 precisa carregar o arquivo sem erro.
    from clinical_kg.graph.model import CLINICAL_RELATIONS, build_graph
    medicoes = collections.defaultdict(list)
    if DEFAULT_MEASUREMENTS.exists():
        with open(DEFAULT_MEASUREMENTS, newline="", encoding="utf-8") as f:
            for m in csv.DictReader(f):
                medicoes[m["case_id"]].append(m)
    por_caso = collections.defaultdict(list)
    for l in linhas:
        por_caso[l["case_id"]].append(l)
    arestas, avisos = 0, 0
    for caso in casos:
        cid = caso["case_id"]
        g = build_graph(caso, textos[cid], entidades.get(cid, []), medicoes.get(cid, []),
                        relations=por_caso.get(cid, []))
        arestas += sum(1 for e in g["edges"] if e["relation"] in CLINICAL_RELATIONS)
        avisos += len(g.get("warnings", []))
    print(f"grafo: {arestas} arestas clínicas, {avisos} avisos (carregou os {len(casos)} casos)\n")

    p1 = {}
    if DEFAULT_RELATIONS.exists():
        with open(DEFAULT_RELATIONS, newline="", encoding="utf-8") as f:
            p1 = {chave_relacao(l): l for l in csv.DictReader(f)}
    ml = {chave_relacao(l) for l in linhas}
    tipos_p1 = collections.Counter(k[3] for k in p1)
    print(f"{'relação':<18}{'P1':>6}{'P2':>6}")
    for r in sorted(set(por_tipo) | set(tipos_p1)):
        print(f"  {r:<16}{tipos_p1[r]:>6}{por_tipo[r]:>6}")
    print(f"  {'total':<16}{len(p1):>6}{len(linhas):>6}")
    if p1:
        print(f"\nrelações idênticas (mesmo par e tipo): {len(ml & set(p1))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="comando", required=True)
    for nome, ajuda, func in (("tabelas", "n-gramas, PMI e LLR do corpus", cmd_tabelas),
                              ("treinar", "validação, ablações e modelo final", cmd_treinar),
                              ("extrair", "relações dos 50 casos com o modelo", cmd_extrair)):
        s = sub.add_parser(nome, help=ajuda)
        s.add_argument("--casos", type=Path, default=DEFAULT_CASES)
        s.add_argument("--entidades", type=Path, default=DEFAULT_ENTITIES)
        if nome == "treinar":
            from clinical_kg.paths import DEFAULT_GOLD_RELATIONS
            s.add_argument("--gold", type=Path, default=DEFAULT_GOLD_RELATIONS)
        if nome == "extrair":
            s.add_argument("--modelo", type=Path, default=MODELO_JSON)
            s.add_argument("--saida", type=Path, default=RELACOES_ML)
        s.set_defaults(func=func)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
