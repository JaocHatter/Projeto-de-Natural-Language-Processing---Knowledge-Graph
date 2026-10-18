#!/usr/bin/env python3
"""Pipeline do Projeto 2, um subcomando por etapa, na ordem em que rodam.

    python -m clinical_kg.learning.pipeline tabelas    # n-gramas, PMI e LLR do corpus
    python -m clinical_kg.learning.pipeline treinar    # validação, ablações e modelo final
    python -m clinical_kg.learning.pipeline extrair    # relações dos 50 casos com o modelo
    python -m clinical_kg.learning.pipeline comparar   # P1 x P2: grafo, exemplos, gatilhos, figuras

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

comparar
    Lê  : relations.csv (P1), relacoes_ml.csv (P2), metricas_*.csv e o gold
    Grava: data/processed/comparacao_{metricas,arestas,exemplos,gatilhos}.csv e as figuras
           em assets/images/ (as figuras exigem matplotlib; `--sem-figuras` as dispensa)

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


def relacoes_p2(modelo: ModeloAprendido, caso: dict, texto: str, entidades: list[dict],
                freq: dict) -> tuple[list[dict], int]:
    """As relações que o modelo aceita em um caso, e quantos candidatos olhou.

    É a única porta de decisão do P2: o subcomando `extrair` e o visualizador
    (inclusive para um caso novo colado pelo usuário) passam por aqui.
    """
    linhas, candidatos = [], 0
    for cand in analyze_case(texto, entidades, freq, threshold=float("-inf")):
        candidatos += 1
        estruturais, entre = atributos_estruturais(texto, cand)
        relacao, probabilidade = modelo.prever(estruturais, entre)
        if relacao == NENHUMA:
            continue
        linha = linha_ml({**caso, "case_text": texto}, cand, relacao, probabilidade)
        assert texto[cand["tail"]["start"]:cand["tail"]["end"]] == linha["tail_text"]
        linhas.append(linha)
    return linhas, candidatos


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
        novas, n = relacoes_p2(modelo, caso, textos[caso["case_id"]],
                               entidades.get(caso["case_id"], []), freq)
        linhas += novas
        candidatos += n

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


def combinar_relacoes(p1: list[dict], p2: list[dict]) -> list[dict]:
    """Une as relações dos dois métodos, marcando cada uma com `origem`.

    Duas linhas são a mesma relação quando têm a mesma cabeça, a mesma cauda e o
    mesmo tipo. `origem` vale "p1", "p2" ou "ambos"; para "ambos" ficam os dados
    (e o `assertion_status`) da linha do Projeto 2.
    """
    por_chave = {chave_relacao(l): {**l, "origem": "p1"} for l in p1}
    for l in p2:
        k = chave_relacao(l)
        por_chave[k] = {**l, "origem": "ambos" if k in por_chave else "p2"}
    return list(por_chave.values())


# --------------------------------------------------------------------------
# comparar: P1 (regras) contra P2 (aprendido), no grafo e no texto
# --------------------------------------------------------------------------
PASTA_FIGURAS = PROJECT_ROOT / "assets/images"
COR_P1, COR_P2, COR_NEUTRA = "#ea580c", "#16a34a", "#94a3b8"   # as mesmas cores do visualizador

ARTIGOS = {"the", "a", "an"}
PALAVRAS_FUNCAO = ARTIGOS | {"of", "and", "or", "in", "on", "at", "for", "by", "as",
                             "is", "are", "be", "it", "this", "that"}


def jaccard(a, b) -> float:
    """|A ∩ B| / |A ∪ B|; 0 quando os dois conjuntos são vazios."""
    a, b = set(a), set(b)
    uniao = a | b
    return len(a & b) / len(uniao) if uniao else 0.0


def tabela_gatilhos(entre: list[str], minimo: int = 3):
    """Frases (bigramas) que mais aparecem entre duas entidades, com PMI e LLR.

    Marca as que contêm uma palavra do léxico de 140 gatilhos escrito à mão no
    Projeto 1. Descarta bigramas só de palavras de função ("of the") e os que
    começam com artigo ("the patient"), que são gramática e não gatilho.

    Devolve (linhas, vistos): `vistos` são as palavras do léxico do P1 que o corpus
    confirma, isto é, que aparecem entre entidades pelo menos `minimo` vezes.
    """
    lexico = regras_p1.TRIGGER_LEXICON
    contagem = collections.Counter()
    for texto in entre:
        contagem.update(contar_ngramas(tokenizar(texto), 2))
    assoc = Associacoes(entre, contagem_minima=minimo)
    linhas = []
    for gram, c in contagem.items():
        if len(gram) != 2 or c < minimo:
            continue
        if gram[0] in ARTIGOS or all(w in PALAVRAS_FUNCAO for w in gram):
            continue
        palavra = next((w for w in gram if w in lexico), None)
        linhas.append({"ngrama": " ".join(gram), "contagem_entre": c,
                       "pmi": round(assoc.pmi(*gram), 2), "llr": round(assoc.llr(*gram), 1),
                       "no_lexico_p1": int(palavra is not None),
                       "categoria_p1": lexico[palavra] if palavra else ""})
    linhas.sort(key=lambda l: (-l["contagem_entre"], l["ngrama"]))
    vistos = {g[0] for g, c in contagem.items() if len(g) == 1 and c >= minimo and g[0] in lexico}
    return linhas, vistos


def ler_relacoes(caminho: Path) -> list[dict]:
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def arestas_do_grafo(casos, textos, entidades, medicoes, linhas):
    """Arestas clínicas do grafo agregado (uma por par de conceitos e tipo, por caso)."""
    from clinical_kg.graph.model import CLINICAL_RELATIONS, build_graph
    por_caso = collections.defaultdict(list)
    for l in linhas:
        por_caso[l["case_id"]].append(l)
    arestas = set()
    for caso in casos:
        cid = caso["case_id"]
        g = build_graph(caso, textos[cid], entidades.get(cid, []), medicoes.get(cid, []),
                        relations=por_caso.get(cid, []))
        arestas |= {(cid, e["source"], e["target"], e["relation"])
                    for e in g["edges"] if e["relation"] in CLINICAL_RELATIONS}
    return arestas


def quem_acerta(gold: str, p1: str, p2: str) -> str:
    a, b = p1 == gold, p2 == gold
    return ("os dois acertam" if a and b else "só P1 acerta" if a else
            "só P2 acerta" if b else "os dois erram")


def exemplos_do_gold(exemplos, textos):
    """Os 413 pares julgados, com o que cada método decidiu FORA de amostra."""
    p1 = {e.chave: r for e, r in prever_validando(exemplos, textos, "p1")}
    p2 = {e.chave: r for e, r in prever_validando(exemplos, textos, "completo")}
    return [{"caso": e.caso, "cabeca": e.cabeca, "cauda": e.cauda,
             "texto_entre": " ".join(e.entre), "gold": e.relacao,
             "p1": p1[e.chave], "p2": p2[e.chave],
             "quem_acerta": quem_acerta(e.relacao, p1[e.chave], p2[e.chave])}
            for e in exemplos]


def desenhar_figuras(metricas, por_caso, por_tipo, gatilhos):
    """Quatro figuras para o README e os slides (cores do visualizador: P1 laranja, P2 verde)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    def gravar(fig, nome):
        fig.tight_layout()
        fig.savefig(PASTA_FIGURAS / nome, dpi=160)
        plt.close(fig)

    # 1. F1 por variante
    nomes = [m["variante"] for m in metricas]
    valores = [float(m["f1"]) for m in metricas]
    cores = [COR_P1 if n == "p1" else COR_P2 for n in nomes]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    barras = ax.bar(nomes, valores, color=cores)
    ax.bar_label(barras, fmt="%.3f", padding=2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (detecção de relação)")
    ax.set_title("F1 por variante, deixando um caso de fora", loc="left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    gravar(fig, "f1_variantes.png")

    # 2. F1 caso a caso: P1 contra o modelo completo
    casos = sorted({r["caso"] for r in por_caso})
    f1 = {(r["variante"], r["caso"]): float(r["f1"]) for r in por_caso}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = range(len(casos))
    ax.bar([i - 0.2 for i in x], [f1[("p1", c)] for c in casos], 0.4, color=COR_P1, label="P1 (regras)")
    ax.bar([i + 0.2 for i in x], [f1[("completo", c)] for c in casos], 0.4, color=COR_P2,
           label="P2 (aprendido)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("_01", "") for c in casos], rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("F1 em cada um dos 10 casos julgados", loc="left")
    ax.legend(frameon=False)
    gravar(fig, "f1_por_caso.png")

    # 3. relações por tipo no corpus inteiro
    tipos = [l["relacao"] for l in por_tipo if l["relacao"] != "TOTAL"]
    dados = {l["relacao"]: l for l in por_tipo}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = range(len(tipos))
    ax.bar([i - 0.27 for i in x], [int(dados[t]["p1"]) for t in tipos], 0.27, color=COR_P1, label="P1")
    ax.bar(list(x), [int(dados[t]["p2"]) for t in tipos], 0.27, color=COR_P2, label="P2")
    ax.bar([i + 0.27 for i in x], [int(dados[t]["ambos"]) for t in tipos], 0.27, color=COR_NEUTRA,
           label="idênticas (mesmo par e tipo)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tipos, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("relações nos 50 casos")
    ax.set_title("Mesmo corpus, grafos diferentes", loc="left")
    ax.legend(frameon=False)
    gravar(fig, "relacoes_por_tipo.png")

    # 4. frases mais frequentes entre entidades, marcando as que o P1 já tinha como gatilho
    topo = gatilhos[:20][::-1]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh([g["ngrama"] for g in topo], [int(g["contagem_entre"]) for g in topo],
            color=[COR_P1 if int(g["no_lexico_p1"]) else COR_NEUTRA for g in topo])
    ax.set_xlabel("vezes entre duas entidades")
    ax.set_title("Frases entre entidades (laranja: já no léxico de gatilhos do P1)", loc="left", fontsize=10)
    gravar(fig, "frases_entre_entidades.png")


def cmd_comparar(args):
    casos = ler_casos(args.casos)
    textos = textos_dos_casos(args.casos)
    entidades = entidades_lf(args.entidades)
    medicoes = collections.defaultdict(list)
    if DEFAULT_MEASUREMENTS.exists():
        for m in ler_relacoes(DEFAULT_MEASUREMENTS):
            medicoes[m["case_id"]].append(m)
    for caminho in (DEFAULT_RELATIONS, RELACOES_ML, SAIDA / "metricas_validacao.csv"):
        if not caminho.exists():
            sys.exit(f"ERRO: falta {caminho}. Rode `treinar` e `extrair` antes.")

    l1, l2 = ler_relacoes(DEFAULT_RELATIONS), ler_relacoes(RELACOES_ML)
    k1, k2 = {chave_relacao(l) for l in l1}, {chave_relacao(l) for l in l2}

    # 1) relações por tipo (par + tipo iguais), no texto
    tipos = sorted({k[3] for k in k1 | k2})
    por_tipo = []
    for t in [*tipos, "TOTAL"]:
        a = {k for k in k1 if t == "TOTAL" or k[3] == t}
        b = {k for k in k2 if t == "TOTAL" or k[3] == t}
        por_tipo.append({"relacao": t, "p1": len(a), "p2": len(b), "ambos": len(a & b),
                         "so_p1": len(a - b), "so_p2": len(b - a), "jaccard": round(jaccard(a, b), 3)})
    gravar_csv(SAIDA / "comparacao_arestas.csv", list(por_tipo[0]),
               [list(l.values()) for l in por_tipo])

    # 2) o grafo agregado por conceito
    g1 = arestas_do_grafo(casos, textos, entidades, medicoes, l1)
    g2 = arestas_do_grafo(casos, textos, entidades, medicoes, l2)

    def conectados(arestas):
        return {(a[0], a[1]) for a in arestas} | {(a[0], a[2]) for a in arestas}

    # 3) gatilhos: frases entre entidades contra o léxico do P1
    entre = textos_entre_candidatos(
        [{"case_id": c["case_id"], "case_text": textos[c["case_id"]]} for c in casos], entidades)
    gatilhos, vistos = tabela_gatilhos(entre)
    topo = gatilhos[:30]
    com_gatilho = sum(1 for g in topo if g["no_lexico_p1"])
    gravar_csv(SAIDA / "comparacao_gatilhos.csv",
               ["ngrama", "contagem_entre", "pmi", "llr", "no_lexico_p1", "categoria_p1"],
               [list(g.values()) for g in gatilhos[:100]])

    # 4) o gold: o que cada método decide fora de amostra
    exemplos = carregar_exemplos(args.casos, args.entidades)
    linhas_ex = exemplos_do_gold(exemplos, textos)
    divergentes = [l for l in linhas_ex if l["quem_acerta"] in ("só P1 acerta", "só P2 acerta")]
    gravar_csv(SAIDA / "comparacao_exemplos.csv", list(linhas_ex[0]),
               [list(l.values()) for l in sorted(divergentes, key=lambda l: (l["quem_acerta"], l["caso"]))])
    contagem = collections.Counter(l["quem_acerta"] for l in linhas_ex)

    metricas = {m["variante"]: m for m in ler_relacoes(SAIDA / "metricas_validacao.csv")}
    total = por_tipo[-1]
    ng1, ng2 = len(g1), len(g2)
    n1, n2 = len(conectados(g1)), len(conectados(g2))
    linhas = [
        ("relações extraídas (50 casos)", total["p1"], total["p2"], "pares únicos (cabeça, cauda, tipo); relacoes_ml.csv pode ter linhas repetidas do mesmo par"),
        ("relações por caso", round(total["p1"] / len(casos), 1), round(total["p2"] / len(casos), 1), ""),
        ("relações idênticas (mesmo par e tipo)", total["ambos"], total["ambos"], f"Jaccard {total['jaccard']}"),
        ("arestas do grafo (conceitos agregados)", ng1, ng2, "arestas clínicas únicas por caso"),
        ("arestas idênticas no grafo", len(g1 & g2), len(g1 & g2), f"Jaccard {jaccard(g1, g2):.3f}"),
        ("grau médio dos nós conectados", round(2 * ng1 / n1, 2), round(2 * ng2 / n2, 2),
         "2 x arestas / nós com alguma relação clínica"),
        ("tipos de relação presentes", len({k[3] for k in k1}), len({k[3] for k in k2}), f"de {len(tipos)}"),
        ("F1 no gold (deixa-um-caso-fora)", metricas["p1"]["f1"], metricas["completo"]["f1"],
         "413 pares julgados à mão, 10 casos"),
        ("tipo acertado no gold", metricas["p1"]["tipo_acertado"], metricas["completo"]["tipo_acertado"],
         "413 pares julgados à mão, 10 casos"),
        (f"gatilhos do P1 vistos entre entidades (>=3x)", f"{len(vistos)} de {len(regras_p1.TRIGGER_LEXICON)}", "",
         "palavras do léxico escrito à mão que o corpus confirma"),
        ("bigramas mais frequentes com palavra-gatilho do P1", f"{com_gatilho} de {len(topo)}", "",
         "o resto são frases que o P1 não tinha como gatilho"),
    ]
    gravar_csv(SAIDA / "comparacao_metricas.csv", ["metrica", "p1", "p2", "nota"], linhas)

    print(f"{'':<46}{'P1':>10}{'P2':>10}")
    for nome, a, b, nota in linhas:
        print(f"{nome:<46}{str(a):>10}{str(b):>10}   {nota}")
    print("\nNo gold, fora de amostra:")
    for nome in ("os dois acertam", "só P2 acerta", "só P1 acerta", "os dois erram"):
        print(f"  {nome:<18}{contagem[nome]:>5}")

    if args.sem_figuras:
        return
    try:
        por_caso = ler_relacoes(SAIDA / "metricas_por_caso.csv")
        desenhar_figuras(list(metricas.values()), por_caso, por_tipo, gatilhos)
        print(f"\nfiguras em {PASTA_FIGURAS.relative_to(PROJECT_ROOT)}/")
    except ImportError:
        print("\n(matplotlib não instalado: figuras puladas; `pip install matplotlib`)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="comando", required=True)
    for nome, ajuda, func in (("tabelas", "n-gramas, PMI e LLR do corpus", cmd_tabelas),
                              ("treinar", "validação, ablações e modelo final", cmd_treinar),
                              ("extrair", "relações dos 50 casos com o modelo", cmd_extrair),
                              ("comparar", "P1 x P2: grafo, exemplos, gatilhos e figuras", cmd_comparar)):
        s = sub.add_parser(nome, help=ajuda)
        s.add_argument("--casos", type=Path, default=DEFAULT_CASES)
        s.add_argument("--entidades", type=Path, default=DEFAULT_ENTITIES)
        if nome == "treinar":
            from clinical_kg.paths import DEFAULT_GOLD_RELATIONS
            s.add_argument("--gold", type=Path, default=DEFAULT_GOLD_RELATIONS)
        if nome == "extrair":
            s.add_argument("--modelo", type=Path, default=MODELO_JSON)
            s.add_argument("--saida", type=Path, default=RELACOES_ML)
        if nome == "comparar":
            s.add_argument("--sem-figuras", action="store_true", help="não desenhar os PNG")
        s.set_defaults(func=func)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
