#!/usr/bin/env python3
"""Ida e volta: grafo -> ficha -> texto -> extratores -> fatos recuperados.

    python -m clinical_kg.paciente.ida_volta --corpus digestivo multicare50 maccrobat

Como a ficha é o gold por construção, o texto gerado é um caso anotado de graça. Mede-se:

    fidelidade      fatos da ficha recuperados com a polaridade certa (presente / negado)
    erro_polaridade fatos recuperados com a polaridade trocada (o escopo de negação errou)
    alucinacao      conceitos extraídos que não estão na ficha (nem nas palavras fixas do modelo)
    relacoes        relações esperadas da ficha recuperadas por cada extrator (P1, P2, P3)

Entidades e asserção são as do P1 para todos; os extratores diferem na decisão de relação. Com o
narrador determinístico, o texto é limpo e a medida isola os extratores; com o narrador LLM, ela
mede também o quanto o LLM foi fiel à ficha (o que ele omitiu ou inventou).

Precisa do gazetteer UMLS (para reextrair as entidades do texto gerado).
"""

import argparse
import collections
import json
import sys

from clinical_kg.paths import RESULTADOS_IDA_VOLTA

from .corpus import assercao_das_entidades, carregar
from .ficha import DIFICULDADES, montar_ficha
from .narrador import NarradorLLM, NarradorModelo
from .perfil import montar_perfis

# As palavras fixas do narrador-modelo: o que o gazetteer achar nelas não é alucinação.
MOLDURA = ("A 50-year-old man presented with fever. He denied pain. Physical examination revealed "
           "edema. Computed tomography showed a mass. Computed tomography was performed. The diagnosis "
           "of pneumonia was made. He was treated with surgery.")
PACIENTE = "PAC"
CATEGORIAS_CONTADAS = ("Symptom", "Finding", "Exam", "Treatment", "Diagnosis")


def fatos_esperados(ficha) -> dict[str, bool]:
    esperados = {i.cui: i.presente for i in ficha.itens}
    esperados[ficha.dx_cui] = True
    return esperados


def relacoes_esperadas(ficha) -> set[tuple]:
    """(frozenset do par, tipo): o que um extrator perfeito tiraria do texto completo."""
    rel = set()
    for i in ficha.de("Symptom"):
        rel.add((frozenset((PACIENTE, i.cui)), "HAS_SYMPTOM"))
    for i in ficha.de("Finding"):
        rel.add((frozenset((PACIENTE, i.cui)), "HAS_FINDING"))
    rel.add((frozenset((PACIENTE, ficha.dx_cui)), "HAS_DIAGNOSIS"))
    for e in ficha.de("Exam"):
        for c in e.resultado_cuis:
            rel.add((frozenset((e.cui, c)), "REVEALED_BY"))
    for t in ficha.de("Treatment"):
        rel.add((frozenset((PACIENTE, t.cui)), "TREATED_WITH"))
    return rel


def chave_relacao(r: dict) -> tuple:
    return frozenset((r.get("head_cui") or PACIENTE, r["tail_cui"])), r["relation"]


class Extratores:
    """Entidades do gazetteer (P1) e relações por três decisores: regras do P1, modelo do P2, modelo do P3."""

    def __init__(self):
        import sqlite3

        from clinical_kg.learning.modelo import ModeloAprendido
        from clinical_kg.paths import DEFAULT_DB, MODELO_P2_JSON, MODELO_P3_JSON
        from clinical_kg.relations.extract import DEFAULT_FREQ, load_frequencies

        self.con = sqlite3.connect(DEFAULT_DB)
        self.freq = load_frequencies(DEFAULT_FREQ)
        self.modelos = {}
        for nome, caminho in (("p2", MODELO_P2_JSON), ("p3", MODELO_P3_JSON)):
            with open(caminho, encoding="utf-8") as f:
                self.modelos[nome] = ModeloAprendido.de_dict(json.load(f))
        self.moldura = {e["cui"] for e in self.entidades(MOLDURA)}

    def entidades(self, texto: str) -> list[dict]:
        from clinical_kg.extraction.entities import DEFAULT_MAX_N, extract
        from clinical_kg.hibrido.assercao import propagar_coordenacao
        ents = extract(texto, self.con, DEFAULT_MAX_N)
        estados = assercao_das_entidades(texto, ents)
        for e, s, s3 in zip(ents, estados, propagar_coordenacao(texto, ents, estados)):
            e["assertion_status"], e["assertion_p3"] = s, s3
        return ents

    def relacoes(self, texto: str, entidades: list[dict]) -> dict[str, list[dict]]:
        from clinical_kg.learning.pipeline import relacoes_p2
        from clinical_kg.relations.extract import analyze_case, to_row

        caso = {"case_id": "virtual", "article_id": ""}
        ents = [{**e, "case_id": "virtual"} for e in entidades]
        saida = {"p1": [to_row(caso, r) for r in analyze_case(texto, ents, self.freq)]}
        for nome, modelo in self.modelos.items():
            saida[nome], _ = relacoes_p2(modelo, caso, texto, ents, self.freq)
        return saida


def medir(ficha, texto: str, ext: Extratores) -> dict:
    esperados = fatos_esperados(ficha)
    ents = ext.entidades(texto)
    linha = {"dx": ficha.dx_nome, "dificuldade": ficha.dificuldade, "fatos": len(esperados)}
    # Asserção do P1 como foi entregue e com a propagação pela coordenação do P3.
    for sufixo, campo in (("", "assertion_status"), ("_p3", "assertion_p3")):
        extraidos = {}
        for e in ents:
            if e["entity_type"] in CATEGORIAS_CONTADAS or e["cui"] in esperados:
                negado = e[campo] == "negated"
                extraidos[e["cui"]] = extraidos.get(e["cui"], False) or not negado   # presente se alguma menção afirma
        recuperados = sum(1 for c, p in esperados.items() if c in extraidos and extraidos[c] == p)
        polaridade = sum(1 for c, p in esperados.items() if c in extraidos and extraidos[c] != p)
        linha[f"fidelidade{sufixo}"] = recuperados / len(esperados)
        linha[f"erro_polaridade{sufixo}"] = polaridade / len(esperados)
    inventados = {c for c in extraidos if c not in esperados and c not in ext.moldura}
    linha["conceitos_extraidos"] = len(extraidos)
    linha["alucinacao"] = len(inventados) / len(extraidos) if extraidos else 0.0
    esperadas = relacoes_esperadas(ficha)
    linha["relacoes_esperadas"] = len(esperadas)
    for nome, rels in ext.relacoes(texto, ents).items():
        obtidas = {chave_relacao(r) for r in rels}
        pares = {p for p, _ in obtidas}
        linha[f"rel_{nome}"] = len(esperadas & obtidas) / len(esperadas) if esperadas else 0.0
        linha[f"par_{nome}"] = sum(1 for p, _ in esperadas if p in pares) / len(esperadas) if esperadas else 0.0
    return linha


def experimento(corpus_nomes, narradores, minimo_casos: int = 3, sementes=(0,)) -> list[dict]:
    corpus = carregar(corpus_nomes)
    perfis = montar_perfis(corpus, minimo_casos)
    ext = Extratores()
    linhas = []
    for perfil in sorted(perfis.values(), key=lambda p: p.cui):
        for dif in DIFICULDADES:
            for s in sementes:
                ficha = montar_ficha(perfil, perfis, dif, s, corpus.nomes)
                for narrador in narradores:
                    texto = narrador.completo(ficha)
                    linhas.append({"narrador": narrador.nome, **medir(ficha, texto, ext), "texto": texto})
    return linhas


def resumir(linhas: list[dict]) -> list[dict]:
    grupos = collections.defaultdict(list)
    for l in linhas:
        grupos[(l["narrador"], l["dificuldade"])].append(l)
        grupos[(l["narrador"], "todas")].append(l)
    metricas = ["fidelidade", "erro_polaridade", "fidelidade_p3", "erro_polaridade_p3", "alucinacao",
                "rel_p1", "rel_p2", "rel_p3", "par_p1", "par_p2", "par_p3"]
    saida = []
    for (narrador, dif), ls in sorted(grupos.items()):
        saida.append({"narrador": narrador, "dificuldade": dif, "fichas": len(ls),
                      **{m: round(sum(l[m] for l in ls) / len(ls), 4) for m in metricas}})
    return saida


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from clinical_kg.learning.pipeline import gravar_csv

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", nargs="+", default=["digestivo", "multicare50", "maccrobat"])
    ap.add_argument("--minimo-casos", type=int, default=3)
    ap.add_argument("--llm", action="store_true", help="também o narrador LLM (precisa do provedor)")
    args = ap.parse_args(argv)

    narradores = [NarradorModelo()]
    if args.llm:
        from clinical_kg.llm.provedor import provedor_padrao
        narradores.append(NarradorLLM(provedor_padrao()))
    linhas = experimento(args.corpus, narradores, args.minimo_casos)
    gravar_csv(RESULTADOS_IDA_VOLTA / "p3_ida_volta.csv", list(linhas[0]),
               [[f"{v:.4f}" if isinstance(v, float) else v for v in l.values()] for l in linhas])
    resumo = resumir(linhas)
    gravar_csv(RESULTADOS_IDA_VOLTA / "p3_ida_volta_resumo.csv", list(resumo[0]), [list(r.values()) for r in resumo])
    print(f"{len(linhas)} fichas narradas e reextraídas")
    print(f"{'narrador':<9}{'dific.':<9}{'n':>5}{'fidel.':>8}{'polar.':>8}{'fid.P3':>8}{'pol.P3':>8}"
          f"{'aluc.':>8}{'rel P1':>8}{'rel P2':>8}{'rel P3':>8}")
    for r in resumo:
        print(f"{r['narrador']:<9}{r['dificuldade']:<9}{r['fichas']:>5}{r['fidelidade']:>8.3f}"
              f"{r['erro_polaridade']:>8.3f}{r['fidelidade_p3']:>8.3f}{r['erro_polaridade_p3']:>8.3f}"
              f"{r['alucinacao']:>8.3f}{r['rel_p1']:>8.3f}{r['rel_p2']:>8.3f}{r['rel_p3']:>8.3f}")


if __name__ == "__main__":
    main()
