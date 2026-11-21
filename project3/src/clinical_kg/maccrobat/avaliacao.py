"""Avaliação externa das etapas do P1 que não são relação: entidades, asserção e medidas.

O MACCROBAT foi anotado só por humanos e por outro grupo, com outro esquema; por isso estas
medidas dizem o quanto cada etapa do P1 generaliza para fora do MultiCaRe, e não o quanto ela
acerta no próprio MultiCaRe. Três tarefas:

    entidades   gazetteer UMLS do P1 contra os trechos das 6 categorias (exato e com sobreposição)
    assercao    escopo de negação/hipótese do P1 sobre as entidades anotadas, contra POLARITY
    medidas     valores numéricos de exame e doses: o P1 acha o valor? liga ao exame/medicamento certo?

As convenções dos dois esquemas não são as mesmas (o MACCROBAT separa "Transthoracic" de
"echocardiography"; o UMLS tem "transthoracic echocardiography"), então a medida exata subestima
o P1 e a medida com sobreposição o superestima: as duas são relatadas.
"""

import collections

from .mapeamento import CATEGORIA_P1, entidades_p1, relacoes_mapeadas

# Os grupos comparáveis: o MACCROBAT não separa sintoma de achado, então o P1 também é agrupado.
GRUPO_MACCROBAT = {
    "Disease_disorder": "Diagnosis",
    "Sign_symptom": "Symptom/Finding",
    "Diagnostic_procedure": "Exam",
    "Medication": "Treatment",
    "Therapeutic_procedure": "Treatment",
    "Biological_structure": "BodyPart",
}
GRUPO_P1 = {"Symptom": "Symptom/Finding", "Finding": "Symptom/Finding", "Diagnosis": "Diagnosis",
            "Exam": "Exam", "Treatment": "Treatment", "BodyPart": "BodyPart"}
GRUPOS = ("Diagnosis", "Symptom/Finding", "Exam", "Treatment", "BodyPart")


def _prf(vp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = vp / (vp + fp) if vp + fp else 0.0
    r = vp / (vp + fn) if vp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _sobrepoe(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# --------------------------------------------------------------------------
# Entidades
# --------------------------------------------------------------------------
def gold_entidades(doc) -> list[tuple[int, int, str]]:
    """(início, fim, grupo) de todos os trechos contínuos das 6 categorias (com sobreposição)."""
    return [(t.inicio, t.fim, GRUPO_MACCROBAT[t.tipo]) for t in doc.trechos.values()
            if t.tipo in GRUPO_MACCROBAT and not t.descontinuo]


def casar_entidades(gold: list[tuple], previstas: list[tuple], modo: str) -> tuple[set, set]:
    """Índices casados de cada lado. `modo`: exato, exato_tipo, sobreposicao, sobreposicao_tipo.

    Cada previsão casa com no máximo um gold e vice-versa (guloso, na ordem do texto), para que
    uma entidade longa do UMLS não conte como acerto de três trechos do MACCROBAT.
    """
    com_tipo = modo.endswith("_tipo")
    exato = modo.startswith("exato")
    usados_g, usados_p = set(), set()
    for i, (pi, pf, pt) in enumerate(previstas):
        for j, (gi, gf, gt) in enumerate(gold):
            if j in usados_g or (com_tipo and pt != gt):
                continue
            if (exato and (pi, pf) == (gi, gf)) or (not exato and _sobrepoe((pi, pf), (gi, gf))):
                usados_g.add(j)
                usados_p.add(i)
                break
    return usados_g, usados_p


def avaliar_entidades(documentos, extrair) -> list[dict]:
    """`extrair(texto) -> [entidade do P1]` (o `extract` do gazetteer, já com a conexão)."""
    modos = ("exato", "exato_tipo", "sobreposicao", "sobreposicao_tipo")
    contagem = collections.Counter()
    for doc in documentos:
        gold = gold_entidades(doc)
        previstas = [(e["start"], e["end"], GRUPO_P1[e["entity_type"]]) for e in extrair(doc.texto)]
        for modo in modos:
            for grupo in ((*GRUPOS, "TOTAL") if modo.endswith("_tipo") else ("TOTAL",)):
                g =[x for x in gold if grupo in ("TOTAL", x[2])]
                p = [x for x in previstas if grupo in ("TOTAL", x[2])]
                ok_g, ok_p = casar_entidades(g, p, modo)
                contagem[(modo, grupo, "vp_gold")] += len(ok_g)
                contagem[(modo, grupo, "vp_prev")] += len(ok_p)
                contagem[(modo, grupo, "gold")] += len(g)
                contagem[(modo, grupo, "prev")] += len(p)
    linhas = []
    for modo in modos:
        # Dentro de um grupo o tipo já é o mesmo dos dois lados: sem "_tipo", só o total difere.
        for grupo in ((*GRUPOS, "TOTAL") if modo.endswith("_tipo") else ("TOTAL",)):
            n_gold, n_prev = contagem[(modo, grupo, "gold")], contagem[(modo, grupo, "prev")]
            p = contagem[(modo, grupo, "vp_prev")] / n_prev if n_prev else 0.0
            r = contagem[(modo, grupo, "vp_gold")] / n_gold if n_gold else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            linhas.append({"modo": modo, "grupo": grupo, "gold": n_gold, "previstas": n_prev,
                           "precisao": p, "cobertura": r, "f1": f1})
    return linhas


# --------------------------------------------------------------------------
# Asserção
# --------------------------------------------------------------------------
def assercoes_p1(texto: str, entidades: list[dict]) -> dict[str, str]:
    """{maccrobat_id: estado do P1} usando o escopo de asserção do extrator de relações."""
    from clinical_kg.relations.core.text_layer import segment
    from clinical_kg.relations.extract import assertion_for, assertion_states

    saida = {}
    for sent in segment(texto):
        dentro = [e for e in entidades if e["start"] >= sent["start"] and e["end"] <= sent["end"]]
        if not dentro or not sent["tokens"]:
            continue
        tokens = [s[0] for s in sent["tokens"]]
        estados = assertion_states(tokens, sent["tokens"], sent["clauses"], texto)
        for e in dentro:
            saida[e["maccrobat_id"]] = assertion_for(estados, sent["tokens"], e["start"], e["end"])
    return saida


def avaliar_assercao(documentos) -> list[dict]:
    """Negação e hipótese contra POLARITY do MACCROBAT, nas entidades anotadas.

    Dois decisores: o escopo do P1 como foi entregue (`p1`) e o P1 com a propagação pela
    coordenação do P3 (`p3`, ver hibrido/assercao.py). As entidades são as do gold (as mesmas
    para os dois), então só o escopo é medido. "historical" e "family" do P1 contam como
    afirmadas: o MACCROBAT não as distingue.
    """
    from clinical_kg.hibrido.assercao import propagar_coordenacao

    from clinical_kg.learning.pipeline import bootstrap_pareado

    contagem = collections.Counter()
    matriz = collections.Counter()
    por_doc = {d: collections.defaultdict(lambda: [0, 0, 0]) for d in ("p1", "p3")}   # negated: vp, fp, fn
    for doc in documentos:
        entidades, _ = entidades_p1(doc)
        p1 = assercoes_p1(doc.texto, entidades)
        estados_p1 = [p1.get(e["maccrobat_id"], "affirmed") for e in entidades]
        estados = {"p1": estados_p1, "p3": propagar_coordenacao(doc.texto, entidades, estados_p1)}
        for decisor, lista in estados.items():
            contas = por_doc[decisor][doc.pmid]
            for e, previsto in zip(entidades, lista):
                real = e["assertion_status"]
                previsto = previsto if previsto in ("negated", "hedged") else "affirmed"
                if decisor == "p1":
                    matriz[(real, previsto)] += 1
                for classe in ("negated", "hedged"):
                    contagem[(decisor, classe, "vp")] += real == classe and previsto == classe
                    contagem[(decisor, classe, "fp")] += real != classe and previsto == classe
                    contagem[(decisor, classe, "fn")] += real == classe and previsto != classe
                contas[0] += real == "negated" and previsto == "negated"
                contas[1] += real != "negated" and previsto == "negated"
                contas[2] += real == "negated" and previsto != "negated"
    # Diferença de F1 da negação (p3 − p1), remuestreando DOCUMENTOS como o P2 remuestreia casos.
    boot = bootstrap_pareado({k: tuple(v) for k, v in por_doc["p3"].items()},
                             {k: tuple(v) for k, v in por_doc["p1"].items()})
    linhas = []
    for decisor in ("p1", "p3"):
        for classe in ("negated", "hedged"):
            vp, fp, fn = (contagem[(decisor, classe, k)] for k in ("vp", "fp", "fn"))
            p, r, f1 = _prf(vp, fp, fn)
            ic = (f"{boot['diferenca']:+.4f} [{boot['ic_inf']:+.4f}, {boot['ic_sup']:+.4f}]"
                  if decisor == "p3" and classe == "negated" else "")
            linhas.append({"decisor": decisor, "classe": classe, "anotadas": vp + fn, "vp": vp,
                           "fp": fp, "fn": fn, "precisao": p, "cobertura": r, "f1": f1,
                           "dif_f1_vs_p1_ic95": ic})
    return linhas, matriz


# --------------------------------------------------------------------------
# Medidas: valor de exame e dose de medicamento
# --------------------------------------------------------------------------
def avaliar_medidas(documentos, extrair=None) -> list[dict]:
    """Os vínculos valor -> exame e dose -> medicamento anotados, refeitos pelo P1.

    Só entram os valores com algum dígito: o extrator de medidas do P1 procura número + unidade,
    e o MACCROBAT também anota valores verbais ("normal", "elevated") que nenhum padrão numérico
    pegaria. Duas medidas, separando as duas etapas do P1:

        detecção: o P1 acha uma medida sobreposta ao valor anotado;
        vínculo : dessas, a medida foi ligada a uma entidade sobreposta ao exame/medicamento anotado.

    O vínculo é medido duas vezes: com as entidades do gold (isola a regra de ligação) e, se
    `extrair` for dado, com as entidades do gazetteer (o P1 de ponta a ponta).
    """
    from clinical_kg.extraction.measurements import find_measurements, link_measurement, split_sentences

    contagem = collections.Counter()
    for doc in documentos:
        entidades_gold, _ = entidades_p1(doc)
        entidades_gaz = extrair(doc.texto) if extrair else None
        inicios = [s for s, _ in split_sentences(doc.texto)]
        medidas = find_measurements(doc.texto)
        for r in relacoes_mapeadas(doc):
            papel = r["papel_p3"]
            if papel not in ("VALOR_DE_EXAME", "DOSE_DE_MEDICAMENTO"):
                continue
            if papel == "DOSE_DE_MEDICAMENTO" and r["arg1_tipo"] != "Dosage":
                continue          # Administration e Frequency não são número + unidade
            valor, alvo = doc.trechos[r["arg1"]], doc.trechos[r["arg2"]]
            contagem[(papel, "anotadas")] += 1
            if not any(c.isdigit() for c in valor.texto):
                continue
            contagem[(papel, "numericas")] += 1
            achada = next((m for m in medidas if _sobrepoe((m["start"], m["end"]), (valor.inicio, valor.fim))), None)
            if achada is None:
                continue
            contagem[(papel, "detectadas")] += 1
            for nome, ents in (("gold", entidades_gold), ("gazetteer", entidades_gaz)):
                if ents is None:
                    continue
                elo = link_measurement(achada, ents, inicios)
                if elo["linked_entity_start"] != "" and _sobrepoe(
                        (int(elo["linked_entity_start"]), int(elo["linked_entity_end"])), (alvo.inicio, alvo.fim)):
                    contagem[(papel, f"ligadas_{nome}")] += 1
    linhas = []
    for papel in ("VALOR_DE_EXAME", "DOSE_DE_MEDICAMENTO"):
        num, det = contagem[(papel, "numericas")], contagem[(papel, "detectadas")]
        linhas.append({
            "vinculo": papel, "anotadas": contagem[(papel, "anotadas")], "numericas": num,
            "detectadas": det, "deteccao": det / num if num else 0.0,
            "ligadas_entidades_gold": contagem[(papel, "ligadas_gold")],
            "acerto_vinculo_gold": contagem[(papel, "ligadas_gold")] / det if det else 0.0,
            "ligadas_entidades_gazetteer": contagem[(papel, "ligadas_gazetteer")] if extrair else "",
            "acerto_vinculo_gazetteer": (contagem[(papel, "ligadas_gazetteer")] / det if det else 0.0) if extrair else "",
            "ponta_a_ponta_gold": contagem[(papel, "ligadas_gold")] / num if num else 0.0,
        })
    return linhas


__all__ = ["CATEGORIA_P1", "GRUPOS", "avaliar_assercao", "avaliar_entidades", "avaliar_medidas",
           "casar_entidades", "gold_entidades"]
