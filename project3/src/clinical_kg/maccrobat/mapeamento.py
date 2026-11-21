"""Do esquema do MACCROBAT para o do Projeto 1.

Entidades: só as 6 categorias do P1 entram, e sem sobreposição, porque o gerador de candidatos do
P1 supõe trechos disjuntos (o gazetteer faz casamento mais longo). Quando dois trechos se
sobrepõem, fica o mais longo, como no P1.

Relações: só duas têm equivalente direto e viram rótulo de treino (`TRANSFERIVEIS`):

    CAUSE                                        -> CAUSED_BY
    MODIFY com Biological_structure modificando   -> LOCATED_IN

As demais são exportadas com o seu papel no P3 (valor de exame, dose, ordem temporal,
correferência), mas não rotulam pares: o MACCROBAT não anota TREATED_WITH, REVEALED_BY etc., então
a ausência de uma dessas relações lá não quer dizer que ela não exista.
"""

import collections

# O MACCROBAT não separa sinal de sintoma. Finding é a categoria mais frequente do P1 para esses
# termos (306 conceitos contra 66 de Symptom), e só os atributos estruturais dependem dessa escolha.
CATEGORIA_P1 = {
    "Disease_disorder": "Diagnosis",
    "Sign_symptom": "Finding",
    "Diagnostic_procedure": "Exam",
    "Medication": "Treatment",
    "Therapeutic_procedure": "Treatment",
    "Biological_structure": "BodyPart",
}
ASSERCAO_P1 = {"NEG": "negated", "UNCERTAIN": "hedged", "MAYBE_POS": "hedged", "MAYBE_NEG": "hedged"}
TRANSFERIVEIS = ("CAUSED_BY", "LOCATED_IN")
TEMPORAIS = ("BEFORE", "AFTER")


def papel_no_p1(tipo: str, tipo_arg1: str, tipo_arg2: str) -> str:
    """O que uma relação do MACCROBAT significa no esquema do P3 ("" se nada)."""
    if tipo == "CAUSE":
        return "CAUSED_BY"
    if tipo == "MODIFY" and tipo_arg1 == "Biological_structure":
        return "LOCATED_IN"
    if tipo == "MODIFY" and tipo_arg1 == "Lab_value" and tipo_arg2 == "Diagnostic_procedure":
        return "VALOR_DE_EXAME"
    if tipo == "MODIFY" and tipo_arg1 in ("Dosage", "Administration", "Frequency") and tipo_arg2 == "Medication":
        return "DOSE_DE_MEDICAMENTO"
    if tipo in TEMPORAIS:
        return "TEMPORAL"
    if tipo == "IDENTICAL":
        return "CORREFERENCIA"
    return ""


def assercao(doc, id_trecho: str) -> str:
    return ASSERCAO_P1.get(doc.atributos.get(id_trecho, {}).get("POLARITY", ""), "affirmed")


def entidades_p1(doc) -> tuple[list[dict], collections.Counter]:
    """Trechos das 6 categorias como entidades do P1, disjuntos e em ordem de posição.

    Devolve também quantos trechos foram descartados e por quê (descontínuos ou sobrepostos).
    """
    descartes = collections.Counter()
    candidatos = []
    for t in doc.trechos.values():
        if t.tipo not in CATEGORIA_P1:
            continue
        if t.descontinuo:
            descartes["descontinuo"] += 1
            continue
        candidatos.append(t)

    candidatos.sort(key=lambda t: (-(t.fim - t.inicio), t.inicio))
    ocupados, escolhidos = [], []
    for t in candidatos:
        if any(t.inicio < fim and inicio < t.fim for inicio, fim in ocupados):
            descartes["sobreposto"] += 1
            continue
        ocupados.append((t.inicio, t.fim))
        escolhidos.append(t)
    escolhidos.sort(key=lambda t: (t.inicio, t.fim))

    entidades = [{
        "case_id": doc.pmid, "start": t.inicio, "end": t.fim, "surface_text": t.texto,
        "term_norm": t.texto.lower(), "entity_type": CATEGORIA_P1[t.tipo], "cui": "", "tui": "",
        "maccrobat_id": t.id, "maccrobat_tipo": t.tipo, "assertion_status": assercao(doc, t.id),
    } for t in escolhidos]
    return entidades, descartes


def relacoes_mapeadas(doc) -> list[dict]:
    """Todas as relações do documento, cada uma com o seu papel no P3."""
    saida = []
    for tipo, a1, a2 in doc.relacoes:
        t1, t2 = doc.trechos.get(a1), doc.trechos.get(a2)
        if t1 is None or t2 is None:
            continue
        saida.append({"pmid": doc.pmid, "tipo_maccrobat": tipo, "papel_p3": papel_no_p1(tipo, t1.tipo, t2.tipo),
                      "arg1": a1, "arg1_tipo": t1.tipo, "arg1_texto": t1.texto,
                      "arg2": a2, "arg2_tipo": t2.tipo, "arg2_texto": t2.texto})
    return saida


def rotulos_por_par(doc) -> dict[frozenset, str]:
    """{par não ordenado de ids de trechos: CAUSED_BY ou LOCATED_IN}, para rotular candidatos."""
    rotulos = {}
    for r in relacoes_mapeadas(doc):
        if r["papel_p3"] in TRANSFERIVEIS:
            rotulos[frozenset((r["arg1"], r["arg2"]))] = r["papel_p3"]
    return rotulos
