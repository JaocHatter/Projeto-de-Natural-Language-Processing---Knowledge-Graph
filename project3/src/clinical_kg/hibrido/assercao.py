"""Asserção do P3: o escopo de negação do P1 + propagação pela coordenação.

A ida e volta dos pacientes virtuais mostrou um erro do P1: em "She denied tenderness and
dysuria", só a primeira entidade sai negada. O escopo do P1 reinicia a cada oração, e o
segmentador corta a oração na conjunção, então a lista coordenada perde o gatilho.

A correção é uma regra pequena, aplicada DEPOIS do P1 (o P1 não muda): uma entidade cujo texto
até a entidade anterior é só coordenação ("," "and" "or" "nor") herda o estado dela, se ela for
negada ou hipotética. Mede-se nos dois golds que têm negação: o MACCROBAT (humano) e as fichas
dos pacientes virtuais.
"""

import re

COORDENACAO = re.compile(r"^(?:\s|,|;|/|\band\b|\bor\b|\bnor\b)+$", re.IGNORECASE)
HERDAVEIS = ("negated", "hedged")


def propagar_coordenacao(texto: str, entidades: list[dict], estados: list[str]) -> list[str]:
    """`entidades` em ordem de posição; devolve os estados com a negação propagada."""
    saida = list(estados)
    ordem = sorted(range(len(entidades)), key=lambda i: entidades[i]["start"])
    for anterior, atual in zip(ordem, ordem[1:]):
        if saida[anterior] in HERDAVEIS and saida[atual] == "affirmed":
            entre = texto[entidades[anterior]["end"]:entidades[atual]["start"]]
            if entre and COORDENACAO.match(entre):
                saida[atual] = saida[anterior]
    return saida


def assercao_p3(texto: str, entidades: list[dict]) -> list[str]:
    from clinical_kg.paciente.corpus import assercao_das_entidades
    return propagar_coordenacao(texto, entidades, assercao_das_entidades(texto, entidades))
