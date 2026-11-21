"""Pares candidatos sobre o MACCROBAT, descritos exatamente como os do gold do MultiCaRe.

O gerador de candidatos é o do P1 (`analyze_case`) e os atributos são os do P2
(`atributos_estruturais`). A diferença está nas entidades: em vez do gazetteer UMLS, entram as
entidades anotadas por humanos no MACCROBAT. Assim a avaliação externa mede a decisão de relação
isolada dos erros de reconhecimento de entidades.

O rótulo de cada par vem das relações do MACCROBAT com equivalente no P1 (CAUSED_BY e
LOCATED_IN); os demais pares ficam NENHUMA para esses dois tipos. Pares com o paciente na cabeça
ficam de fora: o MACCROBAT não anota relações com o paciente.
"""

import collections

from clinical_kg.learning.modelo import NENHUMA, Exemplo, atributos_estruturais
from clinical_kg.maccrobat.mapeamento import TRANSFERIVEIS, entidades_p1, rotulos_por_par
from clinical_kg.relations.core import features as regras_p1
from clinical_kg.relations.extract import DEFAULT_FREQ, analyze_case, load_frequencies


def exemplos_maccrobat(documentos) -> tuple[list[Exemplo], collections.Counter]:
    """Os candidatos de todos os documentos e quantas relações anotadas viraram candidato."""
    freq = load_frequencies(DEFAULT_FREQ)
    exemplos, cobertura = [], collections.Counter()
    for doc in documentos:
        entidades, _ = entidades_p1(doc)
        ids = {e["maccrobat_id"] for e in entidades}
        alvo = {par: rel for par, rel in rotulos_por_par(doc).items() if par <= ids}
        vistas, chaves = set(), set()
        for cand in analyze_case(doc.texto, entidades, freq, threshold=float("-inf")):
            cab, cau = cand["head"], cand["tail"]
            if cab is None:
                continue
            chave = (doc.pmid, cab["start"], cab["end"], cau["start"], cau["end"])
            if chave in chaves:
                continue
            chaves.add(chave)
            par = frozenset((cab["maccrobat_id"], cau["maccrobat_id"]))
            relacao = alvo.get(par, NENHUMA)
            if relacao != NENHUMA:
                vistas.add(par)
            estruturais, entre = atributos_estruturais(doc.texto, cand)
            exemplos.append(Exemplo(
                caso=f"MACCROBAT:{doc.pmid}", chave=chave, estruturais=estruturais, entre=entre,
                eh_relacao=int(relacao != NENHUMA), relacao=relacao,
                cabeca=cab["surface_text"], cauda=cau["surface_text"],
                extras={"p1_score": cand["score"], "p1_relacao": cand["relation"]}))
        for par, rel in alvo.items():
            cobertura[f"{rel}:anotadas"] += 1
            cobertura[f"{rel}:candidatas"] += par in vistas
    return exemplos, cobertura


def decisao_p1(exemplo: Exemplo) -> str:
    """O que as regras do P1, como foram entregues, decidem sobre o par."""
    if exemplo.extras["p1_score"] >= regras_p1.WEIGHTS["decide.threshold"]:
        return exemplo.extras["p1_relacao"]
    return NENHUMA


def medir_por_tipo(previsoes, cobertura, tipos=TRANSFERIVEIS) -> list[dict]:
    """Precisão, cobertura e F1 de cada tipo transferível no MACCROBAT.

    `cobertura_total` divide os acertos por TODAS as relações anotadas desse tipo (inclusive as
    que o gerador do P1 nunca propôs como candidato); `teto_do_gerador` é a fração delas que
    chegou a ser candidata, o máximo que qualquer decisor sobre esses pares poderia recuperar.
    """
    linhas = []
    for tipo in tipos:
        vp = sum(1 for e, p in previsoes if p == tipo and e.relacao == tipo)
        fp = sum(1 for e, p in previsoes if p == tipo and e.relacao != tipo)
        fn = sum(1 for e, p in previsoes if p != tipo and e.relacao == tipo)
        precisao = vp / (vp + fp) if vp + fp else 0.0
        cob = vp / (vp + fn) if vp + fn else 0.0
        f1 = 2 * precisao * cob / (precisao + cob) if precisao + cob else 0.0
        anotadas = cobertura[f"{tipo}:anotadas"]
        linhas.append({"tipo": tipo, "vp": vp, "fp": fp, "fn": fn, "precisao": precisao, "cobertura": cob,
                       "f1": f1, "anotadas": anotadas,
                       "teto_do_gerador": cobertura[f"{tipo}:candidatas"] / anotadas if anotadas else 0.0,
                       "cobertura_total": vp / anotadas if anotadas else 0.0})
    return linhas
