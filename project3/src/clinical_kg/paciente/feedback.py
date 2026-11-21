"""O feedback: a hipótese do estudante contra a ficha, e o diagnóstico diferencial do grafo.

O diferencial é um Bayes ingênuo sobre os perfis: cada diagnóstico D pontua os fatos que o
estudante levantou,

    log P(D) + Σ_presentes log P(f | D) + Σ_negados log (1 − P(f | D)),

com P(f | D) = (casos de D com f + α) / (casos de D + 2α). É o que o grafo "pensaria" com as
mesmas informações que o estudante juntou, e cada termo aponta para os casos reais por trás.

"O que deixou passar" são os fatos da ficha que o estudante não levantou, ordenados pelo quanto
separam o diagnóstico certo dos concorrentes (razão de verossimilhança).
"""

import math

ALFA = 0.5


def _p(perfil, cui: str) -> float:
    n = len(perfil.casos)
    k = len(perfil.fatos[cui].casos) if cui in perfil.fatos else 0
    return (k + ALFA) / (n + 2 * ALFA)


def diferencial(presentes: set[str], negados: set[str], perfis: dict, n: int = 6) -> list[dict]:
    total = sum(len(p.casos) for p in perfis.values()) or 1
    pontos = []
    for cui, p in perfis.items():
        s = math.log(len(p.casos) / total)
        s += sum(math.log(_p(p, f)) for f in presentes)
        s += sum(math.log(1 - _p(p, f)) for f in negados)
        pontos.append((cui, s))
    pontos.sort(key=lambda x: -x[1])
    maior = pontos[0][1] if pontos else 0.0
    soma = sum(math.exp(s - maior) for _, s in pontos) or 1.0
    return [{"cui": cui, "nome": perfis[cui].nome, "probabilidade": math.exp(s - maior) / soma,
             "casos": len(perfis[cui].casos),
             "apoio": sorted(f for f in presentes if f in perfis[cui].fatos)}
            for cui, s in pontos[:n]]


def avaliar(ficha, atendimento, hipotese_cui: str, perfis: dict) -> dict:
    presentes, negados = atendimento.presentes_revelados(), atendimento.negados_revelados()
    ranking = diferencial(presentes, negados, perfis, n=len(perfis))
    posicao = next((i + 1 for i, d in enumerate(ranking) if d["cui"] == ficha.dx_cui), None)
    certo = perfis.get(ficha.dx_cui)
    outros = [d["cui"] for d in ranking[:6] if d["cui"] != ficha.dx_cui]

    def razao(cui):
        if certo is None or not outros:
            return 0.0
        media = sum(_p(perfis[o], cui) for o in outros) / len(outros)
        return math.log(_p(certo, cui) / media)

    perdidos = [i for i in ficha.itens if i.categoria in ("Symptom", "Finding")
                and i.cui not in atendimento.revelados]
    perdidos.sort(key=lambda i: -abs(razao(i.cui)))
    return {
        "correto": hipotese_cui == ficha.dx_cui,
        "diagnostico": ficha.dx_nome,
        "hipotese": perfis[hipotese_cui].nome if hipotese_cui in perfis else hipotese_cui,
        "diferencial": ranking[:6],
        "posicao_do_certo": posicao,
        "perdidos": [{"nome": i.nome, "presente": i.presente, "razao": round(razao(i.cui), 2),
                      "casos": i.casos} for i in perdidos],
        "levantados": len(presentes) + len(negados),
    }


def subgrafo(ficha, atendimento, perfis: dict, diferencial_top: list[dict]) -> dict:
    """Grafo para o Cytoscape do P2: o diagnóstico, os fatos da ficha e os concorrentes.

    A cor do nó (campo `entity_type`) diz o estado: levantado pelo estudante, não levantado,
    negado; os concorrentes ligam-se aos fatos levantados que também aparecem no seu perfil.
    """
    nos, arestas = [], []

    def no(id_, rotulo, estado, contagem=1):
        nos.append({"id": id_, "kind": "Concept", "label": rotulo, "entity_type": estado,
                    "count": contagem})

    def aresta(origem, destino, relacao, rotulo):
        arestas.append({"id": f"e:{origem}>{destino}:{relacao}", "source": origem, "target": destino,
                        "relation": relacao, "link_method": rotulo})

    dx = f"dx:{ficha.dx_cui}"
    no(dx, f"{ficha.dx_nome} (diagnóstico)", "Diagnóstico", len(ficha.casos_do_perfil))
    for i in ficha.itens:
        if i.categoria == "Treatment":
            continue
        estado = ("Negado" if not i.presente else
                  "Levantado" if i.cui in atendimento.revelados else "Não levantado")
        no(f"f:{i.cui}", f"{i.nome} ({i.categoria})", estado, max(1, len(i.casos)))
        rel = "NEGA" if not i.presente else "APRESENTA"
        aresta(dx, f"f:{i.cui}", rel, f"{len(i.casos)} casos reais" if i.presente else "distrator")
    levantados = atendimento.presentes_revelados()
    for d in diferencial_top:
        if d["cui"] == ficha.dx_cui:
            continue
        nid = f"dx:{d['cui']}"
        no(nid, f"{d['nome']} ({d['probabilidade']:.0%})", "Concorrente", d["casos"])
        for f in d["apoio"]:
            if f in levantados:
                aresta(nid, f"f:{f}", "COMPARTILHA", "também no perfil do concorrente")
    return {"nodes": nos, "edges": arestas}


PALETA_FEEDBACK = {
    "Diagnóstico": ("#2a78d6",), "Levantado": ("#16a34a",), "Não levantado": ("#eb6834",),
    "Negado": ("#dc2626",), "Concorrente": ("#64748b",),
}
