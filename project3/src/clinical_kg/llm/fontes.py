"""As variantes do P3 que usam o LLM, todas a partir de votos já calculados (e em cache).

    votos = {chave do par: {"relacao", "confianca", "votos", "validas", "amostras"}}

    prever_llm         P3-A/P3-B: o LLM decide sozinho (o tipo mais votado)
    FonteLLM           P3-C: voto e confiança do LLM como atributos da regressão do P2
    prever_cascata     P3-E: o LLM só arbitra os pares em que P1 e P2 discordam
    rotular_com_llm    P3-F: supervisão fraca, os candidatos dos 50 casos rotulados pelo LLM
"""

from clinical_kg.learning.modelo import NENHUMA, Exemplo

from .anotador import TIPOS


def prever_llm(exemplos, votos: dict, confianca_minima: float = 0.0):
    """O tipo mais votado; abaixo de `confianca_minima`, NENHUMA."""
    saida = []
    for e in exemplos:
        v = votos[e.chave]
        rel = v["relacao"] if v["confianca"] >= confianca_minima else NENHUMA
        saida.append((e, rel))
    return saida


class FonteLLM:
    """Fonte de atributos para `hibrido.transferencia.ModeloHibrido` (P3-C).

    Os votos do LLM zero-shot não dependem do treino (o LLM não viu o gold), então o atributo é o
    mesmo no treino e no teste e não há o que fazer fora de amostra. Atributos:
        llm_relacao     confiança do LLM de que HÁ relação (fração de votos não-NENHUMA)
        llm:<TIPO>      fração de votos naquele tipo
    """

    nome = "llm"

    def __init__(self, votos: dict):
        self.votos = votos

    def ajustar(self, treino):
        return self

    def _atributos(self, e) -> dict:
        v = self.votos[e.chave]
        validas = v["validas"] or 1
        x = {"llm_relacao": sum(n for t, n in v["votos"].items() if t != NENHUMA) / validas}
        for t, n in v["votos"].items():
            if t in TIPOS:
                x[f"llm:{t}"] = n / validas
        return x

    atributo_treino = _atributos
    atributo_teste = _atributos


def prever_cascata(p1, p2, votos: dict):
    """P3-E. `p1`, `p2`: [(exemplo, relação)] nos mesmos exemplos e na mesma ordem.

    Onde os dois concordam, fica a decisão comum (nenhuma chamada ao LLM). Onde discordam, vence
    o que coincide com o LLM; se o LLM não concorda com nenhum, fica o P2 (o melhor dos dois).
    Devolve também quantos pares precisaram do LLM, o custo da variante.
    """
    saida, consultas = [], 0
    for (e, r1), (e2, r2) in zip(p1, p2):
        assert e.chave == e2.chave
        if r1 == r2:
            saida.append((e, r1))
            continue
        consultas += 1
        llm = votos[e.chave]["relacao"]
        saida.append((e, r1 if llm == r1 else r2))
    return saida, consultas


def rotular_com_llm(candidatos: list[Exemplo], votos: dict) -> list[Exemplo]:
    """P3-F: os mesmos candidatos da supervisão fraca do P2, com o rótulo do LLM no lugar do das regras."""
    saida = []
    for e in candidatos:
        rel = votos[e.chave]["relacao"]
        saida.append(Exemplo(caso=e.caso, chave=e.chave, estruturais=e.estruturais, entre=e.entre,
                             eh_relacao=int(rel != NENHUMA), relacao=rel, cabeca=e.cabeca,
                             cauda=e.cauda, extras=e.extras))
    return saida
