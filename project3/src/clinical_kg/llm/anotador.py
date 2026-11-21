"""E3: o LLM como anotador de relações, disciplinado por regras.

Para cada par candidato o LLM recebe a sentença com as duas entidades marcadas, a definição dos
8 tipos e (opcionalmente) exemplos few-shot, e devolve um JSON:

    {"relacao": "CAUSED_BY", "evidencia": "pneumonia secondary to aspiration"}

Regras decidem se a resposta vale:
    * o tipo tem de ser compatível com as categorias das duas entidades (`tipos_permitidos`);
    * a evidência tem de ser um trecho literal da sentença (sem ela, a resposta é descartada).

A confiança não é a que o modelo declara: é a autoconsistência (Wang et al., 2023), a fração das
k amostras válidas que concordam no tipo vencedor.
"""

import collections
import json
import math
import re

from clinical_kg.learning.modelo import NENHUMA

TIPOS = {
    "HAS_SYMPTOM": "the patient has this symptom (subjective complaint: pain, nausea, fever)",
    "HAS_DIAGNOSIS": "the patient has this disease or diagnosis",
    "HAS_FINDING": "the patient has this objective finding (sign, imaging or lab abnormality)",
    "TREATED_WITH": "a condition (or the patient) is treated with this drug or procedure",
    "REVEALED_BY": "an exam or procedure revealed/showed/confirmed this finding or diagnosis",
    "LOCATED_IN": "a finding, disease, procedure or exam is located in / refers to this body part",
    "CAUSED_BY": "one condition is caused by, secondary to or due to the other",
    "COORDINATE_WITH": "the two entities are listed together as siblings (A, B and C)",
}
DO_PACIENTE = ("HAS_SYMPTOM", "HAS_DIAGNOSIS", "HAS_FINDING", "TREATED_WITH")
PACIENTE = "Paciente"

SISTEMA = ("You are a clinical information extraction annotator. You read one sentence from a "
           "medical case report and decide the relation between two marked entities. Answer only "
           "with a JSON object.")


def tipos_permitidos(tipo_cabeca: str, tipo_cauda: str) -> list[str]:
    """Tipos admissíveis para o par (sem contar NENHUMA, que sempre é admissível).

    A regra é permissiva de propósito: no gold, só 1 das 221 relações cai fora dela (as
    categorias do gazetteer erram, e uma regra estrita puniria o LLM pelos erros do P1).
    """
    if tipo_cabeca == PACIENTE:
        return list(DO_PACIENTE)
    tipos = ["TREATED_WITH", "REVEALED_BY", "CAUSED_BY", "COORDINATE_WITH"]
    if "BodyPart" in (tipo_cabeca, tipo_cauda):
        tipos.insert(2, "LOCATED_IN")
    return tipos


def tipos_do_exemplo(e) -> tuple[str, str]:
    def achar(prefixo):
        return next(k.split(":", 1)[1] for k in e.estruturais if k.startswith(prefixo))
    return achar("tipo_cabeca:"), achar("tipo_cauda:")


def limites_da_sentenca(texto: str, inicio: int, fim: int) -> tuple[int, int]:
    """A sentença do P1 que contém o trecho [inicio, fim)."""
    from clinical_kg.relations.core.text_layer import sentence_spans
    for s, t in sentence_spans(texto):
        if s <= inicio and fim <= t:
            return s, t
    return inicio, fim


def sentenca_marcada(texto: str, chave: tuple) -> tuple[str, str]:
    """(sentença com [E1]...[/E1] e [E2]...[/E2], sentença limpa). Paciente: só [E2]."""
    _, ci, cf, ti, tf = chave
    ini = min(ti, ci) if ci >= 0 else ti
    fim = max(tf, cf) if ci >= 0 else tf
    s, t = limites_da_sentenca(texto, ini, fim)
    marcas = [(ti, "[E2]"), (tf, "[/E2]")]
    if ci >= 0:
        marcas += [(ci, "[E1]"), (cf, "[/E1]")]
    partes, cursor = [], s
    for pos, marca in sorted(marcas, key=lambda m: (m[0], m[1].startswith("[/") is False)):
        partes += [texto[cursor:pos], marca]
        cursor = pos
    partes.append(texto[cursor:t])
    return "".join(partes).strip(), texto[s:t].strip()


def descrever_par(e, texto: str) -> dict:
    marcada, limpa = sentenca_marcada(texto, e.chave)
    tipo_cab, tipo_cau = tipos_do_exemplo(e)
    return {"marcada": marcada, "limpa": limpa, "tipo_cabeca": tipo_cab, "tipo_cauda": tipo_cau,
            "cabeca": e.cabeca if tipo_cab != PACIENTE else "the patient", "cauda": e.cauda,
            "permitidos": tipos_permitidos(tipo_cab, tipo_cau)}


def montar_prompt(par: dict, exemplos_fewshot=()) -> str:
    """`exemplos_fewshot`: [(par descrito, relação correta, evidência)], já do treino da rodada."""
    linhas = ["Relation types:"]
    linhas += [f"- {t}: {d}" for t, d in TIPOS.items() if t in par["permitidos"]]
    linhas.append(f"- {NENHUMA}: none of the above holds between these two entities in this sentence")
    for i, (ex, rel, evid) in enumerate(exemplos_fewshot, 1):
        linhas += ["", f"Example {i}:", f"Sentence: {ex['marcada']}",
                   f"E1 = {ex['cabeca']} ({ex['tipo_cabeca']}); E2 = {ex['cauda']} ({ex['tipo_cauda']})",
                   "Answer: " + json.dumps({"relacao": rel, "evidencia": evid}, ensure_ascii=False)]
    e1 = "the patient (E1 is implicit)" if par["tipo_cabeca"] == PACIENTE else \
         f"{par['cabeca']} ({par['tipo_cabeca']})"
    linhas += ["", "Now this one:", f"Sentence: {par['marcada']}",
               f"E1 = {e1}; E2 = {par['cauda']} ({par['tipo_cauda']})",
               f'Answer with JSON: {{"relacao": one of {par["permitidos"] + [NENHUMA]}, '
               '"evidencia": the exact words of the sentence that support it ("" for NENHUMA)}']
    return "\n".join(linhas)


_JSON = re.compile(r"\{.*?\}", re.DOTALL)


def interpretar(resposta: str, par: dict) -> str | None:
    """O tipo da resposta, ou None se ela for inválida pelas regras."""
    achado = _JSON.search(resposta or "")
    if not achado:
        return None
    try:
        dados = json.loads(achado.group())
    except ValueError:
        return None
    relacao = str(dados.get("relacao", "")).strip().upper()
    evidencia = str(dados.get("evidencia", "")).strip()
    if relacao == NENHUMA:
        return NENHUMA
    if relacao not in par["permitidos"]:
        return None
    normalizar = lambda s: " ".join(s.lower().split())
    if not evidencia or normalizar(evidencia) not in normalizar(par["limpa"]):
        return None
    return relacao


class Anotador:
    """k amostras por par; voto majoritário e a fração de votos como confiança.

    Com k = 1 usa temperatura 0 (resposta determinística); com k > 1, `temperatura`.
    """

    def __init__(self, provedor, k: int = 1, temperatura: float = 0.7):
        self.provedor, self.k = provedor, k
        self.temperatura = 0.0 if k == 1 else temperatura

    def anotar(self, par: dict, exemplos_fewshot=()) -> dict:
        prompt = montar_prompt(par, exemplos_fewshot)
        votos = collections.Counter()
        for i in range(self.k):
            relacao = interpretar(self.provedor.completar(prompt, SISTEMA, self.temperatura, i), par)
            if relacao is not None:
                votos[relacao] += 1
        validas = sum(votos.values())
        if not validas:
            return {"relacao": NENHUMA, "confianca": 0.0, "votos": {}, "validas": 0, "amostras": self.k}
        relacao, n = max(sorted(votos.items()), key=lambda kv: kv[1])
        return {"relacao": relacao, "confianca": n / validas, "votos": dict(votos),
                "validas": validas, "amostras": self.k}


def cosseno(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


def escolher_fewshot(alvo, candidatos, vetores: dict, n: int = 4, por_classe: int = 2):
    """Os `n` exemplos de treino mais parecidos com o alvo (Liu et al., 2022).

    No máximo `por_classe` de cada tipo, para que o prompt não vire uma lista de NENHUMA. Quem
    chama passa só exemplos dos casos de treino da rodada: o caso de teste nunca entra.
    """
    ordenados = sorted(candidatos, key=lambda c: -cosseno(vetores[alvo.chave], vetores[c.chave]))
    escolhidos, por_tipo = [], collections.Counter()
    for c in ordenados:
        if c.caso == alvo.caso or por_tipo[c.relacao] >= por_classe:
            continue
        escolhidos.append(c)
        por_tipo[c.relacao] += 1
        if len(escolhidos) == n:
            break
    return escolhidos
