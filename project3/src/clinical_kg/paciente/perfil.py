"""O perfil de um diagnóstico: o que os pacientes reais que o têm apresentam.

Um caso "tem" o diagnóstico D quando o modelo do P3 ligou o paciente a D por HAS_DIAGNOSIS
afirmado. O perfil de D agrega, sobre esses casos, os conceitos afirmados de cada categoria
(sintomas, achados, exames, tratamentos), com:

    casos     quais pacientes reais sustentam o fato (a proveniência que a aplicação mostra)
    ligado    se alguma aresta do grafo liga o fato ao próprio D (ex.: exame REVEALED_BY D,
              D TREATED_WITH tratamento), e não só a coocorrência no mesmo paciente
    revela    para exames: os achados do caso que esse exame revelou (REVEALED_BY)
    valores   para exames: os valores com unidade ligados a ele no texto (etapa 4 do P1)

A frequência de um fato é a fração dos casos do perfil em que ele aparece; ela decide o que é
típico (entra sempre na ficha) e o que é raro (entra conforme a dificuldade).
"""

import collections
import re
from dataclasses import dataclass, field

CATEGORIAS_FATO = ("Symptom", "Finding", "Exam", "Treatment")
# Um resultado de exame é concentração ou porcentagem ("lipase 1572 U/L"); tempo ("30 months")
# e massa (doses) ligados por proximidade a um exame quase sempre são de outra coisa.
UNIDADES_DE_RESULTADO = {"lab_concentration", "percentage"}
# Um conceito que aparece em mais desta fração de TODOS os casos do corpus ("patients",
# "treatment", "examination") não ensina nada sobre um diagnóstico específico.
LIMITE_GENERICO = 0.25


# Filtros de qualidade da APLICAÇÃO (o P1 não muda): o gazetteer acha termos do UMLS que são
# conceitos válidos mas não servem de fato clínico de um paciente virtual.
TUIS_DIAGNOSTICO = {"T047", "T191"}          # doença e neoplasia; T046 ("dilatation") é vago demais
DIAGNOSTICOS_VAGOS = {"tumor", "tumors", "cancer", "carcinoma", "neoplasm", "neoplasms", "metastasis",
                      "metastases", "disease", "diseases", "syndrome", "infection", "illness",
                      "lesion", "mass", "malignancy", "disorder"}
FATOS_VAGOS = {"lesion", "cd", "score", "within normal limits", "asymptomatic", "died", "single",
               "parameters", "remission", "intestinal", "complaint", "analgesia", "basis",
               "enhancement", "drug", "drugs", "medications", "medication", "perform", "expression",
               "elevation", "fat", "pt", "antibody", "antibodies", "therapy", "surgery", "illness",
               "imaging", "staining", "culture", "treatment", "procedure", "normal", "positive",
               "negative", "mild", "severe", "present", "unknown", "patient", "symptoms", "signs",
               "blood pressure", "mm3", "increase", "decrease", "change", "changes", "level", "levels",
               "discussion", "add", "pattern", "crisis", "examination findings", "mental status",
               "severely", "giant", "progressive disease", "agar", "supportive care", "program", "management",
               "near", "difficult", "close", "pulmonary", "rectal", "immune", "intraperitoneal", "nasal",
               "sharp", "bloody", "unable", "unstable", "alive", "foci", "patient died", "patient refused",
               "physical exam", "follow up", "several days", "under general anesthesia", "partial response",
               "complete response", "stable disease", "poor response", "performance status",
               "quality of life", "body mass index", "tumor size", "tumor mutation burden",
               "tumor proportion score", "pathological findings", "oral intake", "supplemental oxygen",
               "normal liver", "too little", "inr"}


def fato_util(nome: str) -> bool:
    """Descarta números ("5 mm", "day 7"), particípios soltos ("detected", "reduced"),
    antecedentes genéricos ("medical history") e a lista de termos vagos."""
    n = nome.lower().strip()
    if any(ch.isdigit() for ch in n) or "history" in n or n in FATOS_VAGOS or len(n) < 3 or "," in n:
        return False
    if any(w in n.split() for w in ("hospital", "clinic", "discharged", "seen", "follow-up")):
        return False
    return not (" " not in n and (n.endswith("ed") or n.endswith("ly") or n.endswith("ing")
                                  and n not in TERMINAM_EM_ING))


# Achados de uma palavra só em -ing que são clínicos de verdade (o resto, "decreasing", não é).
TERMINAM_EM_ING = {"bleeding", "swelling", "vomiting", "wheezing", "itching", "sweating",
                   "bruising", "choking", "fainting", "snoring", "stuttering", "drooling"}


def diagnostico_util(corpus, cui: str) -> bool:
    nome = corpus.nomes.get(cui, "").lower()
    return corpus.tuis.get(cui, "T047") in TUIS_DIAGNOSTICO and nome not in DIAGNOSTICOS_VAGOS \
        and fato_util(nome)


def genericos(corpus, limite: float = LIMITE_GENERICO) -> set[str]:
    """Conceitos presentes em mais de `limite` dos casos (frequência de documento)."""
    df = collections.Counter()
    for caso in corpus.casos.values():
        df.update({e["cui"] for e in caso.entidades})
    n = len(corpus.casos) or 1
    return {cui for cui, c in df.items() if c / n > limite}


@dataclass
class Fato:
    cui: str
    nome: str
    categoria: str
    casos: set = field(default_factory=set)
    ligado: bool = False
    revela: collections.Counter = field(default_factory=collections.Counter)
    valores: list = field(default_factory=list)

    def frequencia(self, n_casos: int) -> float:
        return len(self.casos) / n_casos if n_casos else 0.0


@dataclass
class Perfil:
    cui: str
    nome: str
    casos: list
    fatos: dict                      # cui -> Fato
    demografia: list = field(default_factory=list)      # (idade, sexo) de cada paciente real
    localizacoes: collections.Counter = field(default_factory=collections.Counter)

    def ordenados(self, categoria: str | None = None) -> list[Fato]:
        fatos = [f for f in self.fatos.values() if categoria in (None, f.categoria)]
        return sorted(fatos, key=lambda f: (-len(f.casos), -f.ligado, f.nome))


def diagnosticos_do_caso(caso) -> set[str]:
    return {r["tail_cui"] for r in caso.relacoes
            if r["relation"] == "HAS_DIAGNOSIS" and r["assertion_status"] == "affirmed" and r["tail_cui"]}


def achados_clinicos(corpus, minimo: int = 2) -> set[str]:
    """Achados (T033, o tipo "cesta" do UMLS) que o grafo usa como fato clínico.

    Um achado só entra num perfil se, em pelo menos `minimo` casos, o modelo do P3 o pôs na
    cauda de uma relação clínica afirmada (o paciente TEM, um exame REVELOU, está LOCALIZADO).
    "too little" ou "reduced" aparecem no texto, mas o grafo nunca os trata como achado.
    """
    casos = collections.defaultdict(set)
    for caso in corpus.casos.values():
        for r in caso.relacoes:
            if r["assertion_status"] == "affirmed" and r["relation"] in (
                    "HAS_FINDING", "HAS_SYMPTOM", "REVEALED_BY", "LOCATED_IN", "HAS_DIAGNOSIS"):
                casos[r["tail_cui"]].add(caso.case_id)
                if r["relation"] == "LOCATED_IN":
                    casos[r["head_cui"]].add(caso.case_id)
    return {cui for cui, cs in casos.items() if len(cs) >= minimo}


def montar_perfis(corpus, minimo_casos: int = 2) -> dict[str, Perfil]:
    """Perfis dos diagnósticos com pelo menos `minimo_casos` pacientes."""
    GENERICOS = genericos(corpus)
    CLINICOS = achados_clinicos(corpus)
    por_dx = collections.defaultdict(list)
    for caso in corpus.casos.values():
        for dx in diagnosticos_do_caso(caso):
            if dx not in GENERICOS and diagnostico_util(corpus, dx):
                por_dx[dx].append(caso)

    perfis = {}
    for dx, casos in por_dx.items():
        if len(casos) < minimo_casos:
            continue
        p = Perfil(dx, corpus.nomes.get(dx, dx), [c.case_id for c in casos], {})
        for caso in casos:
            p.demografia.append((caso.idade, caso.sexo))
            outros_dx = diagnosticos_do_caso(caso) - {dx}
            for e in caso.entidades:
                cui = e["cui"]
                cat = corpus.categorias.get(cui, e["entity_type"])
                nome = corpus.nomes.get(cui, e["surface_text"])
                if (cat not in CATEGORIAS_FATO or e["assertion_status"] != "affirmed"
                        or cui in GENERICOS or cui == dx or cui in outros_dx or not fato_util(nome)
                        or (cat == "Finding" and cui not in CLINICOS)):
                    continue
                fato = p.fatos.setdefault(cui, Fato(cui, nome, cat))
                fato.casos.add(caso.case_id)
            for r in caso.relacoes:
                if r["assertion_status"] != "affirmed":
                    continue
                cab, cau = r["head_cui"], r["tail_cui"]
                if dx in (cab, cau):
                    outro = cau if cab == dx else cab
                    if outro in p.fatos:
                        p.fatos[outro].ligado = True
                    if r["relation"] == "LOCATED_IN" and outro:
                        p.localizacoes[corpus.nomes.get(outro, outro)] += 1
                if r["relation"] == "REVEALED_BY" and cab in p.fatos and cau:
                    # O exame revelou `cau` neste caso (o próprio dx inclusive).
                    p.fatos[cab].revela[cau] += 1
            for m in caso.medicoes:
                cui = m.get("linked_entity_cui")
                if (cui in p.fatos and p.fatos[cui].categoria == "Exam" and m["surface_text"]
                        and m["unit_category"] in UNIDADES_DE_RESULTADO
                        and not re.search(r"\d\s*[-–]\s*\d", m["surface_text"])):   # faixa de referência
                    p.fatos[cui].valores.append((caso.case_id, m["surface_text"]))
        perfis[dx] = p
    return perfis


def assinatura(perfil: Perfil) -> dict[str, float]:
    """{cui do fato: frequência} — o vetor que compara perfis entre si (diagnóstico diferencial)."""
    n = len(perfil.casos)
    return {cui: f.frequencia(n) for cui, f in perfil.fatos.items()}
