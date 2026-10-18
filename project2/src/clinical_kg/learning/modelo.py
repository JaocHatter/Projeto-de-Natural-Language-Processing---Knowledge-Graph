#!/usr/bin/env python3
"""Atributos por par candidato e regressão logística escrita à mão.

O gerador de candidatos é o do Projeto 1 (`relations.extract.analyze_case` sem
limiar) e não é alterado: os dois sistemas veem exatamente os mesmos pares, então
qualquer diferença de desempenho vem de quem decide, não de quais pares existem.

    Exemplo               um par candidato com seus rótulos do gold
    carregar_exemplos     candidatos do P1 casados com data/interim/gold_relations.csv
    atributos             dicionário de números que descreve o par
    RegressaoLogistica    SGD + L2 + peso para a classe rara
    UmContraTodos         estende o binário às 8 relações (mais "NENHUMA")

Atributos que o Projeto 1 já calculou sobre o mesmo par (gatilho encontrado,
categoria, `score`, `rule_path`) NÃO são usados: com eles o modelo apenas imitaria
as regras, e a pergunta do projeto é se ele aprende o padrão sozinho.
"""

import collections
import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from clinical_kg.paths import DEFAULT_CASES, DEFAULT_ENTITIES, DEFAULT_GOLD_RELATIONS
from clinical_kg.relations.extract import (DEFAULT_FREQ, analyze_case, load_entities,
                                           load_frequencies)

from .ngramas import Associacoes, ModeloLinguagem, tokenizar

NENHUMA = "NENHUMA"   # rótulo dos pares sem relação

# Escalas para que os atributos numéricos fiquem em ~[0, 1]; o SGD é sensível a isso.
ESCALA_PMI = 12.0
ESCALA_LLR = 300.0
DISTANCIA_MAX = 20


@dataclass
class Exemplo:
    """Um par candidato, com o que o modelo precisa para descrevê-lo e avaliá-lo."""

    caso: str
    chave: tuple                      # (caso, cab_ini, cab_fim, cau_ini, cau_fim)
    estruturais: dict                 # atributos que não dependem do corpus
    entre: list                       # palavras entre as duas entidades
    eh_relacao: int = 0               # gold: há relação?
    relacao: str = NENHUMA            # gold: qual (NENHUMA se não há)
    cabeca: str = ""
    cauda: str = ""
    extras: dict = field(default_factory=dict)


def _faixa(distancia: int) -> str:
    return "0" if distancia == 0 else "1_3" if distancia <= 3 else "4_10" if distancia <= 10 else "mais_10"


def atributos_estruturais(texto: str, cand: dict) -> tuple[dict, list]:
    """Atributos sem léxico (tipos, distância, pontuação) e as palavras entre as entidades."""
    cab, cau = cand["head"], cand["tail"]
    inicio = cab["end"] if cab else cand["sent_start"]
    entre_texto = texto[inicio:cau["start"]]
    entre = tokenizar(entre_texto)

    tipo_cab = cab["entity_type"] if cab else "Paciente"
    f = {
        f"par:{tipo_cab}>{cau['entity_type']}": 1.0,
        f"tipo_cabeca:{tipo_cab}": 1.0,
        f"tipo_cauda:{cau['entity_type']}": 1.0,
        f"distancia:{_faixa(cand['token_gap'])}": 1.0,
        "n_palavras_entre": min(len(entre), DISTANCIA_MAX) / DISTANCIA_MAX,
        "mesma_clausula": 1.0 if cand["same_clause"] else 0.0,
        "paciente_cabeca": 1.0 if cab is None else 0.0,
        "tem_virgula": 1.0 if "," in entre_texto else 0.0,
    }
    return f, entre


def atributos_associacao(entre: list, assoc) -> dict:
    """PMI e LLR dos bigramas do trecho entre as entidades (dependem do corpus)."""
    if len(entre) < 2:
        return {}
    pares = list(zip(entre, entre[1:]))
    pmis = [assoc.pmi(a, b) for a, b in pares]
    llrs = [assoc.llr(a, b) for a, b in pares]
    return {
        "pmi_max": max(pmis) / ESCALA_PMI,
        "pmi_medio": (sum(pmis) / len(pmis)) / ESCALA_PMI,
        "llr_max": min(max(llrs) / ESCALA_LLR, 1.0),
    }


def atributos(exemplo: Exemplo, assoc=None) -> dict:
    """Dicionário final do par. Sem `assoc`, só os atributos estruturais."""
    f = dict(exemplo.estruturais)
    if assoc is not None:
        f.update(atributos_associacao(exemplo.entre, assoc))
    return f


def carregar_exemplos(casos: Path = DEFAULT_CASES, entidades: Path = DEFAULT_ENTITIES,
                      gold: Path = DEFAULT_GOLD_RELATIONS) -> list[Exemplo]:
    """Os pares julgados à mão, descritos com os candidatos do Projeto 1."""
    # Sem newline="": os deslocamentos do gold e das entidades indexam o texto com
    # quebras de linha LF; ler com CRLF intactos deslocaria todos os offsets.
    with open(casos, encoding="utf-8") as f:
        textos = {r["case_id"]: r["case_text"] for r in csv.DictReader(f)}
    ents = load_entities(entidades)
    freq = load_frequencies(DEFAULT_FREQ)
    with open(gold, newline="", encoding="utf-8") as f:
        julgados = list(csv.DictReader(f))

    candidatos = {}
    for caso in {j["case_id"] for j in julgados}:
        for c in analyze_case(textos[caso], ents[caso], freq, threshold=float("-inf")):
            cab = c["head"]
            chave = (caso, cab["start"] if cab else -1, cab["end"] if cab else -1,
                     c["tail"]["start"], c["tail"]["end"])
            candidatos[chave] = c

    exemplos = []
    for j in julgados:
        chave = (j["case_id"], int(j["head_start"]), int(j["head_end"]),
                 int(j["tail_start"]), int(j["tail_end"]))
        cand = candidatos.get(chave)
        if cand is None:
            raise KeyError(f"par do gold sem candidato correspondente: {chave}")
        estruturais, entre = atributos_estruturais(textos[j["case_id"]], cand)
        eh = int(j["is_relation"])
        exemplos.append(Exemplo(
            caso=j["case_id"], chave=chave, estruturais=estruturais, entre=entre,
            eh_relacao=eh, relacao=(j["gold_relation"] if eh else NENHUMA),
            cabeca=j["head_text"], cauda=j["tail_text"],
            # O que o P1 decidiu sobre o mesmo par: só para a linha de base,
            # nunca como atributo (ver `atributos`).
            extras={"p1_score": cand["score"], "p1_relacao": cand["relation"]}))
    return exemplos


# --------------------------------------------------------------------------
# Regressão logística
# --------------------------------------------------------------------------
def _sigmoide(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class RegressaoLogistica:
    """Regressão logística binária por descida de gradiente estocástica.

        P(y=1 | x) = sigmoide(b + sum_i w_i * x_i)

    A cada exemplo, o peso de cada atributo ativo anda contra o gradiente da perda
    logística, com penalidade L2 (`l2`) para não deixar pesos crescerem sobre
    poucos exemplos. A taxa de aprendizado decai como taxa / (1 + época).

    `peso_classe_rara`: com 4 exemplos positivos contra 409 negativos, responder
    sempre "não" acerta 99% e nunca dispara. Multiplicar a atualização dos
    positivos por (negativos / positivos), limitado por `peso_max`, tira esse
    atalho sem deixar um único positivo dominar o treino.
    """

    def __init__(self, epocas: int = 40, taxa: float = 0.5, l2: float = 1e-4,
                 peso_classe_rara: bool = True, peso_max: float = 20.0, semente: int = 0):
        self.epocas, self.taxa, self.l2 = epocas, taxa, l2
        self.peso_classe_rara, self.peso_max, self.semente = peso_classe_rara, peso_max, semente
        self.pesos: dict[str, float] = {}
        self.vies = 0.0

    def _z(self, x: dict) -> float:
        return self.vies + sum(self.pesos.get(n, 0.0) * v for n, v in x.items() if v)

    def ajustar(self, X: list[dict], y: list[int]):
        positivos = sum(y)
        peso_pos = 1.0
        if self.peso_classe_rara and positivos:
            peso_pos = min((len(y) - positivos) / positivos, self.peso_max)
            peso_pos = max(peso_pos, 1.0)
        self.pesos, self.vies = {}, 0.0
        ordem = list(range(len(X)))
        rng = random.Random(self.semente)
        for epoca in range(self.epocas):
            rng.shuffle(ordem)
            passo = self.taxa / (1.0 + epoca)
            for i in ordem:
                erro = _sigmoide(self._z(X[i])) - y[i]
                if y[i]:
                    erro *= peso_pos
                self.vies -= passo * erro
                for nome, valor in X[i].items():
                    if valor:
                        w = self.pesos.get(nome, 0.0)
                        self.pesos[nome] = w - passo * (erro * valor + self.l2 * w)
        return self

    def probabilidade(self, x: dict) -> float:
        return _sigmoide(self._z(x))

    def prever(self, x: dict) -> int:
        return 1 if self.probabilidade(x) >= 0.5 else 0

    def para_dict(self) -> dict:
        return {"pesos": self.pesos, "vies": self.vies}

    @classmethod
    def de_dict(cls, d: dict):
        obj = cls()
        obj.pesos, obj.vies = dict(d["pesos"]), d["vies"]
        return obj

    def pesos_principais(self, n: int = 15) -> list[tuple[str, float]]:
        """Os atributos de maior peso absoluto: o que o modelo aprendeu a olhar."""
        return sorted(self.pesos.items(), key=lambda kv: -abs(kv[1]))[:n]


class UmContraTodos:
    """Multiclasse a partir do binário: um modelo por relação ("esta ou não").

    O rótulo `NENHUMA` entra como mais uma classe. A previsão é a classe cujo modelo
    está mais confiante, então um par é "relação" quando a vencedora não é NENHUMA.
    """

    def __init__(self, **parametros):
        self.parametros = parametros
        self.classes: list[str] = []
        self.modelos: dict[str, RegressaoLogistica] = {}

    def ajustar(self, X: list[dict], rotulos: list[str]):
        self.classes = sorted(set(rotulos))
        self.modelos = {}
        for c in self.classes:
            binario = [1 if r == c else 0 for r in rotulos]
            self.modelos[c] = RegressaoLogistica(**self.parametros).ajustar(X, binario)
        return self

    def probabilidades(self, x: dict) -> dict[str, float]:
        bruto = {c: m.probabilidade(x) for c, m in self.modelos.items()}
        total = sum(bruto.values()) or 1.0
        return {c: p / total for c, p in bruto.items()}

    def prever(self, x: dict) -> str:
        return max(self.classes, key=lambda c: self.modelos[c].probabilidade(x))

    def tem_relacao(self, x: dict) -> int:
        return 0 if self.prever(x) == NENHUMA else 1

    def para_dict(self) -> dict:
        return {"classes": self.classes,
                "modelos": {c: m.para_dict() for c, m in self.modelos.items()}}

    @classmethod
    def de_dict(cls, d: dict):
        obj = cls()
        obj.classes = list(d["classes"])
        obj.modelos = {c: RegressaoLogistica.de_dict(m) for c, m in d["modelos"].items()}
        return obj


# --------------------------------------------------------------------------
# Empilhamento (stacking, Wolpert 1992)
# --------------------------------------------------------------------------
LIMITE_LM = 3.0   # a diferença de log-probabilidades é cortada em [-3, 3] e escalada


class PontuadorLM:
    """Nível 0: dois modelos de linguagem, um por resposta.

    Um aprende com o texto entre as entidades dos pares COM relação, o outro dos
    pares SEM relação. O atributo é a diferença de suas log-probabilidades médias:
    positivo quando o texto "soa" como o de uma relação, negativo quando não.
    Isso comprime milhares de n-gramas em um único número, que é o que 413 exemplos
    conseguem sustentar (n-gramas soltos como atributos pioram o modelo).
    """

    def __init__(self, k: float = 0.4, peso_bigrama: float = 0.7):
        self.k, self.peso_bigrama = k, peso_bigrama

    def ajustar(self, exemplos: list[Exemplo]):
        com = [e.entre for e in exemplos if e.eh_relacao]
        sem = [e.entre for e in exemplos if not e.eh_relacao]
        self.com = ModeloLinguagem(self.k, self.peso_bigrama).ajustar(com)
        self.sem = ModeloLinguagem(self.k, self.peso_bigrama).ajustar(sem)
        return self

    def para_dict(self) -> dict:
        return {"k": self.k, "peso_bigrama": self.peso_bigrama,
                "com": self.com.para_dict(), "sem": self.sem.para_dict()}

    @classmethod
    def de_dict(cls, d: dict):
        obj = cls(d["k"], d["peso_bigrama"])
        obj.com, obj.sem = ModeloLinguagem.de_dict(d["com"]), ModeloLinguagem.de_dict(d["sem"])
        return obj

    def pontuar(self, entre: list) -> float:
        diferenca = self.com.log_prob_media(entre) - self.sem.log_prob_media(entre)
        return max(min(diferenca, LIMITE_LM), -LIMITE_LM) / LIMITE_LM


def pontuacoes_fora_de_amostra(exemplos: list[Exemplo], **parametros) -> dict:
    """O `lm_score` de cada exemplo, medido por modelos que NÃO viram o seu caso.

    É o ponto central do empilhamento. Se o nível 1 treinasse com scores dados por
    modelos que já leram o próprio exemplo (e seu rótulo), veria um sinal bom
    demais, aprenderia a confiar nele e falharia em texto novo. Por isso cada
    exemplo é pontuado por um modelo treinado nos outros casos (validação
    cruzada por caso, não por exemplo: pares do mesmo relato dividem frases).
    """
    casos = sorted({e.caso for e in exemplos})
    saida = {}
    for caso in casos:
        dentro = [e for e in exemplos if e.caso == caso]
        fora = [e for e in exemplos if e.caso != caso]
        pontuador = PontuadorLM(**parametros).ajustar(fora)
        for e in dentro:
            saida[e.chave] = pontuador.pontuar(e.entre)
    return saida


class Empilhamento:
    """Gera os atributos do nível 1 com o `lm_score` do nível 0.

        e = Empilhamento().ajustar(treino)
        X_treino = [e.atributos_treino(x, assoc) for x in treino]   # score fora de amostra
        X_teste  = [e.atributos_teste(x, assoc) for x in teste]     # score do modelo completo

    No treino usa-se o score fora de amostra; na previsão, o dos modelos de
    linguagem treinados em todo o `treino`.
    """

    def __init__(self, **parametros):
        self.parametros = parametros

    def ajustar(self, treino: list[Exemplo]):
        self.fora_de_amostra = pontuacoes_fora_de_amostra(treino, **self.parametros)
        self.pontuador = PontuadorLM(**self.parametros).ajustar(treino)
        return self

    def atributos_treino(self, exemplo: Exemplo, assoc=None) -> dict:
        return {**atributos(exemplo, assoc), "lm_score": self.fora_de_amostra[exemplo.chave]}

    def atributos_teste(self, exemplo: Exemplo, assoc=None) -> dict:
        return {**atributos(exemplo, assoc), "lm_score": self.pontuador.pontuar(exemplo.entre)}


# --------------------------------------------------------------------------
# O modelo completo, com tudo o que ele precisa para decidir um par novo
# --------------------------------------------------------------------------
class ModeloAprendido:
    """Atributos estruturais + PMI/LLR + `lm_score` (empilhamento) + regressão.

    Reúne o que o treino produz (contagens de PMI, os dois modelos de linguagem e
    os pesos) para que a avaliação, a extração e o Streamlit usem o mesmo objeto, e
    para que ele seja gravado em `modelo.json` e recarregado sem retreinar.

    `usar_pmi` e `usar_lm` ligam e desligam cada peça (as ablações do experimento).
    """

    def __init__(self, usar_pmi: bool = True, usar_lm: bool = True, **regressao):
        self.usar_pmi, self.usar_lm, self.regressao = usar_pmi, usar_lm, regressao

    def ajustar(self, exemplos: list[Exemplo], textos_corpus, classificador=None):
        """`textos_corpus`: textos dos casos de treino, de onde saem as contagens de PMI."""
        self.assoc = Associacoes(textos_corpus) if self.usar_pmi else None
        self.empilhamento = Empilhamento().ajustar(exemplos) if self.usar_lm else None
        if self.empilhamento:
            X = [self.empilhamento.atributos_treino(e, self.assoc) for e in exemplos]
        else:
            X = [atributos(e, self.assoc) for e in exemplos]
        self.classificador = (classificador or UmContraTodos(**self.regressao)).ajustar(
            X, [e.relacao for e in exemplos])
        return self

    def atributos_de(self, estruturais: dict, entre: list) -> dict:
        f = dict(estruturais)
        if self.assoc is not None:
            f.update(atributos_associacao(entre, self.assoc))
        if self.empilhamento is not None:
            f["lm_score"] = self.empilhamento.pontuador.pontuar(entre)
        return f

    def prever(self, estruturais: dict, entre: list) -> tuple[str, float]:
        """(relação prevista ou NENHUMA, probabilidade normalizada dessa classe)."""
        x = self.atributos_de(estruturais, entre)
        probs = self.classificador.probabilidades(x)
        classe = self.classificador.prever(x)
        return classe, probs[classe]

    def prever_exemplo(self, e: Exemplo) -> str:
        return self.prever(e.estruturais, e.entre)[0]

    def para_dict(self) -> dict:
        return {
            "usar_pmi": self.usar_pmi, "usar_lm": self.usar_lm,
            "associacoes": self.assoc.para_dict() if self.assoc else None,
            "pontuador": self.empilhamento.pontuador.para_dict() if self.empilhamento else None,
            "classificador": self.classificador.para_dict(),
        }

    @classmethod
    def de_dict(cls, d: dict):
        obj = cls(d["usar_pmi"], d["usar_lm"])
        obj.assoc = Associacoes.de_dict(d["associacoes"]) if d["associacoes"] else None
        if d["pontuador"]:
            obj.empilhamento = Empilhamento()
            obj.empilhamento.pontuador = PontuadorLM.de_dict(d["pontuador"])
        else:
            obj.empilhamento = None
        obj.classificador = UmContraTodos.de_dict(d["classificador"])
        return obj


# --------------------------------------------------------------------------
# Verificação com scikit-learn
# --------------------------------------------------------------------------
def sklearn_disponivel() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


class UmContraTodosSklearn:
    """Mesma ideia e mesma interface de `UmContraTodos`, com a regressão logística
    do scikit-learn. Serve para verificar a implementação à mão: se as duas dão
    resultados próximos nos mesmos dados, o SGD escrito aqui está correto.

    Os hiperparâmetros não são idênticos (scikit-learn usa um solver de segunda
    ordem e `C` em vez de `l2`), então esperamos resultados próximos, não iguais.
    A classe rara recebe o mesmo peso limitado que na versão à mão.
    """

    def __init__(self, C: float = 1.0, peso_max: float = 20.0, semente: int = 0):
        self.C, self.peso_max, self.semente = C, peso_max, semente
        self.classes: list[str] = []
        self.modelos: dict = {}

    def ajustar(self, X: list[dict], rotulos: list[str]):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vetorizador = DictVectorizer(sparse=False)
        matriz = self.vetorizador.fit_transform(X)
        self.classes = sorted(set(rotulos))
        self.modelos = {}
        for c in self.classes:
            binario = [1 if r == c else 0 for r in rotulos]
            positivos = sum(binario)
            peso = max(min((len(binario) - positivos) / positivos, self.peso_max), 1.0)
            self.modelos[c] = LogisticRegression(
                C=self.C, class_weight={0: 1.0, 1: peso}, max_iter=1000,
                random_state=self.semente).fit(matriz, binario)
        return self

    def _probabilidades_brutas(self, x: dict) -> dict[str, float]:
        linha = self.vetorizador.transform([x])
        return {c: float(m.predict_proba(linha)[0][1]) for c, m in self.modelos.items()}

    def probabilidades(self, x: dict) -> dict[str, float]:
        bruto = self._probabilidades_brutas(x)
        total = sum(bruto.values()) or 1.0
        return {c: p / total for c, p in bruto.items()}

    def prever(self, x: dict) -> str:
        bruto = self._probabilidades_brutas(x)
        return max(self.classes, key=lambda c: bruto[c])

    def tem_relacao(self, x: dict) -> int:
        return 0 if self.prever(x) == NENHUMA else 1
