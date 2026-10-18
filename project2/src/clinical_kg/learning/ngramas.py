#!/usr/bin/env python3
"""N-gramas, medidas de associação (PMI e LLR) e modelo de linguagem bigrama.

Base estatística do Projeto 2. O Projeto 1 decidia se havia relação entre duas
entidades com uma lista escrita à mão de gatilhos ("treated with", "caused by").
Aqui o vocabulário de gatilhos vem do próprio corpus:

    contar_ngramas   quais sequências de palavras existem e quantas vezes
    Associacoes      quais bigramas se atraem mais do que o acaso (PMI, LLR)
    ModeloLinguagem  quão provável é um trecho de texto sob um corpus de exemplos

Tudo usa só a biblioteca padrão. As classes recebem os textos no construtor, de
modo que o pipeline possa recalculá-las dentro de cada dobra da validação sem
vazamento (ver `pipeline.py`).

Referências:
    Church & Hanks (1990): PMI para colocações.
    Dunning (1993): razão de verossimilhança (LLR) para dados esparsos.
    Jurafsky & Martin, SLP3: n-gramas e suavização.
"""

import collections
import math

from clinical_kg.extraction.entities import TOKEN_RE, mask_figure_refs

INICIO, FIM = "<s>", "</s>"

# Abaixo disto o PMI vale zero. Com um único par visto uma vez, PMI = log2(N)
# para qualquer par: todos empatam no máximo e nenhum é evidência.
CONTAGEM_MINIMA_PMI = 5


def tokenizar(texto: str) -> list[str]:
    """Palavras em minúsculas, com o mesmo tokenizador do restante do pipeline."""
    return [m.group().lower() for m in TOKEN_RE.finditer(mask_figure_refs(texto))]


def contar_ngramas(tokens: list[str], n_max: int = 3) -> collections.Counter:
    """Contagem de todos os n-gramas de tamanho 1..n_max (chaves são tuplas)."""
    contagem = collections.Counter()
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            contagem[tuple(tokens[i:i + n])] += 1
    return contagem


class Associacoes:
    """PMI e LLR de bigramas adjacentes, contados sobre uma lista de textos.

    PMI(a, b) = log2( P(a, b) / (P(a) * P(b)) ): quanto mais `a b` aparece juntos
    do que o acaso preveria. É alto para unidades como "myocardial infarction" e
    perto de zero para pares independentes.

    LLR (Dunning, 1993) testa a mesma pergunta com uma estatística que não
    supõe contagens esperadas grandes, que é o regime de um corpus pequeno.
    """

    def __init__(self, textos, contagem_minima: int = CONTAGEM_MINIMA_PMI):
        self.contagem_minima = contagem_minima
        self.unigramas = collections.Counter()
        self.bigramas = collections.Counter()
        for texto in textos:
            tokens = tokenizar(texto)
            self.unigramas.update(tokens)
            self.bigramas.update(zip(tokens, tokens[1:]))
        self.total = max(sum(self.bigramas.values()), 1)          # posições de bigrama
        self.total_palavras = max(sum(self.unigramas.values()), 1)
        # Marginais dos bigramas, para a tabela de contingência do LLR.
        self.esquerda = collections.Counter()
        self.direita = collections.Counter()
        for (a, b), c in self.bigramas.items():
            self.esquerda[a] += c
            self.direita[b] += c

    def pmi(self, a: str, b: str) -> float:
        c = self.bigramas.get((a, b), 0)
        if c < self.contagem_minima:
            return 0.0
        conjunta = c / self.total
        esperada = ((self.unigramas[a] / self.total_palavras)
                    * (self.unigramas[b] / self.total_palavras))
        return math.log2(conjunta / esperada) if esperada else 0.0

    def llr(self, a: str, b: str) -> float:
        k11 = self.bigramas.get((a, b), 0)
        if not k11:
            return 0.0
        k12 = self.esquerda[a] - k11           # a seguido de outra palavra
        k21 = self.direita[b] - k11            # b precedido de outra palavra
        k22 = self.total - k11 - k12 - k21     # nenhum dos dois

        def entropia(*c):
            soma = sum(c)
            return sum(x * math.log(x / soma) for x in c if x > 0) if soma else 0.0

        return 2 * (entropia(k11, k12, k21, k22)
                    - entropia(k11 + k12, k21 + k22)
                    - entropia(k11 + k21, k12 + k22))

    def ranking(self, medida: str = "pmi"):
        """Bigramas com contagem suficiente, do mais ao menos associado."""
        pontuar = self.pmi if medida == "pmi" else self.llr
        linhas = [(a, b, c, pontuar(a, b)) for (a, b), c in self.bigramas.items()
                  if c >= self.contagem_minima]
        return sorted(linhas, key=lambda r: -r[3])


class ModeloLinguagem:
    """Modelo bigrama com suavização add-k e retrocesso para unigramas.

        P(w | anterior) = L * P_bigrama + (1 - L) * P_unigrama

    O add-k (k < 1, mais suave que Laplace) evita probabilidade zero para o que o
    corpus nunca viu: 81,6% dos bigramas deste corpus aparecem uma única vez.

    Dois modelos são treinados (textos com relação e textos sem relação) e a
    diferença de suas log-probabilidades vira um único atributo (ver `modelo.py`).
    """

    def __init__(self, k: float = 0.4, peso_bigrama: float = 0.7):
        self.k, self.peso_bigrama = k, peso_bigrama

    def ajustar(self, documentos):
        """`documentos`: listas de tokens; recebe a marcação de fronteira aqui."""
        docs = [[INICIO, *d, FIM] for d in documentos]
        self.unigramas = collections.Counter(w for d in docs for w in d)
        self.bigramas = collections.Counter(p for d in docs for p in zip(d, d[1:]))
        self.total = sum(self.unigramas.values())
        self.vocabulario = len(self.unigramas) + 1   # +1 reserva massa para palavras novas
        return self

    def probabilidade(self, anterior: str, palavra: str) -> float:
        k, v = self.k, self.vocabulario
        bigrama = (self.bigramas[(anterior, palavra)] + k) / (self.unigramas[anterior] + k * v)
        unigrama = (self.unigramas[palavra] + k) / (self.total + k * v)
        return self.peso_bigrama * bigrama + (1 - self.peso_bigrama) * unigrama

    def log_prob_media(self, tokens) -> float:
        """Log-probabilidade média por palavra, para o tamanho não dominar."""
        doc = [INICIO, *tokens, FIM]
        pares = list(zip(doc, doc[1:]))
        return sum(math.log(self.probabilidade(a, b)) for a, b in pares) / len(pares)
