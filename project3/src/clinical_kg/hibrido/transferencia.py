"""P3-G: o classificador do P2 aprende também com os julgamentos humanos do MACCROBAT.

Testa a hipótese que o P2 deixou em aberto: para ir além das regras, faltam mais julgamentos
humanos, e não mais texto. Por isso só os rótulos vêm do MACCROBAT; as contagens de PMI continuam
sendo as do MultiCaRe, como no P2.

A transferência é feita só onde os dois esquemas concordam. No um-contra-todos do P2 cada
relação tem o seu modelo binário; os de CAUSED_BY e LOCATED_IN treinam com o gold + os pares do
MACCROBAT, e os outros seis (e NENHUMA) treinam só com o gold, porque o MACCROBAT não diz nada
sobre TREATED_WITH, REVEALED_BY etc.
"""

import collections

from clinical_kg.learning.modelo import (Empilhamento, ModeloAprendido, RegressaoLogistica,
                                         UmContraTodos, atributos)
from clinical_kg.learning.ngramas import Associacoes
from clinical_kg.maccrobat.mapeamento import TRANSFERIVEIS


class UmContraTodosComExtras(UmContraTodos):
    """`UmContraTodos` em que algumas classes treinam também com exemplos de outro corpus.

    `extras`: {classe: (X_extra, y_extra)}. O limite do peso da classe rara (`limitar_por_suporte`)
    passa a contar também os positivos externos.
    """

    def ajustar(self, X, rotulos, extras=None):
        extras = extras or {}
        # Uma classe ausente do gold da rodada ainda pode ser aprendida dos exemplos externos;
        # sem positivos externos, o conjunto de classes é o mesmo do P2.
        self.classes = sorted(set(rotulos) | {c for c, (_, y) in extras.items() if any(y)})
        self.modelos = {}
        for c in self.classes:
            Xc, yc = list(X), [1 if r == c else 0 for r in rotulos]
            if c in extras:
                Xc += extras[c][0]
                yc += extras[c][1]
            parametros = dict(self.parametros)
            if self.limitar_por_suporte:
                teto = parametros.get("peso_max", RegressaoLogistica().peso_max)
                parametros["peso_max"] = min(teto, sum(yc))
            self.modelos[c] = RegressaoLogistica(**parametros).ajustar(Xc, yc)
        return self


class ModeloHibrido(ModeloAprendido):
    """O `ModeloAprendido` do P2 com duas extensões, que podem ser ligadas juntas ou separadas.

    Exemplos externos (P3-G): entram nos modelos binários de `classes`. Recebem os atributos que o
    modelo usaria num texto novo: PMI das contagens do MultiCaRe e `lm_score` dos modelos de
    linguagem treinados no gold da rodada. Nada do MACCROBAT entra nessas contagens, então o
    único efeito medido é o dos rótulos.

    Fontes de atributos empilhadas (P3-D, P3-C): objetos com `ajustar(treino)`,
    `atributo_treino(e)` (fora de amostra) e `atributo_teste(e)`, como o `lm_score` do P2. Cada
    uma acrescenta um ou mais números ao dicionário do par. Como precisam do par inteiro (e não só
    das palavras entre as entidades), a previsão usa `prever_exemplo`.
    """

    def ajustar(self, exemplos, textos_corpus, externos=(), classes=TRANSFERIVEIS, fontes=()):
        self.assoc = Associacoes(textos_corpus) if self.usar_pmi else None
        self.empilhamento = Empilhamento().ajustar(exemplos) if self.usar_lm else None
        self.fontes = [f.ajustar(exemplos) for f in fontes]
        if self.empilhamento:
            X = [self.empilhamento.atributos_treino(e, self.assoc) for e in exemplos]
        else:
            X = [atributos(e, self.assoc) for e in exemplos]
        for x, e in zip(X, exemplos):
            for f in self.fontes:
                x.update(f.atributo_treino(e))
        X_externo = [self.atributos_exemplo(e) for e in externos]
        extras = {c: (X_externo, [1 if e.relacao == c else 0 for e in externos]) for c in classes}
        self.classificador = UmContraTodosComExtras(**self.regressao).ajustar(
            X, [e.relacao for e in exemplos], extras)
        return self

    def atributos_exemplo(self, e) -> dict:
        x = self.atributos_de(e.estruturais, e.entre)
        for f in getattr(self, "fontes", ()):
            x.update(f.atributo_teste(e))
        return x

    def prever_exemplo_com_probabilidade(self, e) -> tuple[str, float]:
        x = self.atributos_exemplo(e)
        classe = self.classificador.prever(x)
        return classe, self.classificador.probabilidades(x)[classe]

    def prever_exemplo(self, e) -> str:
        return self.prever_exemplo_com_probabilidade(e)[0]


ModeloTransferencia = ModeloHibrido      # nome usado antes de existirem as fontes empilhadas


def prever_validando_hibrido(exemplos, textos, flags, externos=(), classes=TRANSFERIVEIS,
                             fabricar_fontes=lambda: ()):
    """Deixa um caso do gold de fora por vez; o MACCROBAT (se houver) entra inteiro em todas as rodadas.

    `fabricar_fontes()` devolve fontes novas a cada rodada (elas guardam estado do treino).
    """
    saida = []
    for caso in sorted({e.caso for e in exemplos}):
        treino = [e for e in exemplos if e.caso != caso]
        corpus = [t for c, t in textos.items() if c != caso]
        modelo = ModeloHibrido(**flags).ajustar(treino, corpus, externos, classes, fabricar_fontes())
        saida += [(e, modelo.prever_exemplo(e)) for e in exemplos if e.caso == caso]
    return saida


def prever_validando_transferencia(exemplos, textos, externos, flags, classes=TRANSFERIVEIS):
    """P3-G: `prever_validando_hibrido` só com os exemplos externos."""
    return prever_validando_hibrido(exemplos, textos, flags, externos, classes)


def balancear(externos, proporcao_negativos: float, semente: int = 0):
    """Mantém todos os positivos externos e só `proporcao_negativos` negativos por positivo.

    O MACCROBAT tem dezenas de vezes mais pares que o gold; sem balancear, os negativos externos
    dominam os modelos binários transferidos.
    """
    import random

    positivos = [e for e in externos if e.relacao in TRANSFERIVEIS]
    negativos = [e for e in externos if e.relacao not in TRANSFERIVEIS]
    k = min(len(negativos), int(proporcao_negativos * len(positivos)))
    return positivos + random.Random(semente).sample(negativos, k)


def contagem_por_rotulo(exemplos) -> collections.Counter:
    return collections.Counter(e.relacao for e in exemplos)
