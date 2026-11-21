"""P3-D: embeddings de sentença como mais uma fonte de atributos do classificador do P2.

Cada par candidato vira um texto (as duas entidades e o que há entre elas) e um vetor de um
modelo de sentenças biomédico. Os 768 números NÃO entram direto na regressão: com 413 pares
julgados isso seria um parâmetro por 0,5 exemplo, e o P2 já mostrou que n-gramas soltos como
atributos pioram o modelo. Como o `lm_score` do P2, o vetor é comprimido em um único número por
empilhamento (Wolpert, 1992):

    nível 0: regressão logística sobre os vetores, "este par tem relação?"
    nível 1: a regressão do P2, com `emb_score` = 2·P(relação) − 1 como atributo

O nível 0 é avaliado fora de amostra por caso no treino, igual ao `lm_score`.

Os vetores ficam em cache em data/interim/cache/embeddings/ (chave: hash do texto), de modo que só a
primeira execução precisa do modelo; as seguintes nem importam o torch.
"""

import hashlib
import json
from pathlib import Path

from clinical_kg.paths import EMBEDDINGS_CACHE

MODELO_PADRAO = "NeuML/pubmedbert-base-embeddings"


def texto_do_par(texto: str, chave: tuple) -> str:
    """As duas entidades e o que há entre elas, na ordem do texto.

    `chave` = (caso, cab_ini, cab_fim, cau_ini, cau_fim); cabeça -1 é o paciente, e aí o trecho
    começa no início da sentença da cauda (aproximado pelo último ponto antes dela).
    """
    _, ci, cf, ti, tf = chave
    if ci < 0:
        inicio = max(texto.rfind(".", 0, ti) + 1, texto.rfind("\n", 0, ti) + 1)
        return texto[inicio:tf].strip()
    return texto[min(ci, ti):max(cf, tf)].strip()


def _hash(texto: str) -> str:
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


class Codificador:
    """Modelo de sentenças com cache em disco (um .jsonl por modelo)."""

    def __init__(self, modelo: str = MODELO_PADRAO, pasta: Path = EMBEDDINGS_CACHE):
        self.modelo = modelo
        self.caminho = Path(pasta) / (modelo.replace("/", "__") + ".jsonl")
        self.cache: dict[str, list[float]] = {}
        self._st = None
        if self.caminho.exists():
            with open(self.caminho, encoding="utf-8") as f:
                for linha in f:
                    r = json.loads(linha)
                    self.cache[r["h"]] = r["v"]

    def _carregar(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.modelo, device="cpu")
        return self._st

    def codificar(self, textos: list[str], lote: int = 64) -> list[list[float]]:
        """Vetores normalizados (norma 1), na ordem de `textos`."""
        faltam = sorted({t for t in textos if _hash(t) not in self.cache})
        if faltam:
            vetores = self._carregar().encode(faltam, batch_size=lote, normalize_embeddings=True,
                                              show_progress_bar=len(faltam) > 500)
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
            with open(self.caminho, "a", encoding="utf-8") as f:
                for t, v in zip(faltam, vetores):
                    lista = [round(float(x), 5) for x in v]
                    self.cache[_hash(t)] = lista
                    f.write(json.dumps({"h": _hash(t), "v": lista}) + "\n")
        return [self.cache[_hash(t)] for t in textos]


def vetores_dos_exemplos(exemplos, textos: dict[str, str], codificador: Codificador) -> dict:
    """{chave do exemplo: vetor}. `textos` indexa pelo `caso` de cada exemplo."""
    pares = [(e.chave, texto_do_par(textos[e.caso], e.chave)) for e in exemplos]
    vetores = codificador.codificar([t for _, t in pares])
    return {chave: v for (chave, _), v in zip(pares, vetores)}


class PontuadorEmbeddings:
    """Nível 0: "o par tem relação?" a partir só do vetor (regressão logística do scikit-learn)."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def ajustar(self, exemplos, vetores: dict):
        from sklearn.linear_model import LogisticRegression

        y = [e.eh_relacao for e in exemplos]
        self.constante = None
        if len(set(y)) < 2:              # sem as duas classes não há o que aprender
            self.constante = 1.0 if y and y[0] else -1.0
            return self
        self.modelo = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000)
        self.modelo.fit([vetores[e.chave] for e in exemplos], y)
        return self

    def pontuar(self, vetor) -> float:
        if self.constante is not None:
            return self.constante
        return 2.0 * float(self.modelo.predict_proba([vetor])[0][1]) - 1.0


class FonteEmbeddings:
    """Fonte de atributos para `ModeloHibrido`: o `emb_score`, fora de amostra no treino."""

    nome = "emb_score"

    def __init__(self, vetores: dict, C: float = 1.0):
        self.vetores, self.C = vetores, C

    def ajustar(self, treino):
        self.fora = {}
        for caso in sorted({e.caso for e in treino}):
            dentro = [e for e in treino if e.caso == caso]
            fora = [e for e in treino if e.caso != caso]
            p = PontuadorEmbeddings(self.C).ajustar(fora, self.vetores)
            for e in dentro:
                self.fora[e.chave] = p.pontuar(self.vetores[e.chave])
        self.completo = PontuadorEmbeddings(self.C).ajustar(treino, self.vetores)
        return self

    def atributo_treino(self, e) -> dict:
        return {self.nome: self.fora[e.chave]}

    def atributo_teste(self, e) -> dict:
        return {self.nome: self.completo.pontuar(self.vetores[e.chave])}
