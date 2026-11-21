"""Cache das respostas do LLM, um JSON por linha, só acrescentado (nunca reescrito).

A chave é o hash de tudo o que define a resposta: modelo, mensagem do sistema, prompt,
temperatura e o índice da amostra (para a autoconsistência, k amostras do mesmo prompt são
k entradas diferentes). Com o cache versionado junto aos resultados, os números do P3 se
refazem sem chamar o modelo, e o custo em chamadas de cada variante é exato.
"""

import hashlib
import json
from pathlib import Path

from clinical_kg.paths import LLM_CACHE


def chave_pedido(modelo: str, sistema: str, prompt: str, temperatura: float, amostra: int) -> str:
    bruto = json.dumps([modelo, sistema, prompt, round(float(temperatura), 3), int(amostra)],
                       ensure_ascii=False)
    return hashlib.sha256(bruto.encode("utf-8")).hexdigest()


class CacheLLM:
    def __init__(self, caminho: Path = LLM_CACHE):
        self.caminho = Path(caminho)
        self.respostas: dict[str, str] = {}
        if self.caminho.exists():
            with open(self.caminho, encoding="utf-8") as f:
                for linha in f:
                    if linha.strip():
                        r = json.loads(linha)
                        self.respostas[r["chave"]] = r["resposta"]

    def __contains__(self, chave: str) -> bool:
        return chave in self.respostas

    def __len__(self) -> int:
        return len(self.respostas)

    def obter(self, chave: str):
        return self.respostas.get(chave)

    def gravar(self, chave: str, resposta: str, **metadados) -> None:
        self.respostas[chave] = resposta
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with open(self.caminho, "a", encoding="utf-8") as f:
            f.write(json.dumps({"chave": chave, "resposta": resposta, **metadados}, ensure_ascii=False) + "\n")
