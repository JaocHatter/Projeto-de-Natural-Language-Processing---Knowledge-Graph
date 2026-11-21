"""Quem responde aos prompts. Todo provedor passa pelo cache.

Configuração por variáveis de ambiente (nenhuma é obrigatória):

    CLINICAL_KG_LLM_URL     endpoint compatível com a API de chat da OpenAI, por exemplo
                            http://localhost:11434/v1 (Ollama, local e gratuito) ou o de um
                            provedor pago. Sem ela, só o cache é lido.
    CLINICAL_KG_LLM_MODELO  nome do modelo (ex.: llama3.1:8b, qwen2.5:7b)
    CLINICAL_KG_LLM_CHAVE   chave da API, se o endpoint exigir

    provedor_padrao()       o provedor que essas variáveis descrevem
"""

import json
import os
import time
import urllib.request

from .cache import CacheLLM, chave_pedido


class RespostaAusente(RuntimeError):
    """O pedido não está no cache e não há endpoint configurado para respondê-lo."""


class Provedor:
    """Base: `completar` consulta o cache e só chama `_chamar` para o que falta."""

    def __init__(self, modelo: str, cache: CacheLLM | None = None):
        self.modelo = modelo
        self.cache = cache if cache is not None else CacheLLM()
        self.chamadas = 0          # chamadas reais ao modelo nesta execução
        self.acertos_cache = 0

    def _chamar(self, sistema: str, prompt: str, temperatura: float) -> str:
        raise RespostaAusente("provedor sem acesso a um modelo; configure CLINICAL_KG_LLM_URL")

    def completar(self, prompt: str, sistema: str = "", temperatura: float = 0.0, amostra: int = 0) -> str:
        chave = chave_pedido(self.modelo, sistema, prompt, temperatura, amostra)
        guardada = self.cache.obter(chave)
        if guardada is not None:
            self.acertos_cache += 1
            return guardada
        resposta = self._chamar(sistema, prompt, temperatura)
        self.chamadas += 1
        self.cache.gravar(chave, resposta, modelo=self.modelo, temperatura=temperatura, amostra=amostra)
        return resposta


class ProvedorCache(Provedor):
    """Só lê o cache: reproduz os resultados sem rede e sem chave."""


class ProvedorOpenAI(Provedor):
    """Qualquer endpoint `/chat/completions` no formato da OpenAI (inclui Ollama e vLLM)."""

    def __init__(self, url: str, modelo: str, chave: str = "", cache: CacheLLM | None = None,
                 tempo_limite: int = 120, tentativas: int = 3):
        super().__init__(modelo, cache)
        self.url = url.rstrip("/") + "/chat/completions"
        self.chave, self.tempo_limite, self.tentativas = chave, tempo_limite, tentativas

    def _chamar(self, sistema, prompt, temperatura):
        mensagens = ([{"role": "system", "content": sistema}] if sistema else []) + \
                    [{"role": "user", "content": prompt}]
        corpo = json.dumps({"model": self.modelo, "messages": mensagens,
                            "temperature": temperatura}).encode("utf-8")
        cabecalhos = {"Content-Type": "application/json"}
        if self.chave:
            cabecalhos["Authorization"] = f"Bearer {self.chave}"
        for tentativa in range(self.tentativas):
            try:
                pedido = urllib.request.Request(self.url, corpo, cabecalhos)
                with urllib.request.urlopen(pedido, timeout=self.tempo_limite) as r:
                    return json.load(r)["choices"][0]["message"]["content"]
            except (OSError, KeyError, ValueError):
                if tentativa == self.tentativas - 1:
                    raise
                time.sleep(2 ** tentativa)


class ProvedorFalso(Provedor):
    """Para os testes: `responder(sistema, prompt, temperatura) -> str`, sem rede."""

    def __init__(self, responder, modelo: str = "falso", cache: CacheLLM | None = None):
        super().__init__(modelo, cache)
        self.responder = responder

    def _chamar(self, sistema, prompt, temperatura):
        return self.responder(sistema, prompt, temperatura)


def provedor_padrao(cache: CacheLLM | None = None) -> Provedor:
    modelo = os.environ.get("CLINICAL_KG_LLM_MODELO", "sem-modelo")
    url = os.environ.get("CLINICAL_KG_LLM_URL")
    if url:
        return ProvedorOpenAI(url, modelo, os.environ.get("CLINICAL_KG_LLM_CHAVE", ""), cache)
    return ProvedorCache(modelo, cache)
