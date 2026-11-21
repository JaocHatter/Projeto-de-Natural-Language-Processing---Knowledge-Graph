"""O atendimento: o estudante pergunta e pede exames; as respostas saem só da ficha.

Sem LLM, o diálogo é por ações (perguntar por um sintoma, examinar, pedir um exame), o que já
basta para treinar o raciocínio. Uma pergunta livre é casada com os itens da ficha pelas
palavras de conteúdo; o que não está na ficha recebe "não reparei" (e nunca um fato inventado).

`Atendimento` guarda o que foi revelado, para o feedback e para o subgrafo.
"""

import re
from dataclasses import dataclass, field

from .ficha import Ficha, ItemFicha

PALAVRAS_VAZIAS = set("""a an the of and or in on with without to for do does did you your have has
had any some is are was were be been feel feeling felt there this that it i me my about any
ever noticed notice experienced experience problem problems history""".split())


def palavras(texto: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", texto.lower()) if w not in PALAVRAS_VAZIAS and len(w) > 1}


def casar(pergunta: str, itens: list[ItemFicha]) -> list[ItemFicha]:
    """Itens cujo nome está contido na pergunta (todas as palavras de conteúdo do nome)."""
    p = palavras(pergunta)
    return [i for i in itens if palavras(i.nome) and palavras(i.nome) <= p]


@dataclass
class Resposta:
    texto: str
    revelados: list = field(default_factory=list)      # ItemFicha revelados por esta resposta


@dataclass
class Atendimento:
    ficha: Ficha
    historico: list = field(default_factory=list)      # (quem, texto)
    revelados: dict = field(default_factory=dict)      # cui -> ItemFicha
    exames_pedidos: list = field(default_factory=list)

    def _registrar(self, pergunta: str, resposta: Resposta) -> Resposta:
        self.historico += [("estudante", pergunta), ("paciente", resposta.texto)]
        for i in resposta.revelados:
            self.revelados[i.cui] = i
        return resposta

    # -- ações ---------------------------------------------------------------
    def queixa_principal(self) -> Resposta:
        queixas = self.ficha.queixas()
        from .narrador import _lista
        nomes = [i.nome for i in queixas]
        texto = f"I've been having {_lista(nomes)}." if nomes else "I just don't feel well."
        return self._registrar("What brings you here today?", Resposta(texto, queixas))

    def perguntar(self, pergunta: str) -> Resposta:
        clinicos = [i for i in self.ficha.itens if i.categoria in ("Symptom", "Finding")]
        achados = casar(pergunta, clinicos)
        if not achados:
            return self._registrar(pergunta, Resposta("No, I haven't noticed anything like that."))
        partes = []
        for i in achados:
            partes.append(f"Yes, I have {i.nome}." if i.presente else f"No, I don't have {i.nome}.")
        return self._registrar(pergunta, Resposta(" ".join(partes), achados))

    def examinar(self) -> Resposta:
        achados = self.ficha.de("Finding")
        texto = ("On examination: " + "; ".join(i.nome for i in achados) + ".") if achados else \
            "Physical examination is unremarkable."
        return self._registrar("[physical examination]", Resposta(texto, achados))

    def pedir_exame(self, nome: str) -> Resposta:
        self.exames_pedidos.append(nome)
        exame = next((i for i in self.ficha.de("Exam") if i.nome == nome), None)
        if exame is None:
            return self._registrar(f"[exam] {nome}", Resposta(f"{nome}: no abnormality described."))
        if exame.resultado:
            texto = f"{nome}: shows " + "; ".join(exame.resultado) + "."
        elif exame.valores:
            texto = f"{nome}: " + "; ".join(exame.valores) + "."
        else:
            texto = f"{nome}: performed; findings not specific."
        mostrados = [i for i in self.ficha.itens if i.cui in exame.resultado_cuis]
        return self._registrar(f"[exam] {nome}", Resposta(texto, [exame, *mostrados]))

    # -- resumo --------------------------------------------------------------
    def presentes_revelados(self) -> set[str]:
        return {c for c, i in self.revelados.items() if i.presente and i.categoria != "Exam"}

    def negados_revelados(self) -> set[str]:
        return {c for c, i in self.revelados.items() if not i.presente}


def menu_de_exames(ficha: Ficha, perfis: dict, n_concorrentes: int = 4) -> list[str]:
    """Os exames da ficha misturados aos dos diagnósticos concorrentes (o menu não entrega a resposta)."""
    from .ficha import concorrentes
    nomes = {i.nome for i in ficha.de("Exam")}
    perfil = perfis.get(ficha.dx_cui)
    if perfil is not None:
        for outro, _ in concorrentes(perfil, perfis, n_concorrentes):
            nomes |= {f.nome for f in outro.ordenados("Exam")[:2]}
    return sorted(nomes)
