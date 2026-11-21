"""A ficha em texto. Dois narradores com a mesma interface:

    NarradorModelo   determinístico, frases-modelo em inglês (a língua dos extratores do P1)
    NarradorLLM      o LLM narra só a ficha; se não houver LLM configurado, cai no modelo

Cada narrador produz duas versões:

    apresentacao(ficha)   o que o estudante lê no início: idade, sexo e a queixa principal
    completo(ficha)       o caso inteiro (exame físico, exames, diagnóstico, tratamento),
                          que é o que volta aos extratores na ida e volta
"""

import json

from .ficha import Ficha


def _lista(nomes: list[str]) -> str:
    nomes = [n for n in nomes if n]
    if len(nomes) <= 1:
        return "".join(nomes)
    return ", ".join(nomes[:-1]) + " and " + nomes[-1]


def maiuscula(nome: str) -> str:
    """Siglas curtas em maiúsculas ("mri" -> "MRI"); o resto só com a inicial maiúscula."""
    if len(nome) <= 4 and " " not in nome:
        return nome.upper()
    return nome[0].upper() + nome[1:]


def _pessoa(ficha: Ficha) -> tuple[str, str, str]:
    """(descrição, pronome sujeito, possessivo)."""
    sexo = (ficha.sexo or "").lower()
    homem = sexo.startswith("m")
    mulher = sexo.startswith("f") or sexo.startswith("w")
    idade = f"{ficha.idade}-year-old " if ficha.idade else ""
    # "An 8-year-old", "An 18-year-old", "An 80-year-old": o artigo segue a pronúncia do número.
    artigo = "An" if ficha.idade and (ficha.idade.startswith("8") or ficha.idade in ("11", "18")) else "A"
    crianca = ficha.idade.isdigit() and int(ficha.idade) < 18 if ficha.idade else False
    if homem:
        return f"{artigo} {idade}{'boy' if crianca else 'man'}", "He", "his"
    if mulher:
        return f"{artigo} {idade}{'girl' if crianca else 'woman'}", "She", "her"
    return f"{artigo} {idade}patient", "The patient", "their"


class NarradorModelo:
    nome = "modelo"

    def apresentacao(self, ficha: Ficha) -> str:
        quem, _, _ = _pessoa(ficha)
        queixas = [i.nome for i in ficha.queixas()]
        return f"{quem} presented with {_lista(queixas)}." if queixas else f"{quem} presented for evaluation."

    def completo(self, ficha: Ficha) -> str:
        quem, ele, seu = _pessoa(ficha)
        frases = []
        sintomas = [i.nome for i in ficha.de("Symptom")]
        frases.append(f"{quem} presented with {_lista(sintomas)}." if sintomas
                      else f"{quem} presented for evaluation.")
        negados = [i.nome for i in ficha.negados()]
        if negados:
            frases.append(f"{ele} denied {_lista(negados)}.")
        achados = [i.nome for i in ficha.de("Finding")]
        if achados:
            frases.append(f"Physical examination revealed {_lista(achados)}.")
        for exame in ficha.de("Exam"):
            nome = maiuscula(exame.nome)
            if exame.resultado:
                frases.append(f"{nome} showed {_lista(exame.resultado)}.")
            elif exame.valores:
                frases.append(f"{nome} revealed {_lista(exame.valores)}.")
            else:
                verbo = "were" if exame.nome.endswith("s") and not exame.nome.endswith("ss") else "was"
                frases.append(f"{nome} {verbo} performed.")
        frases.append(f"The diagnosis of {ficha.dx_nome} was made.")
        tratamentos = [i.nome for i in ficha.de("Treatment")]
        if tratamentos:
            frases.append(f"{ele} was treated with {_lista(tratamentos)}.")
        return " ".join(frases)


INSTRUCAO_LLM = """You write a short clinical case report in English from a structured patient record.
Rules:
- Use ONLY the facts in the record. Mention every fact exactly once.
- Facts with "present": false are things the patient DENIES or that are ABSENT; state them as negated.
- Do not add any symptom, sign, exam, result, value, diagnosis or treatment that is not in the record.
- {extra}
Record (JSON):
{ficha}
Write the case report as one paragraph."""


class NarradorLLM(NarradorModelo):
    """O LLM reescreve a ficha em prosa; a ida e volta mede o quanto ele foi fiel."""

    nome = "llm"

    def __init__(self, provedor, temperatura: float = 0.7):
        self.provedor, self.temperatura = provedor, temperatura

    def _ficha_json(self, ficha: Ficha, completo: bool) -> str:
        itens = [{"name": i.nome, "category": i.categoria, "present": i.presente,
                  **({"result": i.resultado or i.valores} if i.categoria == "Exam" else {})}
                 for i in ficha.itens if completo or (i.presente and i.categoria == "Symptom")]
        dados = {"age": ficha.idade, "sex": ficha.sexo, "facts": itens}
        if completo:
            dados["diagnosis"] = ficha.dx_nome
        return json.dumps(dados, ensure_ascii=False)

    def _narrar(self, ficha: Ficha, completo: bool) -> str | None:
        from clinical_kg.llm.provedor import RespostaAusente
        extra = ("Include the diagnosis and the treatment." if completo else
                 "Write only the presenting complaint (one sentence); do not reveal any diagnosis.")
        prompt = INSTRUCAO_LLM.format(extra=extra, ficha=self._ficha_json(ficha, completo))
        try:
            return self.provedor.completar(prompt, temperatura=self.temperatura, amostra=ficha.semente).strip()
        except (RespostaAusente, OSError):
            return None

    def apresentacao(self, ficha: Ficha) -> str:
        return self._narrar(ficha, False) or super().apresentacao(ficha)

    def completo(self, ficha: Ficha) -> str:
        return self._narrar(ficha, True) or super().completo(ficha)
