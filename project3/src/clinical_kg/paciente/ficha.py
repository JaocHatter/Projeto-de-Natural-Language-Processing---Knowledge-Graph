"""A ficha de um paciente virtual: uma amostra do perfil de um diagnóstico.

A ficha é o gold do paciente virtual *por construção*: tudo o que o narrador e o diálogo dizem
sai dela, e cada item guarda os casos reais que o sustentam. A dificuldade decide a mistura:

    facil     só fatos típicos (os mais frequentes no perfil), nenhum distrator
    medio     típicos + um fato raro + um achado negado tirado de um diagnóstico concorrente
    dificil   menos típicos, mais raros e dois negados

Os "negados" são fatos frequentes em diagnósticos parecidos e ausentes deste perfil: o paciente
os nega, o que obriga o estudante a descartar o diferencial, e não só a reconhecer o padrão.
"""

import math
import random
from dataclasses import asdict, dataclass, field

from .perfil import Perfil, assinatura

DIFICULDADES = {
    #           típicos  raros  negados  exames  tratamentos
    "facil":   (4,       0,     0,       3,      2),
    "medio":   (3,       1,     1,       3,      2),
    "dificil": (2,       2,     2,       2,      2),
}
LIMIAR_TIPICO = 0.3


@dataclass
class ItemFicha:
    cui: str
    nome: str
    categoria: str                 # Symptom, Finding, Exam, Treatment
    presente: bool = True          # False: o paciente nega (distrator)
    frequencia: float = 0.0        # no perfil do diagnóstico (0 para os negados)
    casos: list = field(default_factory=list)
    resultado: list = field(default_factory=list)   # exames: nomes do que o exame mostra
    resultado_cuis: list = field(default_factory=list)
    valores: list = field(default_factory=list)     # exames: valores com unidade


@dataclass
class Ficha:
    dx_cui: str
    dx_nome: str
    idade: str
    sexo: str
    dificuldade: str
    semente: int
    itens: list
    casos_do_perfil: list
    localizacao: str = ""

    def de(self, categoria: str, presente: bool = True) -> list[ItemFicha]:
        return [i for i in self.itens if i.categoria == categoria and i.presente == presente]

    def negados(self) -> list[ItemFicha]:
        return [i for i in self.itens if not i.presente]

    def queixas(self) -> list[ItemFicha]:
        return self.de("Symptom")

    def para_dict(self) -> dict:
        return asdict(self)


def cosseno(a: dict, b: dict) -> float:
    num = sum(v * b.get(k, 0.0) for k, v in a.items())
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def concorrentes(perfil: Perfil, perfis: dict, n: int = 5) -> list[tuple[Perfil, float]]:
    """Os diagnósticos cujo perfil mais se parece com este (o diferencial do grafo)."""
    alvo = assinatura(perfil)
    outros = [(p, cosseno(alvo, assinatura(p))) for cui, p in perfis.items() if cui != perfil.cui]
    return sorted(outros, key=lambda x: -x[1])[:n]


def _revela_o_diagnostico(nome: str, dx_nome: str) -> bool:
    a, b = nome.lower(), dx_nome.lower()
    return a in b or b in a


def _sobrepostos(a: str, b: str) -> bool:
    """"vomiting" e "nausea and vomiting" dizem a mesma coisa: só um entra na ficha."""
    a, b = a.lower(), b.lower()
    return a in b or b in a


def _amostra(rng, fatos, k, pesos=None, ja=()):
    """`k` fatos sem reposição, com probabilidade proporcional aos pesos, sem repetir o que já está em `ja`."""
    fatos, pesos = list(fatos), list(pesos or [1.0] * len(fatos))
    escolhidos, nomes = [], [getattr(x, "nome", x) for x in ja]
    while fatos and len(escolhidos) < k:
        i = rng.choices(range(len(fatos)), weights=pesos)[0]
        f = fatos.pop(i)
        pesos.pop(i)
        if any(_sobrepostos(f.nome, n) for n in nomes):
            continue
        escolhidos.append(f)
        nomes.append(f.nome)
    return escolhidos


def montar_ficha(perfil: Perfil, perfis: dict, dificuldade: str = "medio", semente: int = 0,
                 nomes: dict | None = None) -> Ficha:
    n_tip, n_raro, n_neg, n_exa, n_trat = DIFICULDADES[dificuldade]
    rng = random.Random(f"{perfil.cui}:{dificuldade}:{semente}")
    n = len(perfil.casos)
    nomes = nomes or {}

    def item(f, presente=True):
        return ItemFicha(f.cui, f.nome, f.categoria, presente, f.frequencia(n) if presente else 0.0,
                         sorted(f.casos))

    clinicos = [f for f in perfil.ordenados() if f.categoria in ("Symptom", "Finding")
                and not _revela_o_diagnostico(f.nome, perfil.nome)]
    tipicos = [f for f in clinicos if f.frequencia(n) >= LIMIAR_TIPICO] or clinicos[:n_tip]
    # Um fato de um caso só é, quase sempre, ruído do gazetteer: "raro" exige dois pacientes.
    raros = [f for f in clinicos if f not in tipicos and len(f.casos) >= 2]
    itens = [item(f) for f in _amostra(rng, tipicos, n_tip, [f.frequencia(n) for f in tipicos])]
    itens += [item(f) for f in _amostra(rng, raros, n_raro, ja=itens)]

    # Exames: os mais frequentes no perfil; no empate, os que o grafo liga ao diagnóstico.
    exames = sorted(perfil.ordenados("Exam"), key=lambda f: (-len(f.casos), -f.ligado))
    presentes = {i.cui for i in itens}
    for f in exames[:n_exa]:
        it = item(f)
        mostra = [cui for cui, _ in f.revela.most_common() if cui in presentes or cui == perfil.cui]
        it.resultado_cuis = [c for c in mostra if c != perfil.cui]
        it.resultado = [nomes.get(c, c) for c in it.resultado_cuis]
        if f.valores:      # os valores de UM paciente real, para não misturar pessoas
            caso_valor = rng.choice(sorted({c for c, _ in f.valores}))
            it.valores = [v for c, v in f.valores if c == caso_valor][:2]
        itens.append(it)

    tratamentos = sorted(perfil.ordenados("Treatment"), key=lambda f: (-len(f.casos), -f.ligado))
    itens += [item(f) for f in tratamentos[:n_trat]]

    # Negados: sintomas frequentes nos concorrentes e ausentes deste perfil ("denies nausea").
    ausentes = []
    for outro, _ in concorrentes(perfil, perfis):
        m = len(outro.casos)
        for f in outro.ordenados():
            if (f.categoria == "Symptom" and f.cui not in perfil.fatos
                    and f.frequencia(m) >= LIMIAR_TIPICO and f.cui not in {a.cui for a in ausentes}
                    and not any(_sobrepostos(f.nome, x.nome) for x in [*itens, *ausentes])
                    and not _revela_o_diagnostico(f.nome, outro.nome)):
                ausentes.append(f)
    itens += [item(f, presente=False) for f in ausentes[:n_neg]]

    idade, sexo = rng.choice(perfil.demografia) if perfil.demografia else ("", "")
    local = perfil.localizacoes.most_common(1)[0][0] if perfil.localizacoes else ""
    return Ficha(perfil.cui, perfil.nome, idade, sexo, dificuldade, semente, itens,
                 sorted(perfil.casos), local)
