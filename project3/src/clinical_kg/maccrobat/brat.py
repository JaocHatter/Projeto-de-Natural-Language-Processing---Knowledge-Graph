"""Leitor do formato brat standoff, o usado pelo MACCROBAT.

Cada documento tem um .txt (o relato) e um .ann com as anotações, uma por linha:

    T1	Sign_symptom 10 15	fever         trecho do texto (pode ser descontínuo: "10 15;20 25")
    E1	Sign_symptom:T1                     evento ancorado num trecho
    R1	MODIFY Arg1:T2 Arg2:E1              relação entre trechos ou eventos
    *	OVERLAP E1 E2 E3                    grupo de eventos simultâneos
    A1	POLARITY E1 NEG                     atributo de um evento ou trecho

Eventos são resolvidos para o trecho que os ancora, então relações, grupos e atributos
sempre apontam para ids de trechos (`T...`). O texto vem dos bytes do zip, sem conversão de
quebras de linha, porque os deslocamentos indexam o .txt exatamente como foi gravado.
"""

import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


@dataclass
class Trecho:
    id: str
    tipo: str
    inicio: int
    fim: int
    texto: str
    descontinuo: bool = False


@dataclass
class Documento:
    pmid: str
    texto: str
    trechos: dict = field(default_factory=dict)        # id -> Trecho
    relacoes: list = field(default_factory=list)       # (tipo, id do trecho 1, id do trecho 2)
    simultaneos: list = field(default_factory=list)    # grupos OVERLAP, como listas de ids de trechos
    atributos: dict = field(default_factory=dict)      # id do trecho -> {"POLARITY": "NEG", ...}


def ler_documento(pmid: str, texto: str, anotacoes: str) -> Documento:
    doc = Documento(pmid=pmid, texto=texto)
    eventos, relacoes, grupos, atributos = {}, [], [], []
    for linha in anotacoes.splitlines():
        if not linha.strip() or linha.startswith("#"):
            continue
        rotulo, resto = linha.split("\t", 1)
        cabeca = resto.split("\t", 1)[0]
        if rotulo.startswith("T"):
            tipo, posicoes = cabeca.split(" ", 1)
            fragmentos = [tuple(map(int, f.split())) for f in posicoes.split(";")]
            inicio, fim = fragmentos[0][0], fragmentos[-1][1]
            doc.trechos[rotulo] = Trecho(rotulo, tipo, inicio, fim, texto[inicio:fim], len(fragmentos) > 1)
        elif rotulo.startswith("E"):
            gatilho = cabeca.split(" ", 1)[0]          # "Tipo:T3"; argumentos extras não são usados
            eventos[rotulo] = gatilho.split(":", 1)[1]
        elif rotulo.startswith("R"):
            tipo, arg1, arg2 = cabeca.split(" ")[:3]
            relacoes.append((tipo, arg1.split(":", 1)[1], arg2.split(":", 1)[1]))
        elif rotulo.startswith("*"):
            partes = cabeca.split(" ")
            grupos.append((partes[0], partes[1:]))
        elif rotulo.startswith("A"):
            partes = cabeca.split(" ")
            atributos.append((partes[0], partes[1], partes[2] if len(partes) > 2 else "true"))

    def trecho_de(ref: str) -> str:
        return eventos.get(ref, ref)

    doc.relacoes = [(tipo, trecho_de(a), trecho_de(b)) for tipo, a, b in relacoes]
    doc.simultaneos = [[trecho_de(r) for r in refs] for tipo, refs in grupos if tipo == "OVERLAP"]
    for nome, ref, valor in atributos:
        doc.atributos.setdefault(trecho_de(ref), {})[nome] = valor
    return doc


def ler_zip(caminho: Path) -> list[Documento]:
    """Todos os documentos do zip, em ordem de PMID. Lê na memória, sem extrair nada para o disco."""
    with zipfile.ZipFile(caminho) as z:
        por_nome = {PurePosixPath(n).name: n for n in z.namelist() if not n.endswith("/")}
        pmids = sorted(nome[:-4] for nome in por_nome if nome.endswith(".ann"))
        return [ler_documento(p, z.read(por_nome[f"{p}.txt"]).decode("utf-8"),
                              z.read(por_nome[f"{p}.ann"]).decode("utf-8"))
                for p in pmids]
