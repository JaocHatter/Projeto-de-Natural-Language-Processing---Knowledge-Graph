#!/usr/bin/env python3
"""O corpus da aplicação: casos clínicos passados pelo P1 e pelo modelo final do P3.

    python -m clinical_kg.paciente.corpus construir --nome multicare50
    python -m clinical_kg.paciente.corpus construir --nome digestivo --recorte digestivo --maximo 1500
    python -m clinical_kg.paciente.corpus construir --nome maccrobat --fonte maccrobat

Cada corpus é uma pasta data/processed/app/<nome>/ com quatro CSV no esquema do P1:

    casos.csv       case_id, article_id, age, gender (o texto fica de fora: só as evidências)
    entidades.csv   as entidades do gazetteer, com `assertion_status` (escopo de negação do P1)
    medicoes.csv    valores com unidade, ligados à entidade mais próxima (etapa 4 do P1)
    relacoes.csv    as relações decididas pelo modelo final do P3 (P3-G)

Construir precisa do gazetteer UMLS; carregar não. Assim quem não tem a licença do UMLS ainda
roda a aplicação com os CSV já construídos.

Recortes do MultiCaRe completo (cases.parquet + metadata.parquet, versão 3.0.1) são escolhidos
pelos termos MeSH principais do artigo; ver RECORTES.
"""

import argparse
import collections
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

from clinical_kg.paths import (DEFAULT_CASES, MULTICARE_CASES_PARQUET, MULTICARE_METADATA_PARQUET,
                               PASTA_APP, PROJECT_ROOT)


# Prefixos de termos MeSH principais (antes da barra) que definem cada recorte.
RECORTES = {
    "digestivo": ("Pancrea", "Stomach", "Gastr", "Liver", "Hepat", "Biliary", "Bile Duct",
                  "Cholecyst", "Cholang", "Gallbladder", "Duoden", "Intestin", "Colon", "Colonic",
                  "Rectal", "Esophag", "Appendic", "Peptic Ulcer", "Jejun", "Ileal", "Ileus",
                  "Peritonitis", "Crohn", "Colitis", "Abdominal Pain"),
}
COLUNAS_ENTIDADES = ["case_id", "start", "end", "surface_text", "cui", "tui", "entity_type",
                     "preferred_term", "assertion_status"]
COLUNAS_MEDICOES = ["case_id", "start", "end", "surface_text", "value", "unit", "unit_category",
                    "linked_entity_cui", "linked_entity_type", "link_method"]
COLUNAS_RELACOES = ["case_id", "relation", "assertion_status", "head_cui", "head_entity_type",
                    "head_text", "tail_cui", "tail_entity_type", "tail_text", "score",
                    "sent_start", "sent_end"]


# --------------------------------------------------------------------------
# Construção (precisa do gazetteer)
# --------------------------------------------------------------------------
def assercao_das_entidades(texto: str, entidades: list[dict]) -> list[str]:
    """O estado de asserção (escopo do P1) de cada entidade, na ordem da lista."""
    from clinical_kg.relations.core.text_layer import segment
    from clinical_kg.relations.extract import assertion_for, assertion_states

    estado = ["affirmed"] * len(entidades)
    for sent in segment(texto):
        if not sent["tokens"]:
            continue
        dentro = [i for i, e in enumerate(entidades)
                  if e["start"] >= sent["start"] and e["end"] <= sent["end"]]
        if not dentro:
            continue
        tokens = [s[0] for s in sent["tokens"]]
        estados = assertion_states(tokens, sent["tokens"], sent["clauses"], texto)
        for i in dentro:
            estado[i] = assertion_for(estados, sent["tokens"], entidades[i]["start"], entidades[i]["end"])
    return estado


def casos_do_recorte(nome: str, maximo: int, semente: int = 0) -> list[dict]:
    """Casos do MultiCaRe completo cujo artigo tem um termo MeSH principal do recorte."""
    import random

    import pyarrow.parquet as pq

    prefixos = RECORTES[nome]
    artigos = set()
    for linha in pq.read_table(MULTICARE_METADATA_PARQUET).to_pylist():
        m = linha["article_metadata"]
        # Só ~11 mil dos 76 mil artigos têm termos principais; sem eles, vale a lista completa.
        termos = m.get("major_mesh_terms") or m.get("mesh_terms") or []
        if any(t.split(" / ")[0].startswith(prefixos) for t in termos):
            artigos.add(linha["article_id"])
    casos = []
    for linha in pq.read_table(MULTICARE_CASES_PARQUET).to_pylist():   # casos aninhados por artigo
        if linha["article_id"] in artigos:
            casos += [{**c, "article_id": linha["article_id"]} for c in linha["cases"] or []
                      if c.get("case_text")]
    casos.sort(key=lambda c: c["case_id"])
    random.Random(semente).shuffle(casos)
    return [{"case_id": c["case_id"], "article_id": c["article_id"], "age": c.get("age") or "",
             "gender": c.get("gender") or "", "case_text": c["case_text"]} for c in casos[:maximo]]


def casos_multicare50() -> list[dict]:
    with open(DEFAULT_CASES, encoding="utf-8") as f:
        return list(csv.DictReader(f))


SEXO_MACCROBAT = {"man": "Male", "male": "Male", "boy": "Male", "gentleman": "Male", "men": "Male",
                  "woman": "Female", "female": "Female", "girl": "Female", "lady": "Female",
                  "women": "Female", "mother": "Female", "father": "Male"}


def idade_maccrobat(texto: str) -> str:
    """"28-year-old" -> "28"; "6-month-old", "3 weeks" -> "0" (menos de um ano); sem número -> ""."""
    import re
    m = re.search(r"(\d+)\s*-?\s*(year|yr|month|week|day)", texto.lower())
    if not m:
        return ""
    return m.group(1) if m.group(2) in ("year", "yr") else "0"


def sexo_maccrobat(texto: str) -> str:
    return SEXO_MACCROBAT.get(texto.lower().strip(" .,"), "")


def casos_maccrobat(documentos=None) -> list[dict]:
    """Os 200 relatos do MACCROBAT2020 como casos da aplicação.

    Idade e sexo saem dos trechos que os anotadores marcaram como `Age` e `Sex` (o primeiro de
    cada), e não de uma regra sobre o texto. O case_id leva o prefixo MACCROBAT_ para que a
    proveniência de cada fato da ficha diga de que fonte o paciente real veio.
    """
    if documentos is None:
        from clinical_kg.maccrobat.brat import ler_zip
        from clinical_kg.maccrobat.preparar import garantir_zip
        documentos = ler_zip(garantir_zip())
    casos = []
    for doc in documentos:
        trechos = sorted(doc.trechos.values(), key=lambda t: t.inicio)
        idade = next((idade_maccrobat(t.texto) for t in trechos if t.tipo == "Age" and idade_maccrobat(t.texto)), "")
        sexo = next((sexo_maccrobat(t.texto) for t in trechos if t.tipo == "Sex" and sexo_maccrobat(t.texto)), "")
        casos.append({"case_id": f"MACCROBAT_{doc.pmid}", "article_id": doc.pmid, "age": idade,
                      "gender": sexo, "case_text": doc.texto})
    return casos


def analisar_caso(caso: dict, con, modelo, freq) -> tuple[list, list, list]:
    """Entidades, medições e relações de um caso, nos esquemas de COLUNAS_*."""
    from clinical_kg.extraction.measurements import analyze_case as medidas_do_caso
    from clinical_kg.learning.pipeline import relacoes_p2

    texto = caso["case_text"]
    entidades, medicoes = medidas_do_caso(texto, con)
    estados = assercao_das_entidades(texto, entidades)
    ents = [{"case_id": caso["case_id"], "start": e["start"], "end": e["end"],
             "surface_text": e["surface_text"].replace("\n", " "), "cui": e["cui"], "tui": e["tui"],
             "entity_type": e["entity_type"], "preferred_term": e["preferred_term"],
             "assertion_status": s} for e, s in zip(entidades, estados)]
    meds = [{"case_id": caso["case_id"], **{k: m.get(k, "") for k in COLUNAS_MEDICOES[1:]}}
            for m in medicoes]
    para_relacoes = [{**e, "case_id": caso["case_id"]} for e in entidades]
    rels, _ = relacoes_p2(modelo, caso, texto, para_relacoes, freq)
    rels = [{k: r.get(k, "") for k in COLUNAS_RELACOES} for r in rels]
    return ents, meds, rels


def gravar(caminho: Path, colunas: list[str], linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colunas, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)


def construir(nome: str, casos: list[dict], progresso: bool = True) -> Path:
    import sqlite3
    import time

    from clinical_kg.hibrido.pipeline import gazetteer_disponivel, treinar_modelo_final
    from clinical_kg.paths import DEFAULT_DB
    from clinical_kg.relations.extract import DEFAULT_FREQ, load_frequencies

    if not gazetteer_disponivel():
        sys.exit(f"ERRO: o gazetteer UMLS completo não está em {DEFAULT_DB} "
                 "(aponte CLINICAL_KG_GAZETTEER para ele)")
    con = sqlite3.connect(DEFAULT_DB)
    modelo = treinar_modelo_final()
    freq = load_frequencies(DEFAULT_FREQ)
    entidades, medicoes, relacoes = [], [], []
    inicio = time.time()
    for i, caso in enumerate(casos, 1):
        e, m, r = analisar_caso(caso, con, modelo, freq)
        entidades += e
        medicoes += m
        relacoes += r
        if progresso and (i % 100 == 0 or i == len(casos)):
            print(f"  {i}/{len(casos)} casos ({time.time() - inicio:.0f}s)", flush=True)
    pasta = PASTA_APP / nome
    gravar(pasta / "casos.csv", ["case_id", "article_id", "age", "gender"], casos)
    gravar(pasta / "entidades.csv", COLUNAS_ENTIDADES, entidades)
    gravar(pasta / "medicoes.csv", COLUNAS_MEDICOES, medicoes)
    gravar(pasta / "relacoes.csv", COLUNAS_RELACOES, relacoes)
    print(f"{len(casos)} casos, {len(entidades)} entidades, {len(medicoes)} medições, "
          f"{len(relacoes)} relações -> {pasta.relative_to(PROJECT_ROOT)}")
    return pasta


# --------------------------------------------------------------------------
# Carga (não precisa do gazetteer)
# --------------------------------------------------------------------------
@dataclass
class Caso:
    case_id: str
    idade: str = ""
    sexo: str = ""
    entidades: list = field(default_factory=list)
    medicoes: list = field(default_factory=list)
    relacoes: list = field(default_factory=list)


@dataclass
class Corpus:
    nome: str
    casos: dict                          # case_id -> Caso
    nomes: dict                          # cui -> nome mais frequente no texto
    categorias: dict                     # cui -> categoria mais frequente (com a correção de exames)
    tuis: dict = field(default_factory=dict)


def limpar_nome(nome: str) -> str:
    """"a hemoglobin level" -> "hemoglobin level"; "cholangiopancreatography (mrcp" -> sem o parêntese aberto."""
    nome = " ".join(nome.split())
    for artigo in ("a ", "an ", "the "):
        if nome.startswith(artigo):
            nome = nome[len(artigo):]
    if nome.count("(") > nome.count(")"):
        nome = nome[:nome.rfind("(")].strip()
    return nome.strip(" ,;:")


def exames_de_laboratorio(unidades: dict, minimo: int = 3, fracao: float = 0.5) -> set[str]:
    """CUIs que o gazetteer tipa como tratamento (substância, T121) mas o texto usa como exame.

    "lipase 1200 U/L", "albumin 2.1 g/dL": a substância aparece com valores de concentração de
    laboratório, não com doses ("prednisone 40 mg" tem unidade de massa e continua tratamento).
    """
    saida = set()
    for cui, c in unidades.items():
        total = sum(c.values())
        if total >= minimo and c["lab_concentration"] / total >= fracao:
            saida.add(cui)
    return saida


def _ler(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def corpora_disponiveis() -> list[str]:
    return sorted(p.name for p in PASTA_APP.glob("*") if (p / "entidades.csv").exists())


def carregar(nomes) -> Corpus:
    """Um ou mais corpora da aplicação, unidos (os case_id não se repetem entre eles)."""
    if isinstance(nomes, str):
        nomes = [nomes]
    casos: dict[str, Caso] = {}
    superficies = collections.defaultdict(collections.Counter)
    tipos = collections.defaultdict(collections.Counter)
    unidades = collections.defaultdict(collections.Counter)
    tuis = {}
    for nome in nomes:
        pasta = PASTA_APP / nome
        for c in _ler(pasta / "casos.csv"):
            idade = c.get("age", "")
            casos[c["case_id"]] = Caso(c["case_id"], idade.split(".")[0] if idade else "",
                                       c.get("gender", ""))
        for e in _ler(pasta / "entidades.csv"):
            if e["case_id"] in casos and e["cui"]:
                casos[e["case_id"]].entidades.append(e)
                superficies[e["cui"]][e["surface_text"].lower()] += 1
                tipos[e["cui"]][e["entity_type"]] += 1
                tuis[e["cui"]] = e["tui"]
        for m in _ler(pasta / "medicoes.csv"):
            if m["case_id"] in casos:
                casos[m["case_id"]].medicoes.append(m)
                if m["linked_entity_cui"]:
                    unidades[m["linked_entity_cui"]][m["unit_category"]] += 1
        for r in _ler(pasta / "relacoes.csv"):
            if r["case_id"] in casos:
                casos[r["case_id"]].relacoes.append(r)
    nomes_cui = {cui: limpar_nome(c.most_common(1)[0][0]) for cui, c in superficies.items()}
    categorias = {cui: c.most_common(1)[0][0] for cui, c in tipos.items()}
    for cui in exames_de_laboratorio(unidades):
        if categorias.get(cui) == "Treatment":
            categorias[cui] = "Exam"
    return Corpus("+".join(nomes), casos, nomes_cui, categorias, tuis)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="comando", required=True)
    c = sub.add_parser("construir", help="passa os casos pelo P1 e pelo P3 e grava o corpus")
    c.add_argument("--nome", required=True)
    c.add_argument("--fonte", choices=("multicare50", "recorte", "maccrobat"),
                   help="multicare50 (os 50 casos do P1), recorte (do MultiCaRe completo, ver --recorte) ou "
                        "maccrobat (os 200 relatos do MACCROBAT2020)")
    c.add_argument("--recorte", choices=sorted(RECORTES), help="recorte do MultiCaRe completo")
    c.add_argument("--maximo", type=int, default=1500)
    args = ap.parse_args(argv)
    fonte = args.fonte or ("recorte" if args.recorte else "multicare50")
    if fonte == "recorte":
        if not args.recorte:
            ap.error("--fonte recorte precisa de --recorte")
        casos = casos_do_recorte(args.recorte, args.maximo)
    elif fonte == "maccrobat":
        casos = casos_maccrobat()
    else:
        casos = casos_multicare50()
    print(f"{len(casos)} casos")
    construir(args.nome, casos)


if __name__ == "__main__":
    main()
