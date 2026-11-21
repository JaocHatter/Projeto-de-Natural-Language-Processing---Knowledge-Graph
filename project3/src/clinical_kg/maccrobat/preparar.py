#!/usr/bin/env python3
"""E0: baixa o MACCROBAT2020, mapeia para o esquema do P1 e grava as tabelas.

    python -m clinical_kg.maccrobat.preparar                     # zip já em data/raw/maccrobat/
    python -m clinical_kg.maccrobat.preparar --baixar-multicare   # também cruza com o MultiCaRe

Grava:
    data/interim/maccrobat/maccrobat_entidades.csv   todos os trechos anotados, com a categoria do P1
    data/interim/maccrobat/maccrobat_relacoes.csv    todas as relações, com o papel de cada uma no P3
    data/interim/maccrobat/maccrobat_multicare.csv   documentos que também estão no MultiCaRe (por PMID)
    data/processed/resultados/maccrobat/maccrobat_resumo.csv    contagens usadas no README
"""

import argparse
import collections
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from clinical_kg.paths import (MACCROBAT_ENTIDADES, MACCROBAT_MULTICARE, MACCROBAT_RELACOES,
                               MACCROBAT_ZIP, MULTICARE_METADATA_PARQUET, PROJECT_ROOT,
                               RESULTADOS_MACCROBAT)

from .brat import ler_zip
from .mapeamento import CATEGORIA_P1, entidades_p1, relacoes_mapeadas

URL_MACCROBAT = "https://ndownloader.figshare.com/files/21489405"
MD5_MACCROBAT = "9d090f3e0719e77e7a33c38750548bb8"      # o publicado pelo figshare
ZENODO_MULTICARE = "https://zenodo.org/api/records/10079369/versions/latest"
RESUMO = RESULTADOS_MACCROBAT / "maccrobat_resumo.csv"


def md5(caminho: Path) -> str:
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(url, timeout=120) as resposta, open(temporario, "wb") as f:
        while bloco := resposta.read(1 << 20):
            f.write(bloco)
    temporario.replace(destino)


def garantir_zip(caminho: Path = MACCROBAT_ZIP) -> Path:
    if not caminho.exists():
        print(f"baixando o MACCROBAT2020 de {URL_MACCROBAT}")
        baixar(URL_MACCROBAT, caminho)
    if md5(caminho) != MD5_MACCROBAT:
        sys.exit(f"ERRO: {caminho} não tem o md5 publicado pelo figshare ({MD5_MACCROBAT}).")
    return caminho


def garantir_metadata_multicare(caminho: Path = MULTICARE_METADATA_PARQUET) -> Path:
    if not caminho.exists():
        with urllib.request.urlopen(ZENODO_MULTICARE, timeout=60) as r:
            registro = json.load(r)
        arquivo = next(f for f in registro["files"] if f["key"] == "metadata.parquet")
        print(f"baixando metadata.parquet do MultiCaRe {registro['metadata']['version']} "
              f"({arquivo['size'] / 2**20:.0f} MB)")
        baixar(arquivo["links"]["self"], caminho)
    return caminho


def sobreposicao_multicare(pmids: set[str], metadata_parquet: Path) -> list[dict]:
    """Artigos do MultiCaRe cujo PMID é o nome de um documento do MACCROBAT."""
    import pyarrow.parquet as pq

    saida = []
    for linha in pq.read_table(metadata_parquet).to_pylist():
        m = linha["article_metadata"]
        pmid = (m.get("pmid") or "").split(".")[0].strip()
        if pmid in pmids:
            saida.append({"pmid": pmid, "article_id": linha["article_id"], "ano": m.get("year"),
                          "licenca": m.get("license"), "case_amount": m.get("case_amount"),
                          "titulo": m.get("title")})
    return sorted(saida, key=lambda r: r["pmid"])


def gravar(caminho: Path, linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip", type=Path, default=MACCROBAT_ZIP)
    ap.add_argument("--multicare-metadata", type=Path, default=MULTICARE_METADATA_PARQUET)
    ap.add_argument("--baixar-multicare", action="store_true",
                    help="baixa metadata.parquet do MultiCaRe (20 MB, fora do git) para o cruzamento")
    args = ap.parse_args(argv)

    documentos = ler_zip(garantir_zip(args.zip))
    linhas_ent, linhas_rel = [], []
    resumo = collections.Counter()
    for doc in documentos:
        usados, descartes = entidades_p1(doc)
        ids_usados = {e["maccrobat_id"] for e in usados}
        resumo["sentencas"] += sum(1 for linha in doc.texto.splitlines() if linha.strip())
        resumo.update({f"descarte_{k}": v for k, v in descartes.items()})
        for t in doc.trechos.values():
            resumo[f"trecho_{t.tipo}"] += 1
            polaridade = doc.atributos.get(t.id, {}).get("POLARITY", "")
            if polaridade:
                resumo[f"polaridade_{polaridade}"] += 1
            linhas_ent.append({"pmid": doc.pmid, "id": t.id, "tipo": t.tipo, "inicio": t.inicio, "fim": t.fim,
                               "texto": t.texto, "descontinuo": int(t.descontinuo), "polaridade": polaridade,
                               "categoria_p1": CATEGORIA_P1.get(t.tipo, ""), "usado_no_p1": int(t.id in ids_usados)})
        for r in relacoes_mapeadas(doc):
            resumo[f"relacao_{r['tipo_maccrobat']}"] += 1
            if r["papel_p3"]:
                resumo[f"papel_{r['papel_p3']}"] += 1
            linhas_rel.append({**r, "args_usados_no_p1": int(r["arg1"] in ids_usados and r["arg2"] in ids_usados)})
        resumo["grupos_OVERLAP"] += len(doc.simultaneos)
    resumo["documentos"] = len(documentos)
    resumo["trechos"] = len(linhas_ent)
    resumo["entidades_p1"] = sum(e["usado_no_p1"] for e in linhas_ent)

    gravar(MACCROBAT_ENTIDADES, linhas_ent)
    gravar(MACCROBAT_RELACOES, linhas_rel)
    gravar(RESUMO, [{"metrica": k, "valor": v} for k, v in sorted(resumo.items())])

    print(f"{len(documentos)} documentos, {resumo['sentencas']} sentenças, {resumo['trechos']} trechos, "
          f"{len(linhas_rel)} relações")
    print(f"entidades no esquema do P1: {resumo['entidades_p1']} "
          f"(descartados: {resumo['descarte_sobreposto']} sobrepostos, {resumo['descarte_descontinuo']} descontínuos)")
    for papel in ("CAUSED_BY", "LOCATED_IN", "VALOR_DE_EXAME", "DOSE_DE_MEDICAMENTO", "TEMPORAL", "CORREFERENCIA"):
        print(f"  {papel:<22}{resumo[f'papel_{papel}']:>7}")

    if args.baixar_multicare:
        garantir_metadata_multicare(args.multicare_metadata)
    if args.multicare_metadata.exists():
        comuns = sobreposicao_multicare({d.pmid for d in documentos}, args.multicare_metadata)
        if comuns:
            gravar(MACCROBAT_MULTICARE, comuns)
        print(f"documentos também no MultiCaRe: {len(comuns)} de {len(documentos)}")
    else:
        print("(cruzamento com o MultiCaRe pulado: use --baixar-multicare)")
    for caminho in (MACCROBAT_ENTIDADES, MACCROBAT_RELACOES, RESUMO):
        print(f"  {caminho.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
