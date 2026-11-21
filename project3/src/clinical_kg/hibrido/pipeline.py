#!/usr/bin/env python3
"""Pipeline do Projeto 3 (as partes que não precisam de LLM).

    python -m clinical_kg.hibrido.pipeline externo    # P1 e P2 decidindo relações no MACCROBAT
    python -m clinical_kg.hibrido.pipeline etapas     # entidades, asserção e medidas do P1 no MACCROBAT
    python -m clinical_kg.hibrido.pipeline treinar    # variantes do P3 no gold, com bootstrap
    python -m clinical_kg.hibrido.pipeline extrair    # modelo final do P3 e o seu grafo nos 50 casos
    python -m clinical_kg.hibrido.pipeline figuras    # figuras do README e dos slides

externo
    Lê  : data/raw/maccrobat/MACCROBAT2020.zip, o gold e os casos do MultiCaRe
    Grava: data/processed/resultados/maccrobat/p3_externo_maccrobat.csv

etapas (precisa do gazetteer UMLS; ver CLINICAL_KG_GAZETTEER em paths.py)
    Grava: data/processed/resultados/maccrobat/p3_externo_{entidades,assercao,medidas}.csv

treinar
    Grava: data/processed/resultados/gold/p3_metricas_validacao.csv, p3_metricas_por_caso.csv,
           p3_metricas_por_relacao.csv, p3_metricas_bootstrap.csv e p3_sensibilidade_embeddings.csv
    As variantes com embeddings (P3-D) precisam de sentence-transformers na primeira execução;
    depois os vetores ficam em cache em data/interim/cache/embeddings/.

extrair
    Grava: data/processed/modelos/modelo_p3.json, data/processed/extracao/relacoes_p3.csv (esquema do
           relations.csv do P1) e data/processed/resultados/gold/p3_comparacao_arestas.csv (P1 x P2 x P3)

O protocolo é o do P2: validação deixando um caso do gold de fora e bootstrap pareado por caso,
e TODAS as linhas de base (P1, P2) são recalculadas na mesma execução, para que as linhas do P3
sejam comparáveis às do P2 na mesma máquina.
"""

import argparse
import collections
import csv
import json
import sys
import time

from clinical_kg.learning.modelo import ModeloAprendido, carregar_exemplos
from clinical_kg.learning.pipeline import (VARIANTES, bootstrap_pareado, chave_relacao, confronto,
                                           entidades_lf, gravar_csv, jaccard, ler_casos, media_dp,
                                           medir, prever_validando, prf, relacoes_p2,
                                           textos_dos_casos)
from clinical_kg.maccrobat.brat import ler_zip
from clinical_kg.maccrobat.mapeamento import TRANSFERIVEIS
from clinical_kg.maccrobat.preparar import garantir_zip
from clinical_kg.paths import (DEFAULT_CASES, DEFAULT_DB, DEFAULT_ENTITIES, DEFAULT_RELATIONS,
                               MODELO_P3_JSON, PROJECT_ROOT, RELACOES_ML, RELACOES_P3,
                               RESULTADOS_GOLD, RESULTADOS_MACCROBAT)

from .pares_maccrobat import decisao_p1, exemplos_maccrobat, medir_por_tipo
from .transferencia import (ModeloHibrido, balancear, contagem_por_rotulo, prever_validando_hibrido)

DESCRICAO = {
    "p1": "Projeto 1 (regras à mão)",
    "p2_estrutura": "P2: só tipos, distância e pontuação",
    "p2_completo": "P2 completo (estrutura + PMI/LLR + modelo de linguagem)",
    "p3d_estrutura": "P3-D: P2 estrutura + embeddings de sentença (empilhamento)",
    "p3d_completo": "P3-D: P2 completo + embeddings de sentença (empilhamento)",
    "p3g_estrutura": "P3-G: P2 estrutura + julgamentos do MACCROBAT (CAUSED_BY, LOCATED_IN)",
    "p3g_completo": "P3-G: P2 completo + julgamentos do MACCROBAT (CAUSED_BY, LOCATED_IN)",
    "p3gd_completo": "P3-G + P3-D: P2 completo + MACCROBAT + embeddings",
}
PROPORCAO_NEGATIVOS = 3.0
C_EMBEDDINGS = 1.0                 # fixado antes de olhar o gold (o padrão do scikit-learn)
SENSIBILIDADE_C = (0.1, 1.0, 10.0)
MODELO_P3 = MODELO_P3_JSON


def carregar_maccrobat():
    documentos = ler_zip(garantir_zip())
    exemplos, cobertura = exemplos_maccrobat(documentos)
    return documentos, exemplos, cobertura


def relativo(caminho):
    try:
        return caminho.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return caminho


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v


# --------------------------------------------------------------------------
# externo: decisão de relação fora do MultiCaRe
# --------------------------------------------------------------------------
def cmd_externo(args):
    _, mac, cobertura = carregar_maccrobat()
    gold = carregar_exemplos()
    textos = textos_dos_casos()
    print(f"MACCROBAT: {len(mac)} pares candidatos; rótulos {dict(contagem_por_rotulo(mac))}")

    decisores = {"p1": [(e, decisao_p1(e)) for e in mac]}
    for nome in ("estrutura", "completo"):
        modelo = ModeloAprendido(**VARIANTES[nome]).ajustar(gold, list(textos.values()))
        decisores[f"p2_{nome}"] = [(e, modelo.prever_exemplo(e)) for e in mac]

    linhas = []
    print(f"\n{'decisor':<15}{'tipo':<12}{'prec':>7}{'cob':>7}{'F1':>7}{'teto':>7}{'cob. total':>12}")
    for nome, previsoes in decisores.items():
        for m in medir_por_tipo(previsoes, cobertura):
            print(f"{nome:<15}{m['tipo']:<12}{m['precisao']:>7.3f}{m['cobertura']:>7.3f}{m['f1']:>7.3f}"
                  f"{m['teto_do_gerador']:>7.3f}{m['cobertura_total']:>12.3f}")
            linhas.append((nome, m["tipo"], m["vp"], m["fp"], m["fn"], f"{m['precisao']:.4f}",
                           f"{m['cobertura']:.4f}", f"{m['f1']:.4f}", m["anotadas"],
                           f"{m['teto_do_gerador']:.4f}", f"{m['cobertura_total']:.4f}"))
    saida = RESULTADOS_MACCROBAT / "p3_externo_maccrobat.csv"
    gravar_csv(saida, ["decisor", "tipo", "vp", "fp", "fn", "precisao", "cobertura", "f1",
                       "anotadas", "teto_do_gerador", "cobertura_total"], linhas)
    print(f"\n{relativo(saida)}")


# --------------------------------------------------------------------------
# etapas: entidades, asserção e medidas do P1 no MACCROBAT
# --------------------------------------------------------------------------
def gazetteer_disponivel(caminho=DEFAULT_DB, minimo_termos: int = 100_000) -> bool:
    """O gazetteer completo (~1,1 milhão de termos), e não a amostra de teste."""
    import sqlite3
    if not caminho.exists():
        return False
    try:
        with sqlite3.connect(caminho) as con:
            return con.execute("SELECT COUNT(*) FROM terms").fetchone()[0] >= minimo_termos
    except sqlite3.Error:
        return False


def extrator_gazetteer(caminho=DEFAULT_DB):
    """`extrair(texto)` com cache, usando o reconhecedor de entidades do P1."""
    import sqlite3
    from clinical_kg.extraction.entities import DEFAULT_MAX_N, extract

    con = sqlite3.connect(caminho)
    cache = {}

    def extrair(texto):
        if texto not in cache:
            cache[texto] = extract(texto, con, DEFAULT_MAX_N)
        return cache[texto]
    return extrair


def cmd_etapas(args):
    from clinical_kg.maccrobat.avaliacao import avaliar_assercao, avaliar_entidades, avaliar_medidas

    documentos = ler_zip(garantir_zip())
    extrair = extrator_gazetteer() if gazetteer_disponivel() else None
    if extrair is None:
        print(f"(gazetteer UMLS completo não encontrado em {DEFAULT_DB}: entidades puladas; "
              "aponte CLINICAL_KG_GAZETTEER para ele)")
    else:
        linhas = avaliar_entidades(documentos, extrair)
        print(f"{'modo':<20}{'grupo':<17}{'gold':>7}{'prev':>7}{'prec':>7}{'cob':>7}{'F1':>7}")
        for l in linhas:
            print(f"{l['modo']:<20}{l['grupo']:<17}{l['gold']:>7}{l['previstas']:>7}"
                  f"{l['precisao']:>7.3f}{l['cobertura']:>7.3f}{l['f1']:>7.3f}")
        gravar_csv(RESULTADOS_MACCROBAT / "p3_externo_entidades.csv", list(linhas[0]),
                   [[fmt(v) for v in l.values()] for l in linhas])

    assercao, matriz = avaliar_assercao(documentos)
    print("\nAsserção (entidades do gold; p1 = escopo do P1, p3 = + propagação pela coordenação):")
    for l in assercao:
        print(f"  {l['decisor']}  {l['classe']:<9} anotadas {l['anotadas']:>5}  prec {l['precisao']:.3f}  "
              f"cob {l['cobertura']:.3f}  F1 {l['f1']:.3f}")
    gravar_csv(RESULTADOS_MACCROBAT / "p3_externo_assercao.csv", list(assercao[0]),
               [[fmt(v) for v in l.values()] for l in assercao])
    gravar_csv(RESULTADOS_MACCROBAT / "p3_externo_assercao_matriz.csv", ["real", "previsto", "n"],
               [(r, p, n) for (r, p), n in sorted(matriz.items())])

    medidas = avaliar_medidas(documentos, extrair)
    print("\nMedidas (valores numéricos):")
    for l in medidas:
        print(f"  {l['vinculo']:<21} numéricas {l['numericas']:>5}  detecção {l['deteccao']:.3f}  "
              f"vínculo (entidades gold) {l['acerto_vinculo_gold']:.3f}  "
              f"vínculo (gazetteer) {fmt(l['acerto_vinculo_gazetteer'])}")
    gravar_csv(RESULTADOS_MACCROBAT / "p3_externo_medidas.csv", list(medidas[0]),
               [[fmt(v) for v in l.values()] for l in medidas])
    print(f"\n{relativo(RESULTADOS_MACCROBAT)}/p3_externo_{{entidades,assercao,medidas}}.csv")


# --------------------------------------------------------------------------
# treinar: as variantes do P3 no gold, no protocolo do P2
# --------------------------------------------------------------------------
def preparar_embeddings(gold, textos, externos):
    """Vetores do gold e dos pares do MACCROBAT usados no treino; None se não houver como calcular."""
    from .embeddings import Codificador, vetores_dos_exemplos
    try:
        codificador = Codificador()
        vetores = vetores_dos_exemplos(gold, textos, codificador)
        if externos:
            docs = {f"MACCROBAT:{d.pmid}": d.texto for d in ler_zip(garantir_zip())}
            vetores.update(vetores_dos_exemplos(externos, docs, codificador))
        return vetores
    except ImportError as erro:
        print(f"(embeddings pulados: {erro})")
        return None


def cmd_treinar(args):
    inicio = time.time()
    gold = carregar_exemplos()
    textos = textos_dos_casos()
    _, mac, _ = carregar_maccrobat()
    externos = balancear(mac, PROPORCAO_NEGATIVOS)
    print(f"gold: {len(gold)} pares em {len({e.caso for e in gold})} casos; "
          f"MACCROBAT (balanceado 1:{PROPORCAO_NEGATIVOS:g}): {dict(contagem_por_rotulo(externos))}")

    previsoes = {
        "p1": prever_validando(gold, textos, "p1"),
        "p2_estrutura": prever_validando(gold, textos, "estrutura"),
        "p2_completo": prever_validando(gold, textos, "completo"),
    }
    vetores = None if args.sem_embeddings else preparar_embeddings(gold, textos, externos)
    sensibilidade = []
    if vetores is not None:
        from .embeddings import FonteEmbeddings

        def fontes(C=C_EMBEDDINGS):
            return lambda: [FonteEmbeddings(vetores, C)]
        for nome in ("estrutura", "completo"):
            previsoes[f"p3d_{nome}"] = prever_validando_hibrido(
                gold, textos, VARIANTES[nome], fabricar_fontes=fontes())
        # Sensibilidade ao C do nível 0, só para relatar (o C usado é o fixado acima).
        base = medir(previsoes["p2_completo"])
        for C in SENSIBILIDADE_C:
            m = medir(prever_validando_hibrido(gold, textos, VARIANTES["completo"], fabricar_fontes=fontes(C)))
            r = bootstrap_pareado(m["contagem_por_caso"], base["contagem_por_caso"])
            sensibilidade.append((C, f"{m['f1']:.4f}", f"{r['diferenca']:+.4f}",
                                  f"{r['ic_inf']:+.4f}", f"{r['ic_sup']:+.4f}"))
    previsoes["p3g_estrutura"] = prever_validando_hibrido(gold, textos, VARIANTES["estrutura"], externos)
    previsoes["p3g_completo"] = prever_validando_hibrido(gold, textos, VARIANTES["completo"], externos)
    if vetores is not None:
        previsoes["p3gd_completo"] = prever_validando_hibrido(
            gold, textos, VARIANTES["completo"], externos, fabricar_fontes=fontes())
    resultados = {nome: medir(p) for nome, p in previsoes.items()}

    print(f"\n{'variante':<16}{'prec':>7}{'cob':>7}{'F1':>7}{'tipo':>7}   F1 por caso     vs P2 (V-E-D)")
    linhas, linhas_caso = [], []
    for nome, m in resultados.items():
        media, dp = media_dp(list(m["por_caso"].values()))
        vs_p2 = "-" if nome.startswith(("p1", "p2")) else "{}-{}-{}".format(
            *confronto(m["por_caso"], resultados["p2_completo"]["por_caso"]))
        print(f"{nome:<16}{m['p']:>7.3f}{m['r']:>7.3f}{m['f1']:>7.3f}{m['tipo']:>7.3f}"
              f"   {media:.3f} ± {dp:.3f}   {vs_p2}")
        linhas.append((nome, DESCRICAO[nome], f"{m['p']:.4f}", f"{m['r']:.4f}", f"{m['f1']:.4f}",
                       f"{m['tipo']:.4f}", f"{media:.4f}", f"{dp:.4f}", vs_p2))
        linhas_caso += [(nome, c, f"{f:.4f}") for c, f in m["por_caso"].items()]
    gravar_csv(RESULTADOS_GOLD / "p3_metricas_validacao.csv",
               ["variante", "descricao", "precisao", "cobertura", "f1", "tipo_acertado",
                "f1_caso_media", "f1_caso_dp", "casos_vs_p2_V-E-D"], linhas)
    gravar_csv(RESULTADOS_GOLD / "p3_metricas_por_caso.csv", ["variante", "caso", "f1"], linhas_caso)

    linhas_rel = []
    print("\nPor relação (vp/fp/fn e F1), as que o MACCROBAT transfere:")
    for nome, m in resultados.items():
        for rel, (vp, fp, fn) in m["por_relacao"].items():
            p, r, f1 = prf(vp, fp, fn)
            linhas_rel.append((nome, rel, vp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"))
            if rel in TRANSFERIVEIS:
                print(f"  {nome:<16}{rel:<12}{vp:>4}/{fp}/{fn}   F1 {f1:.3f}")
    gravar_csv(RESULTADOS_GOLD / "p3_metricas_por_relacao.csv",
               ["variante", "relacao", "vp", "fp", "fn", "precisao", "cobertura", "f1"], linhas_rel)

    contrastes = [("p3g_completo", "p2_completo"), ("p3g_estrutura", "p2_estrutura"),
                  ("p3d_completo", "p2_completo"), ("p3d_estrutura", "p2_estrutura"),
                  ("p3gd_completo", "p2_completo"), ("p3gd_completo", "p3g_completo"),
                  ("p3g_completo", "p1"), ("p2_completo", "p1")]
    contrastes = [(a, b) for a, b in contrastes if a in resultados and b in resultados]
    linhas_boot = []
    print("\nDiferença de F1 (A − B), bootstrap por caso, IC 95%:")
    for a, b in contrastes:
        r = bootstrap_pareado(resultados[a]["contagem_por_caso"], resultados[b]["contagem_por_caso"])
        linhas_boot.append((a, b, f"{r['diferenca']:+.4f}", f"{r['ic_inf']:+.4f}",
                            f"{r['ic_sup']:+.4f}", f"{r['prob_positiva']:.3f}"))
        marca = "distinguível de zero" if r["ic_inf"] > 0 or r["ic_sup"] < 0 else "inclui o zero"
        print(f"  {a:<14} − {b:<13} {r['diferenca']:+.3f}  [{r['ic_inf']:+.3f}, {r['ic_sup']:+.3f}]  {marca}")
    gravar_csv(RESULTADOS_GOLD / "p3_metricas_bootstrap.csv",
               ["a", "b", "diferenca_f1", "ic95_inf", "ic95_sup", "prob_a_maior"], linhas_boot)
    if sensibilidade:
        gravar_csv(RESULTADOS_GOLD / "p3_sensibilidade_embeddings.csv",
                   ["C_nivel0", "f1_p3d_completo", "diferenca_vs_p2_completo", "ic95_inf", "ic95_sup"],
                   sensibilidade)
    print(f"\nPronto em {time.time() - inicio:.0f}s; tabelas em {relativo(RESULTADOS_GOLD)}/p3_metricas_*.csv")


# --------------------------------------------------------------------------
# extrair: o modelo final do P3 decide todos os candidatos dos 50 casos
# --------------------------------------------------------------------------
def treinar_modelo_final():
    """P3-G completo com todo o gold, o texto de todos os casos e o MACCROBAT balanceado."""
    gold = carregar_exemplos()
    textos = textos_dos_casos()
    _, mac, _ = carregar_maccrobat()
    externos = balancear(mac, PROPORCAO_NEGATIVOS)
    return ModeloHibrido(**VARIANTES["completo"]).ajustar(gold, list(textos.values()), externos)


def extrair_relacoes(modelo, casos, textos, entidades) -> list[dict]:
    from clinical_kg.relations.extract import DEFAULT_FREQ, load_frequencies
    freq = load_frequencies(DEFAULT_FREQ)
    linhas = []
    for caso in casos:
        novas, _ = relacoes_p2(modelo, caso, textos[caso["case_id"]],
                               entidades.get(caso["case_id"], []), freq)
        for l in novas:
            l["rule_path"] = "p3g.regressao_logistica"
        linhas += novas
    return linhas


def comparar_tres(l1, l2, l3) -> list[dict]:
    """Relações por tipo nos três métodos e o quanto coincidem (mesmo par e tipo)."""
    k1, k2, k3 = ({chave_relacao(l) for l in ls} for ls in (l1, l2, l3))
    saida = []
    for t in [*sorted({k[3] for k in k1 | k2 | k3}), "TOTAL"]:
        a, b, c = ({k for k in ks if t == "TOTAL" or k[3] == t} for ks in (k1, k2, k3))
        saida.append({"relacao": t, "p1": len(a), "p2": len(b), "p3": len(c),
                      "jaccard_p1_p2": round(jaccard(a, b), 3), "jaccard_p2_p3": round(jaccard(b, c), 3),
                      "jaccard_p1_p3": round(jaccard(a, c), 3),
                      "so_p3": len(c - a - b), "p3_e_nao_p2": len(c - b)})
    return saida


def cmd_extrair(args):
    from clinical_kg.learning.pipeline import ler_relacoes
    from clinical_kg.relations.extract import CSV_COLUMNS

    modelo = treinar_modelo_final()
    MODELO_P3.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELO_P3, "w", encoding="utf-8") as f:
        json.dump(modelo.para_dict(), f, ensure_ascii=False, separators=(",", ":"))

    casos = ler_casos(DEFAULT_CASES)
    textos = textos_dos_casos(DEFAULT_CASES)
    entidades = entidades_lf(DEFAULT_ENTITIES)
    linhas = extrair_relacoes(modelo, casos, textos, entidades)
    with open(RELACOES_P3, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(linhas)
    print(f"{len(casos)} casos, {len(linhas)} relações; {relativo(RELACOES_P3)}, {relativo(MODELO_P3)}")

    tabela = comparar_tres(ler_relacoes(DEFAULT_RELATIONS), ler_relacoes(RELACOES_ML), linhas)
    gravar_csv(RESULTADOS_GOLD / "p3_comparacao_arestas.csv", list(tabela[0]), [list(l.values()) for l in tabela])
    print(f"\n{'relação':<18}{'P1':>6}{'P2':>6}{'P3':>6}{'J(P2,P3)':>10}{'J(P1,P3)':>10}")
    for l in tabela:
        print(f"  {l['relacao']:<16}{l['p1']:>6}{l['p2']:>6}{l['p3']:>6}"
              f"{l['jaccard_p2_p3']:>10.3f}{l['jaccard_p1_p3']:>10.3f}")


# --------------------------------------------------------------------------
# figuras
# --------------------------------------------------------------------------
def cmd_figuras(args):
    from .figuras import desenhar_todas
    for caminho in desenhar_todas():
        print(relativo(caminho))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):      # console do Windows (cp1252) não tem "−" nem "±"
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="comando", required=True)
    comandos = (("externo", "P1 e P2 decidindo relações no MACCROBAT", cmd_externo),
                ("etapas", "entidades, asserção e medidas do P1 no MACCROBAT", cmd_etapas),
                ("treinar", "variantes do P3 no gold, com bootstrap", cmd_treinar),
                ("extrair", "modelo final do P3 e o seu grafo nos 50 casos", cmd_extrair),
                ("figuras", "figuras do README e dos slides", cmd_figuras))
    for nome, ajuda, func in comandos:
        s = sub.add_parser(nome, help=ajuda)
        if nome == "treinar":
            s.add_argument("--sem-embeddings", action="store_true",
                           help="pula P3-D (não precisa de sentence-transformers)")
        s.set_defaults(func=func)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
