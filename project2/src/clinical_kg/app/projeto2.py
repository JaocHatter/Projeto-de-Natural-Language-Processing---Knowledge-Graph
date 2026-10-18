#!/usr/bin/env python3
"""Visualizador do Projeto 2: o grafo do P1 (regras), o do P2 (aprendido) e a comparação.

Casos do corpus: usa as entidades e relações já extraídas em `data/processed/` (rode
`bash run_all.sh` antes para gerar os arquivos); não precisa do gazetteer.

Caso novo: o texto colado ou enviado passa pelo gazetteer UMLS (entidades e medições) e depois
pelos dois métodos, ao vivo. Isso exige `data/interim/gazetteer.db`, ou a variável de ambiente
CLINICAL_KG_GAZETTEER apontando para ele.

Executar:
    streamlit run src/clinical_kg/app/projeto2.py

Métodos do seletor:
    P1  regras à mão (CRF com pesos definidos pela equipe), `relations.csv`
    P2  regressão logística + n-gramas/PMI + modelo de linguagem, `relacoes_ml.csv`
    Comparação  as duas, com a aresta colorida pelo método que a produziu
"""

import csv
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from clinical_kg.app.main import PALETTE, _colour_css, analyze, file_revision, render_legend
from clinical_kg.corpus.patients import new_patient
from clinical_kg.extraction.entities import DEFAULT_MAX_N
from clinical_kg.graph.export import extract_relations, load_umls_relations
from clinical_kg.graph.ui import graph_panel
from clinical_kg.learning.modelo import ModeloAprendido
from clinical_kg.learning.pipeline import (MODELO_JSON, RELACOES_ML, SAIDA, TABELA_NGRAMAS,
                                           TABELA_PMI, chave_relacao, combinar_relacoes,
                                           entidades_lf, relacoes_p2)
from clinical_kg.paths import (DEFAULT_CASES, DEFAULT_DB, DEFAULT_ENTITIES, DEFAULT_MEASUREMENTS,
                               DEFAULT_METADATA, DEFAULT_RELATIONS, DEFAULT_TOKEN_FREQ)
from clinical_kg.relations.core import features as regras_p1
from clinical_kg.relations.extract import load_frequencies

METODOS = {
    "P1 · regras à mão": "p1",
    "P2 · aprendido": "p2",
    "Comparação P1 × P2": "comparacao",
}
ARQUIVOS_NECESSARIOS = [DEFAULT_CASES, DEFAULT_ENTITIES, DEFAULT_RELATIONS, RELACOES_ML]


def ler_csv(caminho: Path, **opcoes) -> list[dict]:
    with open(caminho, encoding="utf-8", **opcoes) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# Dados em cache
# --------------------------------------------------------------------------
@st.cache_data
def carregar_dados(revisao: tuple):
    # Os casos e as medições são lidos com quebra de linha LF (os deslocamentos
    # das entidades indexam esse texto); as relações, como o P1 as grava.
    casos = ler_csv(DEFAULT_CASES)
    por_caso = lambda linhas: {c["case_id"]: [l for l in linhas if l["case_id"] == c["case_id"]]
                               for c in casos}
    medicoes = ler_csv(DEFAULT_MEASUREMENTS) if DEFAULT_MEASUREMENTS.exists() else []
    return {
        "casos": casos,
        "entidades": entidades_lf(DEFAULT_ENTITIES),
        "medicoes": por_caso(medicoes),
        "p1": por_caso(ler_csv(DEFAULT_RELATIONS, newline="")),
        "p2": por_caso(ler_csv(RELACOES_ML, newline="")),
        "artigos": ({r["article_id"]: r for r in ler_csv(DEFAULT_METADATA, newline="")}
                    if DEFAULT_METADATA.exists() else {}),
    }


@st.cache_data
def ontologia(revisao: tuple):
    return load_umls_relations()


@st.cache_resource
def carregar_modelo(revisao: tuple) -> ModeloAprendido:
    with open(MODELO_JSON, encoding="utf-8") as f:
        return ModeloAprendido.de_dict(json.load(f))


@st.cache_data
def analisar_caso_novo(texto: str, revisao_modelo: tuple, revisao_gazetteer: tuple) -> dict:
    """Um caso novo, ao vivo: gazetteer -> entidades e medições -> P1 e P2."""
    meta = new_patient(texto)
    entidades, medicoes = analyze(texto, DEFAULT_MAX_N, revisao_gazetteer)
    caso = {**meta, "case_text": texto}
    p1 = extract_relations(caso, entidades)
    freq = load_frequencies(DEFAULT_TOKEN_FREQ)
    p2, _ = relacoes_p2(carregar_modelo(revisao_modelo), caso, texto, entidades, freq)
    return {"caso": caso, "entidades": entidades, "medicoes": medicoes, "p1": p1, "p2": p2}


@st.cache_data
def tabela(caminho: Path, revisao: tuple) -> pd.DataFrame:
    return pd.read_csv(caminho)


def revisao(*caminhos: Path) -> tuple:
    return tuple(c.stat().st_mtime_ns if c.exists() else 0 for c in caminhos)


# --------------------------------------------------------------------------
# Abas
# --------------------------------------------------------------------------
def aba_grafo(metodo, ctx, artigo=None):
    """`ctx`: caso, entidades, medicoes, p1 e p2 (do corpus ou de um caso novo)."""
    caso, p1, p2 = ctx["caso"], ctx["p1"], ctx["p2"]
    chaves1, chaves2 = {chave_relacao(l) for l in p1}, {chave_relacao(l) for l in p2}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Arestas P1", len(chaves1))
    c2.metric("Arestas P2", len(chaves2))
    c3.metric("Nos dois", len(chaves1 & chaves2))
    c4.metric("Só em um", len(chaves1 ^ chaves2))

    if metodo == "p1":
        relacoes = p1
    elif metodo == "p2":
        relacoes = p2
    else:
        relacoes = combinar_relacoes(p1, p2)
        st.markdown(
            '<span style="color:#ea580c">■</span> só P1 (regras) &nbsp; '
            '<span style="color:#16a34a">■</span> só P2 (aprendido) &nbsp; '
            '<span style="color:#7c3aed">■</span> os dois concordam &nbsp; '
            '<span style="color:#dc2626">■</span> negada (cor do P1)',
            unsafe_allow_html=True)
        st.caption("Cada aresta leva a cor do método que a produziu. As arestas dos dois métodos "
                   "usam as mesmas entidades; o que muda é quem decide quais pares são relações.")

    meta = {k: v for k, v in caso.items() if k != "case_text"}
    meta.setdefault("data_source", "data/interim/cases_clean.csv")
    st.markdown(render_legend(set(PALETTE)), unsafe_allow_html=True)
    graph_panel(meta, caso["case_text"], ctx["entidades"], ctx["medicoes"], PALETTE, artigo,
                relations=relacoes,
                umls_relations=ontologia(revisao(DEFAULT_RELATIONS.parent.parent
                                                 / "interim/umls_relations.csv")))


def aba_comparacao(dados):
    metricas = SAIDA / "metricas_validacao.csv"
    if not metricas.exists():
        st.warning("Rode `python -m clinical_kg.learning.pipeline treinar` para gerar as métricas.")
        return
    st.subheader("Validação deixando um caso de fora")
    st.caption("413 pares julgados à mão em 10 casos. Em cada rodada, o modelo, o PMI e os modelos "
               "de linguagem são refeitos sem o caso de teste. Os dois métodos veem os mesmos pares.")
    m = tabela(metricas, revisao(metricas))
    mostrar = m[["variante", "descricao", "precisao", "cobertura", "f1", "tipo_acertado",
                 "f1_caso_media", "f1_caso_dp", "casos_vs_p1_V-E-D"]].rename(columns={
        "descricao": "descrição", "precisao": "precisão", "f1_caso_media": "F1/caso (média)",
        "f1_caso_dp": "F1/caso (dp)", "casos_vs_p1_V-E-D": "casos vs P1 (V-E-D)",
        "tipo_acertado": "tipo certo"})
    st.dataframe(mostrar, hide_index=True, width="stretch")
    st.bar_chart(m, x="variante", y="f1", sort=False, height=260)
    st.warning("As quatro variantes do P2 ficam entre 0,74 e 0,76 de F1, dentro do ruído (desvio entre "
               "casos ≈ 0,06 a 0,09). O que se sustenta é que o aprendido supera as regras em 9 de 10 "
               "casos; não que o PMI ou o modelo de linguagem, sozinhos, melhorem o F1.")

    boot = SAIDA / "metricas_bootstrap.csv"
    if boot.exists():
        st.subheader("Intervalos de confiança (bootstrap por caso)")
        st.caption("Diferença de F1 (A − B), 10.000 repetições. Se o intervalo de 95% inclui o zero, a diferença "
                   "não se distingue do acaso com estes 10 casos.")
        st.dataframe(tabela(boot, revisao(boot)).rename(columns={
            "diferenca_f1": "diferença F1", "ic95_inf": "IC95 inf", "ic95_sup": "IC95 sup",
            "prob_a_maior": "P(A > B)"}), hide_index=True, width="stretch")

    fraco = SAIDA / "metricas_fraco.csv"
    if fraco.exists():
        st.subheader("Supervisão fraca: treinar só com as etiquetas das regras do P1")
        st.caption("As variantes *fraco_* aprendem com o que o P1 decidiu nos 50 casos (sem gold) e são provadas no "
                   "gold. *Recuperadas*: relações reais que o P1 perdeu e o modelo achou; *FP removidos/novos*: "
                   "falsos positivos do P1 que o modelo eliminou ou criou.")
        st.dataframe(tabela(fraco, revisao(fraco)).rename(columns={
            "concordancia_com_p1": "concorda com o P1", "relacoes_recuperadas": "recuperadas",
            "relacoes_perdidas": "perdidas", "falsos_positivos_removidos": "FP removidos",
            "falsos_positivos_novos": "FP novos"}), hide_index=True, width="stretch")

    por_caso = tabela(SAIDA / "metricas_por_caso.csv", revisao(SAIDA / "metricas_por_caso.csv"))
    st.subheader("F1 caso a caso")
    st.bar_chart(por_caso[por_caso.variante.isin(["p1", "completo"])]
                 .pivot(index="caso", columns="variante", values="f1"), height=300)

    por_rel = tabela(SAIDA / "metricas_por_relacao.csv", revisao(SAIDA / "metricas_por_relacao.csv"))
    st.subheader("F1 por tipo de relação (tipo e par corretos)")
    st.bar_chart(por_rel.pivot(index="relacao", columns="variante", values="f1"), height=300)
    st.caption("CAUSED_BY tem 4 exemplos no gold e não foi aprendida; HAS_DIAGNOSIS, HAS_FINDING e "
               "COORDINATE_WITH melhoram bastante.")

    st.subheader("O grafo, nos 50 casos")
    t1 = {chave_relacao(l) for ls in dados["p1"].values() for l in ls}
    t2 = {chave_relacao(l) for ls in dados["p2"].values() for l in ls}
    tipos = sorted({k[3] for k in t1 | t2})
    quadro = pd.DataFrame({"P1": [sum(1 for k in t1 if k[3] == t) for t in tipos],
                           "P2": [sum(1 for k in t2 if k[3] == t) for t in tipos],
                           "iguais": [sum(1 for k in t1 & t2 if k[3] == t) for t in tipos]}, index=tipos)
    st.bar_chart(quadro[["P1", "P2"]], height=300)
    st.caption(f"P1: {len(t1)} relações · P2: {len(t2)} · idênticas (mesmo par e tipo): {len(t1 & t2)}. "
               "Desempenho parecido ou melhor, grafos bem diferentes. Em 10 dos 50 casos o modelo final "
               "viu os pares julgados no treino, então esses casos são otimistas.")


def aba_pmi():
    if not TABELA_PMI.exists():
        st.warning("Rode `python -m clinical_kg.learning.pipeline tabelas` para gerar as tabelas.")
        return
    st.subheader("Bigramas que se atraem mais que o acaso (PMI)")
    st.caption("PMI = log₂ P(a,b) / (P(a)·P(b)). Alto: formam uma unidade. Só entram bigramas vistos "
               "pelo menos 5 vezes. LLR (Dunning, 1993) mede o mesmo com menos viés em corpus pequeno.")
    pmi = tabela(TABELA_PMI, revisao(TABELA_PMI))
    st.dataframe(pmi.head(40), hide_index=True, width="stretch", height=300)
    st.caption("No topo estão termos clínicos (\"pleural effusion\"), não gatilhos de relação. Os gatilhos "
               "são palavras frequentes e têm PMI baixo; por isso a tabela seguinte olha só para o "
               "texto entre as entidades.")

    st.subheader("Frases mais usadas entre duas entidades")
    ng = tabela(TABELA_NGRAMAS, revisao(TABELA_NGRAMAS))
    entre = ng[(ng.contagem_entre >= 3)].sort_values("contagem_entre", ascending=False).head(40).copy()
    entre["no léxico de gatilhos do P1"] = entre.ngrama.map(
        lambda f: any(p in regras_p1.TRIGGER_LEXICON for p in f.split()))
    st.dataframe(entre[["ngrama", "n", "contagem_entre", "contagem_corpus",
                        "no léxico de gatilhos do P1"]], hide_index=True, width="stretch", height=360)


def aba_pesos():
    pesos_csv = SAIDA / "pesos_modelo.csv"
    if not pesos_csv.exists():
        st.warning("Rode `python -m clinical_kg.learning.pipeline treinar` para gerar os pesos.")
        return
    pesos = tabela(pesos_csv, revisao(pesos_csv))
    st.subheader("O que a regressão logística aprendeu")
    st.caption("Um modelo por relação (um-contra-todos). Peso positivo empurra para a relação; negativo "
               "afasta. Os atributos de associação (pmi_*, llr_max) e lm_score são os que vêm dos n-gramas.")
    classe = st.selectbox("Relação", sorted(pesos.classe.unique()),
                          index=sorted(pesos.classe.unique()).index("TREATED_WITH")
                          if "TREATED_WITH" in set(pesos.classe) else 0)
    sel = pesos[pesos.classe == classe].copy()
    sel["abs"] = sel.peso.abs()
    st.bar_chart(sel.sort_values("abs", ascending=False).head(15).set_index("atributo")[["peso"]],
                 horizontal=True, height=420)
    st.dataframe(sel[sel.atributo.str.startswith(("pmi", "llr", "lm_"))][["atributo", "peso"]],
                 hide_index=True, width="stretch")


def aba_sobre():
    st.markdown("""
**Cadeia do Projeto 2:** n-gramas → PMI/LLR → modelo de linguagem (empilhamento, Wolpert 1992) →
regressão logística → grafo.

* **Candidatos:** os pares de entidades do Projeto 1, sem alteração.
* **Atributos:** tipos das entidades, distância, pontuação, PMI/LLR dos bigramas entre as entidades e
  um *score* de modelo de linguagem (calculado fora de amostra para não vazar o rótulo).
* **Decisão:** regressão logística um-contra-todos escrita à mão (verificada com scikit-learn).
* **Do Projeto 1 continua:** as entidades (gazetteer UMLS) e o `assertion_status` (negação, hipótese,
  histórico).

Os arquivos vêm de `data/processed/`. O modelo final está em `modelo.json`.
""")
    if MODELO_JSON.exists():
        st.caption(f"modelo.json: {MODELO_JSON.stat().st_size / 1024:.0f} KB")


def caso_novo():
    """Texto colado/enviado -> gazetteer -> P1 e P2. Devolve o contexto, ou None."""
    if not DEFAULT_DB.exists():
        st.sidebar.error("Gazetteer não encontrado. Copie `gazetteer.db` para "
                         f"`{DEFAULT_DB}` ou defina a variável CLINICAL_KG_GAZETTEER.")
        return None
    envio = st.sidebar.file_uploader("Relato de caso (.txt)", type=["txt"])
    colado = st.sidebar.text_area("…ou cole o texto aqui", height=220,
                                  placeholder="A 62-year-old man presented with chest pain and was "
                                              "treated with aspirin...")
    texto = envio.getvalue().decode("utf-8", errors="replace") if envio is not None else colado
    texto = texto.replace("\r\n", "\n")
    if not texto.strip():
        return None
    with st.spinner("Extraindo entidades e relações..."):
        ctx = analisar_caso_novo(texto, revisao(MODELO_JSON), file_revision(DEFAULT_DB))
    st.sidebar.caption(f"{len(ctx['entidades'])} entidades · {len(ctx['medicoes'])} medições · "
                       f"{len(ctx['p1'])} relações P1 · {len(ctx['p2'])} relações P2")
    return ctx


# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Projeto 2 · Grafo de conhecimento clínico", layout="wide")
    st.title("Projeto 2 · P1 (regras) × P2 (aprendido)")

    ausentes = [c for c in ARQUIVOS_NECESSARIOS if not c.exists()]
    if ausentes:
        st.error("Faltam arquivos: " + ", ".join(str(a) for a in ausentes)
                 + ". Rode `bash run_all.sh` para gerá-los.")
        st.stop()

    dados = carregar_dados(revisao(*ARQUIVOS_NECESSARIOS))
    st.markdown(_colour_css(), unsafe_allow_html=True)

    rotulo = st.sidebar.radio("Método", list(METODOS), index=2)
    metodo = METODOS[rotulo]
    origem = st.sidebar.radio("Origem do caso", ["Caso do corpus", "Caso novo (ao vivo)"])
    st.sidebar.markdown("---")

    ctx, artigo = None, None
    if origem == "Caso do corpus":
        ids = [c["case_id"] for c in dados["casos"]]
        escolhido = st.sidebar.selectbox(f"Caso ({len(ids)})", ids)
        caso = next(c for c in dados["casos"] if c["case_id"] == escolhido)
        st.sidebar.markdown(f"**Paciente** `{caso['case_id']}`  \nIdade: **{caso['age'] or '—'}**  \n"
                            f"Sexo: **{caso['gender'] or '—'}**")
        cid = caso["case_id"]
        ctx = {"caso": caso, "entidades": dados["entidades"].get(cid, []),
               "medicoes": dados["medicoes"][cid], "p1": dados["p1"][cid], "p2": dados["p2"][cid]}
        artigo = dados["artigos"].get(caso.get("article_id"))
    else:
        ctx = caso_novo()

    abas = st.tabs(["Grafo", "Comparação experimental", "PMI e n-gramas", "Pesos do modelo", "Sobre"])
    with abas[0]:
        if ctx is None:
            st.info("Cole ou envie o texto de um caso na barra lateral para ver os dois grafos.")
        else:
            aba_grafo(metodo, ctx, artigo)
    with abas[1]:
        aba_comparacao(dados)
    with abas[2]:
        aba_pmi()
    with abas[3]:
        aba_pesos()
    with abas[4]:
        aba_sobre()


if __name__ == "__main__":
    main()
