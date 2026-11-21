#!/usr/bin/env python3
"""Projeto 3: Paciente Virtual, um gerador de cenários de treino preso ao grafo de conhecimento.

    streamlit run src/clinical_kg/app/projeto3.py        (ou: make app-p3)

Fluxo: escolhe-se um diagnóstico (ou sorteia-se) e a dificuldade; do grafo sai a ficha do
paciente; o estudante faz a anamnese, examina, pede exames e propõe um diagnóstico; o feedback
mostra o diferencial que o grafo calcula com as mesmas informações e o subgrafo por trás.

Não precisa do gazetteer UMLS nem de LLM: lê os corpora já construídos em data/processed/app/
(`python -m clinical_kg.paciente.corpus construir`). Com CLINICAL_KG_LLM_URL configurada, o caso
também pode ser narrado pelo LLM (restrito à ficha).
"""

import csv
import os
import random

import pandas as pd
import streamlit as st

from clinical_kg.graph.ui import render_graph
from clinical_kg.paciente.corpus import carregar, corpora_disponiveis
from clinical_kg.paciente.dialogo import Atendimento, menu_de_exames
from clinical_kg.paciente.feedback import PALETA_FEEDBACK, avaliar, diferencial, subgrafo
from clinical_kg.paciente.ficha import DIFICULDADES, concorrentes, montar_ficha
from clinical_kg.paciente.narrador import NarradorLLM, NarradorModelo
from clinical_kg.paciente.perfil import montar_perfis
from clinical_kg.paths import PROJECT_ROOT, RESULTADOS_GOLD, RESULTADOS_IDA_VOLTA, RESULTADOS_MACCROBAT

st.set_page_config(page_title="Paciente Virtual · Projeto 3", layout="wide")
ROTULO_DIFICULDADE = {"facil": "Fácil", "medio": "Médio", "dificil": "Difícil"}


@st.cache_resource(show_spinner="Montando os perfis de diagnóstico a partir do grafo…")
def base(corpora: tuple, minimo: int):
    corpus = carregar(list(corpora))
    return corpus, montar_perfis(corpus, minimo)


def narrador():
    if os.environ.get("CLINICAL_KG_LLM_URL") and st.session_state.get("usar_llm"):
        from clinical_kg.llm.provedor import provedor_padrao
        return NarradorLLM(provedor_padrao())
    return NarradorModelo()


def novo_paciente(perfis, nomes, dx_cui, dificuldade, semente):
    ficha = montar_ficha(perfis[dx_cui], perfis, dificuldade, semente, nomes)
    st.session_state.update(ficha=ficha, atendimento=Atendimento(ficha), resultado=None,
                            apresentacao=narrador().apresentacao(ficha))


def barra_lateral(corpus, perfis):
    st.sidebar.header("Novo paciente")
    ordenados = sorted(perfis.values(), key=lambda p: p.nome)
    rotulos = {p.cui: f"{p.nome} ({len(p.casos)} casos)" for p in ordenados}
    modo = st.sidebar.radio("Diagnóstico", ["Sortear", "Escolher"], horizontal=True)
    escolhido = None
    if modo == "Escolher":
        escolhido = st.sidebar.selectbox("Diagnóstico do paciente", list(rotulos), format_func=rotulos.get)
    dificuldade = st.sidebar.radio("Dificuldade", list(DIFICULDADES), format_func=ROTULO_DIFICULDADE.get,
                                   index=1, horizontal=True)
    if os.environ.get("CLINICAL_KG_LLM_URL"):
        st.sidebar.checkbox("Narrar com o LLM (restrito à ficha)", key="usar_llm")
    if st.sidebar.button("Gerar paciente", type="primary", width="stretch") or "ficha" not in st.session_state:
        semente = random.randrange(10**6)
        dx = escolhido or random.Random(semente).choice(sorted(perfis))
        novo_paciente(perfis, corpus.nomes, dx, dificuldade, semente)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Corpus: **{corpus.nome}** · {len(corpus.casos)} casos reais · "
                       f"{len(perfis)} perfis de diagnóstico")


def aba_atendimento(perfis):
    ficha, at = st.session_state["ficha"], st.session_state["atendimento"]
    st.subheader("Caso")
    st.info(st.session_state["apresentacao"])

    esquerda, direita = st.columns([3, 2])
    with direita:
        st.markdown("**Ações**")
        c1, c2 = st.columns(2)
        if c1.button("Queixa principal", width="stretch"):
            at.queixa_principal()
        if c2.button("Exame físico", width="stretch"):
            at.examinar()

        sintomas = {i.nome for i in ficha.itens if i.categoria in ("Symptom", "Finding")}
        for outro, _ in concorrentes(perfis[ficha.dx_cui], perfis, 4):
            sintomas |= {f.nome for f in outro.ordenados("Symptom")[:4]}
        alvo = st.selectbox("Perguntar sobre…", sorted(sintomas), index=None,
                            placeholder="sintoma ou sinal")
        if st.button("Perguntar", disabled=alvo is None, width="stretch"):
            at.perguntar(f"Do you have {alvo}?")
        livre = st.text_input("…ou pergunta livre (em inglês)", placeholder="Do you have any fever?")
        if st.button("Enviar pergunta", disabled=not livre, width="stretch"):
            at.perguntar(livre)

        exame = st.selectbox("Pedir exame", menu_de_exames(ficha, perfis), index=None, placeholder="exame")
        if st.button("Pedir", disabled=exame is None, width="stretch"):
            at.pedir_exame(exame)

        st.markdown("**Hipótese diagnóstica**")
        rotulos = {p.cui: p.nome for p in sorted(perfis.values(), key=lambda p: p.nome)}
        hipotese = st.selectbox("Diagnóstico", list(rotulos), format_func=rotulos.get, index=None,
                                placeholder="escolha um diagnóstico", label_visibility="collapsed")
        if st.button("Propor diagnóstico", type="primary", disabled=hipotese is None, width="stretch"):
            st.session_state["resultado"] = avaliar(ficha, at, hipotese, perfis)
            st.success("Veja a aba **Feedback**.")

    with esquerda:
        st.markdown("**Atendimento**")
        if not at.historico:
            st.caption("Comece pela queixa principal, pergunte, examine e peça exames.")
        for quem, texto in at.historico:
            with st.chat_message("user" if quem == "estudante" else "assistant",
                                 avatar="🧑‍⚕️" if quem == "estudante" else "🧑"):
                st.write(texto)
        if at.revelados:
            st.caption("Levantado até agora: " + ", ".join(
                (i.nome if i.presente else f"sem {i.nome}") for i in at.revelados.values()))
        ranking = diferencial(at.presentes_revelados(), at.negados_revelados(), perfis, n=3)
        with st.expander("O que o grafo pensaria com essas informações"):
            st.caption("Bayes ingênuo sobre os perfis do grafo; só usa o que você já levantou.")
            for d in ranking:
                st.write(f"{d['nome']} — {d['probabilidade']:.0%}")


def aba_feedback(perfis):
    r = st.session_state.get("resultado")
    ficha, at = st.session_state["ficha"], st.session_state["atendimento"]
    if r is None:
        st.info("Proponha um diagnóstico na aba **Atendimento** para ver o feedback.")
        return
    if r["correto"]:
        st.success(f"Correto: **{r['diagnostico']}**.")
    else:
        st.error(f"Sua hipótese: **{r['hipotese']}**. O diagnóstico era **{r['diagnostico']}**.")
    pos = r["posicao_do_certo"]
    st.write(f"Com os {r['levantados']} fatos que você levantou, o grafo põe o diagnóstico certo em "
             f"**{pos}º lugar** entre {len(perfis)}.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Diagnóstico diferencial do grafo**")
        st.dataframe(pd.DataFrame([{"diagnóstico": d["nome"], "probabilidade": f"{d['probabilidade']:.0%}",
                                    "casos reais": d["casos"]} for d in r["diferencial"]]),
                     hide_index=True, width="stretch")
    with c2:
        st.markdown("**O que você deixou passar** (ordenado pelo quanto separa o diagnóstico)")
        if r["perdidos"]:
            st.dataframe(pd.DataFrame([{"fato": ("" if p["presente"] else "ausente: ") + p["nome"],
                                        "poder discriminante": p["razao"],
                                        "casos reais": ", ".join(p["casos"][:3])} for p in r["perdidos"]]),
                         hide_index=True, width="stretch")
        else:
            st.write("Nada: você levantou todos os fatos da ficha.")

    st.markdown("**Subgrafo usado** — verde: levantado · laranja: não levantado · vermelho: negado · "
                "cinza: diagnósticos concorrentes que compartilham o que você levantou")
    render_graph(subgrafo(ficha, at, perfis, r["diferencial"]), PALETA_FEEDBACK)
    with st.expander("Caso completo (o que o narrador escreveria)"):
        st.write(NarradorModelo().completo(ficha))


def aba_ficha():
    ficha = st.session_state["ficha"]
    st.caption("A ficha é o gold do paciente virtual por construção: cada item guarda os casos reais "
               "que o sustentam. (Vista do professor: revela o diagnóstico.)")
    st.write(f"**Diagnóstico:** {ficha.dx_nome} · **dificuldade:** {ROTULO_DIFICULDADE[ficha.dificuldade]} · "
             f"**perfil:** {len(ficha.casos_do_perfil)} casos reais")
    st.dataframe(pd.DataFrame([{"categoria": i.categoria, "fato": i.nome,
                                "presente": "sim" if i.presente else "não (distrator)",
                                "frequência no perfil": f"{i.frequencia:.0%}",
                                "resultado": "; ".join(i.resultado or i.valores),
                                "casos reais": ", ".join(i.casos[:4])} for i in ficha.itens]),
                 hide_index=True, width="stretch")


def ler(caminho):
    if not caminho.exists():
        return None
    with open(caminho, newline="", encoding="utf-8") as f:
        return pd.DataFrame(list(csv.DictReader(f)))


def aba_resultados():
    g, m, iv = RESULTADOS_GOLD, RESULTADOS_MACCROBAT, RESULTADOS_IDA_VOLTA
    for titulo, caminho in (("Variantes no gold do MultiCaRe (deixa um caso de fora)", g / "p3_metricas_validacao.csv"),
                            ("Diferenças de F1 (bootstrap pareado por caso)", g / "p3_metricas_bootstrap.csv"),
                            ("Relações no MACCROBAT (gold externo, só humano)", m / "p3_externo_maccrobat.csv"),
                            ("Entidades do P1 no MACCROBAT", m / "p3_externo_entidades.csv"),
                            ("Asserção do P1 no MACCROBAT", m / "p3_externo_assercao.csv"),
                            ("Medidas do P1 no MACCROBAT", m / "p3_externo_medidas.csv"),
                            ("Ida e volta dos pacientes virtuais", iv / "p3_ida_volta_resumo.csv"),
                            ("Grafos nos 50 casos: P1 × P2 × P3", g / "p3_comparacao_arestas.csv")):
        df = ler(caminho)
        if df is not None:
            st.markdown(f"**{titulo}** · `{caminho.relative_to(PROJECT_ROOT).as_posix()}`")
            st.dataframe(df, hide_index=True, width="stretch")


def main():
    st.title("Paciente Virtual")
    st.caption("Casos de treino gerados a partir do grafo de conhecimento extraído de relatos reais "
               "(MultiCaRe e MACCROBAT). Tudo o que o paciente diz sai da ficha, e cada fato aponta para os casos reais "
               "que o sustentam.")
    disponiveis = corpora_disponiveis()
    if not disponiveis:
        st.error("Nenhum corpus em data/processed/app/. Rode "
                 "`python -m clinical_kg.paciente.corpus construir --nome multicare50`.")
        return
    padrao = [c for c in ("digestivo", "multicare50", "maccrobat") if c in disponiveis] or disponiveis[:1]
    corpora = st.sidebar.multiselect("Corpus do grafo", disponiveis, default=padrao)
    minimo = st.sidebar.slider("Mínimo de casos reais por diagnóstico", 2, 10, 3)
    if not corpora:
        st.warning("Escolha ao menos um corpus.")
        return
    corpus, perfis = base(tuple(sorted(corpora)), minimo)
    if not perfis:
        st.warning("Nenhum diagnóstico com casos suficientes; diminua o mínimo.")
        return
    if "ficha" in st.session_state and st.session_state["ficha"].dx_cui not in perfis:
        del st.session_state["ficha"]
    barra_lateral(corpus, perfis)
    abas = st.tabs(["Atendimento", "Feedback", "Ficha (professor)", "Resultados do P3"])
    with abas[0]:
        aba_atendimento(perfis)
    with abas[1]:
        aba_feedback(perfis)
    with abas[2]:
        aba_ficha()
    with abas[3]:
        aba_resultados()


main()
