"""Figuras do Projeto 3, desenhadas a partir das tabelas em data/processed/resultados/ (precisa de matplotlib).

Cores dos visualizadores: P1 laranja, P2 verde, P3 roxo.
"""

import csv

from clinical_kg.paths import (PROJECT_ROOT, RESULTADOS_GOLD, RESULTADOS_IDA_VOLTA,
                               RESULTADOS_MACCROBAT)

PASTA = PROJECT_ROOT / "assets/images"
COR = {"p1": "#ea580c", "p2": "#16a34a", "p3": "#7c3aed", "neutra": "#94a3b8"}


def _ler(nome):
    pasta = (RESULTADOS_MACCROBAT if nome.startswith(("p3_externo", "maccrobat")) else
             RESULTADOS_IDA_VOLTA if nome.startswith("p3_ida_volta") else RESULTADOS_GOLD)
    caminho = pasta / nome
    if not caminho.exists():
        return None
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cor(variante: str) -> str:
    return COR["p1"] if variante == "p1" else COR["p2"] if variante.startswith("p2") else COR["p3"]


def desenhar_todas() -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    PASTA.mkdir(parents=True, exist_ok=True)
    gravadas = []

    def gravar(fig, nome):
        fig.tight_layout()
        fig.savefig(PASTA / nome, dpi=160)
        plt.close(fig)
        gravadas.append(PASTA / nome)

    # 1. F1 das variantes no gold, com o IC da diferença contra o P2 completo
    metricas = _ler("p3_metricas_validacao.csv")
    boot = {(b["a"], b["b"]): b for b in (_ler("p3_metricas_bootstrap.csv") or [])}
    if metricas:
        nomes = [m["variante"] for m in metricas][::-1]
        f1 = {m["variante"]: float(m["f1"]) for m in metricas}
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        barras = ax.barh(nomes, [f1[n] for n in nomes], color=[_cor(n) for n in nomes])
        for barra, n in zip(barras, nomes):
            b = boot.get((n, "p2_completo"))
            nota = f"  {f1[n]:.3f}" + (f"   Δ vs P2 {float(b['diferenca_f1']):+.3f} "
                                       f"[{float(b['ic95_inf']):+.3f}, {float(b['ic95_sup']):+.3f}]" if b else "")
            ax.text(barra.get_width(), barra.get_y() + barra.get_height() / 2, nota, va="center", fontsize=8)
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("F1 no gold (413 pares, deixando um caso de fora)")
        ax.set_title("Variantes do P3 na mesma tabela do P2", loc="left")
        gravar(fig, "p3_f1_variantes.png")

    # 2. Generalização: o P1 e o P2 no MACCROBAT (anotação só humana)
    ent = _ler("p3_externo_entidades.csv")
    rel = _ler("p3_externo_maccrobat.csv")
    ass = _ler("p3_externo_assercao.csv")
    if ent and rel and ass:
        fig, eixos = plt.subplots(1, 3, figsize=(12, 3.6))
        linhas = [l for l in ent if l["modo"] == "sobreposicao_tipo" and l["grupo"] != "TOTAL"]
        eixos[0].barh([l["grupo"] for l in linhas], [float(l["f1"]) for l in linhas], color=COR["p1"])
        eixos[0].set_xlim(0, 1)
        eixos[0].set_title("Entidades do gazetteer (F1, sobreposição + tipo)", loc="left", fontsize=9)
        tipos = ["CAUSED_BY", "LOCATED_IN"]
        decisores = [("p1", COR["p1"]), ("p2_completo", COR["p2"])]
        for k, (d, cor) in enumerate(decisores):
            vals = [float(next(l["f1"] for l in rel if l["decisor"] == d and l["tipo"] == t)) for t in tipos]
            eixos[1].bar([i + 0.38 * k for i in range(len(tipos))], vals, 0.38, color=cor,
                         label="P1 (regras)" if d == "p1" else "P2 (aprendido)")
        eixos[1].set_xticks([i + 0.19 for i in range(len(tipos))])
        eixos[1].set_xticklabels(tipos)
        eixos[1].set_ylim(0, 1)
        eixos[1].legend(frameon=False, fontsize=8)
        eixos[1].set_title("Relações (F1)", loc="left", fontsize=9)
        classes = ["negated", "hedged"]
        for k, (d, cor, rot) in enumerate((("p1", COR["p1"], "P1"), ("p3", COR["p3"], "P1 + coordenação (P3)"))):
            vals = [float(next(l["f1"] for l in ass if l["decisor"] == d and l["classe"] == c)) for c in classes]
            eixos[2].bar([i + 0.38 * k for i in range(2)], vals, 0.38, color=cor, label=rot)
        eixos[2].set_xticks([0.19, 1.19])
        eixos[2].set_xticklabels(["negação", "hipótese"])
        eixos[2].set_ylim(0, 1)
        eixos[2].legend(frameon=False, fontsize=8)
        eixos[2].set_title("Asserção (F1)", loc="left", fontsize=9)
        fig.suptitle("MACCROBAT2020: 200 relatos anotados só por humanos", x=0.01, ha="left")
        gravar(fig, "p3_externo_maccrobat.png")

    # 3. Ida e volta: fidelidade e relações recuperadas
    iv = _ler("p3_ida_volta_resumo.csv")
    if iv:
        linhas = [l for l in iv if l["dificuldade"] != "todas" and l["narrador"] == "modelo"]
        ordem = {"facil": 0, "medio": 1, "dificil": 2}
        linhas.sort(key=lambda l: ordem[l["dificuldade"]])
        fig, eixos = plt.subplots(1, 2, figsize=(10, 3.6))
        x = range(len(linhas))
        eixos[0].bar([i - 0.2 for i in x], [float(l["erro_polaridade"]) for l in linhas], 0.4,
                     color=COR["p1"], label="asserção do P1")
        eixos[0].bar([i + 0.2 for i in x], [float(l["erro_polaridade_p3"]) for l in linhas], 0.4,
                     color=COR["p3"], label="P1 + coordenação (P3)")
        eixos[0].set_xticks(list(x))
        eixos[0].set_xticklabels([l["dificuldade"] for l in linhas])
        eixos[0].set_ylabel("fatos com polaridade trocada")
        eixos[0].legend(frameon=False, fontsize=8)
        eixos[0].set_title("Erro de polaridade por dificuldade", loc="left", fontsize=9)
        todas = next(l for l in iv if l["dificuldade"] == "todas" and l["narrador"] == "modelo")
        nomes = ["P1", "P2", "P3"]
        vals = [float(todas[f"rel_{n.lower()}"]) for n in nomes]
        barras = eixos[1].bar(nomes, vals, color=[COR["p1"], COR["p2"], COR["p3"]])
        eixos[1].bar_label(barras, fmt="%.2f", padding=2)
        eixos[1].set_ylim(0, 1)
        eixos[1].set_title("Relações da ficha recuperadas do texto", loc="left", fontsize=9)
        fig.suptitle(f"Ida e volta: {todas['fichas']} pacientes virtuais narrados e reextraídos", x=0.01, ha="left")
        gravar(fig, "p3_ida_volta.png")
    return gravadas
