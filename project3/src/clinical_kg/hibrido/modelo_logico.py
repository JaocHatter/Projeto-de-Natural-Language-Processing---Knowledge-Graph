#!/usr/bin/env python3
"""Desenha o modelo lógico do grafo do Projeto 3 (SVG) no estilo da figura do P1/P2.

    python -m clinical_kg.hibrido.modelo_logico        # grava assets/images/modelo_logico_p3.svg

O PNG do README é o SVG renderizado por um navegador (Chrome) na largura natural.
Só o que está implementado aparece em traço cheio; o planejado (proposições do E1 e ordem
temporal) aparece tracejado e marcado "planejado".
"""

from html import escape

from clinical_kg.paths import PROJECT_ROOT

SAIDA = PROJECT_ROOT / "assets/images/modelo_logico_p3.svg"
LARGURA, ALTURA = 1660, 1260
LINHA = 30           # altura de cada atributo
CAB = 40             # altura do cabeçalho

COR = {
    "estrutural": "#4b5563", "texto": "#16a34a", "p3": "#7c3aed", "medida": "#ca8a04",
    "app": "#db2777", "plano": "#94a3b8", "ontologia": "#0f766e",
}


class Caixa:
    def __init__(self, x, y, w, titulo, cor, atributos, novo=None, plano=False):
        self.x, self.y, self.w, self.titulo, self.cor = x, y, w, titulo, cor
        self.atributos, self.novo, self.plano = atributos, set(novo or ()), plano
        self.h = CAB + LINHA * len(atributos)

    def svg(self):
        tracejado = ' stroke-dasharray="7 5"' if self.plano else ""
        partes = [f'<g opacity="{0.75 if self.plano else 1}">',
                  f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" rx="4" fill="#fff" '
                  f'stroke="#334155" stroke-width="1.2"{tracejado}/>',
                  f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{CAB}" rx="4" fill="{self.cor}"/>',
                  f'<text x="{self.x + self.w / 2}" y="{self.y + 26}" text-anchor="middle" fill="#fff" '
                  f'font-weight="700" font-size="17">{escape(self.titulo)}</text>']
        for i, (nome, tipo) in enumerate(self.atributos):
            y = self.y + CAB + LINHA * i
            if i:
                partes.append(f'<line x1="{self.x}" y1="{y}" x2="{self.x + self.w}" y2="{y}" stroke="#e5e7eb"/>')
            destaque = nome in self.novo
            partes.append(
                f'<text x="{self.x + 12}" y="{y + 20}" font-size="13.5" fill="{COR["p3"] if destaque else "#111827"}"'
                f'{" font-weight=\"700\"" if destaque else ""}>{escape(nome)}: '
                f'<tspan font-style="italic" fill="#6b7280" font-weight="400">{escape(tipo)}</tspan></text>')
        partes.append("</g>")
        return "\n".join(partes)

    # pontos de ancoragem
    def dir(self, f=0.5):
        return self.x + self.w, self.y + self.h * f

    def esq(self, f=0.5):
        return self.x, self.y + self.h * f

    def cima(self, f=0.5):
        return self.x + self.w * f, self.y

    def baixo(self, f=0.5):
        return self.x + self.w * f, self.y + self.h


def seta(p, q, cor, rotulo="", pos=None, curva=0.0, estilo="", tamanho=13, ancora="start", linhas=None):
    (x1, y1), (x2, y2) = p, q
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    cx, cy = mx - (y2 - y1) * curva, my + (x2 - x1) * curva
    dash = {"tracejado": ' stroke-dasharray="8 6"', "pontilhado": ' stroke-dasharray="2 5"'}.get(estilo, "")
    marcador = f"m-{cor.strip('#')}"
    out = [f'<path d="M{x1},{y1} Q{cx},{cy} {x2},{y2}" fill="none" stroke="{cor}" stroke-width="2.2"{dash} '
           f'marker-end="url(#{marcador})"/>']
    textos = linhas or ([rotulo] if rotulo else [])
    if textos:
        tx, ty = pos or (cx, cy)
        for i, t in enumerate(textos):
            peso = ' font-weight="700"' if i == 0 and linhas else ""
            out.append(f'<text x="{tx}" y="{ty + i * (tamanho + 4)}" font-size="{tamanho if i == 0 else tamanho - 1.5}" '
                       f'fill="{cor}" text-anchor="{ancora}"{peso}>{escape(t)}</text>')
    return "\n".join(out)


def desenhar() -> str:
    artigo = Caixa(70, 80, 230, "Article", "#7c8db0",
                   [("article_id", "string"), ("title", "string"), ("year", "string"), ("fonte", "multicare | maccrobat")],
                   novo={"fonte"})
    pessoa = Caixa(70, 300, 230, "Person", "#334155",
                   [("case_id", "string"), ("age", "string"), ("gender", "string")])
    idade = Caixa(420, 70, 200, "Age", "#8b5cf6", [("value", "string"), ("method", "string")])
    sexo = Caixa(420, 230, 200, "Sex", "#0891b2", [("value", "string"), ("method", "string")])
    conceito_a = Caixa(470, 520, 300, "Concept", "#2a78d6",
                       [("cui", "string"), ("entity_type", "string"), ("preferred_term", "string"),
                        ("assertion_status", "P1 + coordenação (P3)")], novo={"assertion_status"})
    conceito_b = Caixa(1150, 470, 300, "Concept", "#eb6834",
                       [("cui", "string"), ("entity_type", "string"), ("preferred_term", "string")])
    medida = Caixa(1150, 760, 300, "Measurement", "#64748b",
                   [("value", "string"), ("unit", "string"), ("unit_category", "string")])
    perfil = Caixa(470, 880, 300, "DiagnosisProfile", COR["app"],
                   [("dx_cui", "string"), ("n_casos", "int"), ("corpus", "multicare50 | digestivo")],
                   novo={"dx_cui", "n_casos", "corpus"})
    virtual = Caixa(70, 880, 300, "VirtualPatient (ficha)", COR["app"],
                    [("dx_cui", "string"), ("idade, sexo", "de um caso real"), ("dificuldade", "facil|medio|dificil"),
                     ("semente", "int")], novo={"dx_cui", "idade, sexo", "dificuldade", "semente"})
    proposicao = Caixa(1150, 120, 300, "Proposition (E1)", COR["plano"],
                       [("texto", "string"), ("sentenca", "offset")], plano=True)

    caixas = [artigo, pessoa, idade, sexo, conceito_a, conceito_b, medida, perfil, virtual, proposicao]
    e, t, p3, m, app, pl = (COR[k] for k in ("estrutural", "texto", "p3", "medida", "app", "plano"))
    arestas = [
        seta(artigo.baixo(), pessoa.cima(), e, "REPORTS_PERSON", pos=(195, 290)),
        seta(pessoa.dir(0.2), idade.esq(0.6), e, "HAS_AGE", pos=(330, 150)),
        seta(pessoa.dir(0.45), sexo.esq(0.6), e, "HAS_SEX", pos=(330, 300)),
        seta(pessoa.baixo(0.8), conceito_a.cima(0.3), e, pos=(395, 400), curva=0.12,
             linhas=["MENTIONS_*", "(Diagnosis/Treatment/Exam/", "Symptom/Finding/BodyPart)"]),
        seta(conceito_a.dir(0.35), conceito_b.esq(0.55), t, curva=-0.06, pos=(800, 440), tamanho=13,
             linhas=["relação tipada (texto)", "HAS_SYMPTOM, HAS_DIAGNOSIS, HAS_FINDING,",
                     "TREATED_WITH, REVEALED_BY, LOCATED_IN,", "CAUSED_BY, COORDINATE_WITH",
                     "metodo: p1 | p2 | p3g | llm   score   assertion_status"]),
        seta(conceito_b.baixo(0.5), medida.cima(0.5), m, "ASSOCIATED_WITH_MEASUREMENT", pos=(1310, 730)),
        seta(conceito_a.baixo(0.5), perfil.cima(0.5), app, pos=(630, 800),
             linhas=["HAS_PROFILE", "(só Diagnosis, T047/T191)"]),
        seta(perfil.dir(0.35), conceito_b.esq(0.85), app, curva=0.08, pos=(975, 880),
             linhas=["INCLUDES", "frequencia, ligado, casos[ ]"]),
        seta(perfil.cima(0.12), pessoa.dir(0.85), app, curva=-0.1, estilo="pontilhado", pos=(300, 620), ancora="end",
             linhas=["AGGREGATES", "(casos reais do perfil)"]),
        seta(virtual.dir(0.5), perfil.esq(0.5), app, "SAMPLED_FROM", pos=(374, 990)),
        seta(virtual.cima(0.2), pessoa.baixo(0.2), app, estilo="tracejado", pos=(122, 730),
             linhas=["DERIVED_FROM", "(casos de cada fato)"]),
        seta(virtual.cima(0.75), conceito_a.esq(0.85), app, curva=0.05, pos=(180, 800),
             linhas=["PRESENTS", "categoria, presente (sim | negado),", "resultado, valores"]),
        seta(conceito_b.cima(0.25), proposicao.baixo(0.25), pl, estilo="tracejado", pos=(1160, 400),
             linhas=["MENTIONED_IN (planejado)"]),
        seta(conceito_b.dir(0.8), conceito_b.dir(0.2), pl, curva=-1.4, estilo="tracejado", pos=(1475, 525),
             linhas=["BEFORE / AFTER /", "OVERLAP (planejado;", "anotado no MACCROBAT)"]),
    ]
    marcadores = "".join(
        f'<marker id="m-{c.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for c in set(COR.values()))
    legenda_y = 1150
    legenda = [(e, "estrutural (identidade/demografia)"), (t, "relação tipada extraída do texto (P1, P2, P3-G; LLM a rodar)"),
               (m, "heurística (medição)"), (app, "Paciente Virtual (E6): perfis e fichas tirados do grafo"),
               (pl, "planejado (E1, ordem temporal)")]
    itens = []
    for i, (cor, texto) in enumerate(legenda):
        x = 960 + (i % 2) * 300 if False else 960
        y = legenda_y - 120 + i * 24
        itens.append(f'<line x1="{x}" y1="{y}" x2="{x + 34}" y2="{y}" stroke="{cor}" stroke-width="3"/>'
                     f'<text x="{x + 44}" y="{y + 5}" font-size="13.5" fill="#1f2937">{escape(texto)}</text>')
    nota = ('<text x="70" y="1240" font-size="13" fill="#4b5563">Em roxo, os atributos novos do P3. "Concept" '
            'aparece duas vezes só para desenhar as arestas entre conceitos: é o mesmo tipo de nó, parametrizado '
            'por entity_type.</text>')
    corpo = "\n".join([*(c.svg() for c in caixas), *arestas, *itens, nota])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{LARGURA}" height="{ALTURA}" '
            f'viewBox="0 0 {LARGURA} {ALTURA}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'
            f'<defs>{marcadores}</defs><rect width="100%" height="100%" fill="#fff"/>'
            f'<text x="70" y="45" font-size="26" font-weight="700" fill="#111827">Modelo Lógico do Projeto 3</text>'
            f'{corpo}</svg>')


def main():
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(desenhar(), encoding="utf-8")
    print(SAIDA.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
