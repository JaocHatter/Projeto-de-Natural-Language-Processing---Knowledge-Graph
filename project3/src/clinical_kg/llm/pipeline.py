#!/usr/bin/env python3
"""As variantes com LLM (P3-A, P3-B, P3-C, P3-E, P3-F), do pedido à tabela.

    python -m clinical_kg.llm.pipeline estimar             # quantas chamadas cada variante custa
    python -m clinical_kg.llm.pipeline anotar --modo zero  # pede ao LLM o que falta no cache
    python -m clinical_kg.llm.pipeline anotar --modo few --k 5
    python -m clinical_kg.llm.pipeline anotar --conjunto fracos   # os 2.391 candidatos (P3-F)
    python -m clinical_kg.llm.pipeline avaliar             # tabela, só com o cache (sem rede)

`anotar` precisa de CLINICAL_KG_LLM_URL e CLINICAL_KG_LLM_MODELO (ver llm/provedor.py); por
exemplo, com o Ollama local:

    ollama pull qwen2.5:7b
    CLINICAL_KG_LLM_URL=http://localhost:11434/v1 CLINICAL_KG_LLM_MODELO=qwen2.5:7b \\
        python -m clinical_kg.llm.pipeline anotar --modo zero

`avaliar` lê só o cache e mede no protocolo do P2 (deixa um caso de fora, bootstrap por caso).
As variantes cujas respostas ainda não estão no cache são puladas, com a contagem do que falta.
Os exemplos few-shot de um par vêm só dos OUTROS casos do gold, como no resto do projeto.
"""

import argparse
import sys

from clinical_kg.learning.modelo import carregar_candidatos_fracos, carregar_exemplos
from clinical_kg.learning.pipeline import (VARIANTES, bootstrap_pareado, gravar_csv, medir,
                                           prever_validando, textos_dos_casos)
from clinical_kg.paths import RESULTADOS_GOLD

from .anotador import Anotador, descrever_par, escolher_fewshot
from .provedor import RespostaAusente, provedor_padrao

K_AUTOCONSISTENCIA = 5
N_FEWSHOT = 4


def fewshot_do_par(e, gold, vetores, textos):
    from clinical_kg.hibrido.embeddings import texto_do_par
    escolhidos = escolher_fewshot(e, [g for g in gold if g.caso != e.caso], vetores, N_FEWSHOT)
    return [(descrever_par(g, textos[g.caso]), g.relacao,
             texto_do_par(textos[g.caso], g.chave) if g.relacao != "NENHUMA" else "")
            for g in escolhidos]


def votar(exemplos, textos, anotador, gold=None, vetores=None) -> tuple[dict, int]:
    """{chave: voto} para os pares cujas respostas existem; e quantos pares ficaram sem resposta."""
    votos, faltam = {}, 0
    for e in exemplos:
        par = descrever_par(e, textos[e.caso])
        fewshot = fewshot_do_par(e, gold, vetores, textos) if vetores is not None else ()
        try:
            votos[e.chave] = anotador.anotar(par, fewshot)
        except RespostaAusente:
            faltam += 1
    return votos, faltam


def configuracoes(args_k=None):
    """(nome, conjunto, few-shot?, k) de cada coleção de votos que as variantes usam."""
    return [("zero_k1", "gold", False, 1), ("few_k1", "gold", True, 1),
            ("few_k5", "gold", True, args_k or K_AUTOCONSISTENCIA), ("zero_k1_fracos", "fracos", False, 1)]


def cmd_estimar(args):
    gold = carregar_exemplos()
    fracos = carregar_candidatos_fracos()
    print(f"gold: {len(gold)} pares; candidatos dos 50 casos: {len(fracos)}")
    p1 = prever_validando(gold, textos_dos_casos(), "p1")
    p2 = prever_validando(gold, textos_dos_casos(), "completo")
    discordam = sum(1 for (_, a), (_, b) in zip(p1, p2) if a != b)
    linhas = [("P3-A zero-shot", len(gold)), ("P3-B few-shot", len(gold)),
              (f"P3-B few-shot, autoconsistência k={K_AUTOCONSISTENCIA}", K_AUTOCONSISTENCIA * len(gold)),
              ("P3-C empilhamento (usa os votos do P3-A)", 0),
              ("P3-E cascata (só onde P1 e P2 discordam)", discordam),
              ("P3-F supervisão fraca (todos os candidatos)", len(fracos)),
              (f"P3-F com k={K_AUTOCONSISTENCIA}", K_AUTOCONSISTENCIA * len(fracos))]
    for nome, n in linhas:
        print(f"  {nome:<52}{n:>7} chamadas")
    gravar_csv(RESULTADOS_GOLD / "p3_custo_llm.csv", ["variante", "chamadas"], linhas)


def cmd_anotar(args):
    textos = textos_dos_casos()
    gold = carregar_exemplos()
    exemplos = gold if args.conjunto == "gold" else carregar_candidatos_fracos()
    if args.limite:
        exemplos = exemplos[:args.limite]
    provedor = provedor_padrao()
    vetores = None
    if args.modo == "few":
        from clinical_kg.hibrido.embeddings import Codificador, vetores_dos_exemplos
        vetores = vetores_dos_exemplos(exemplos + gold, textos, Codificador())
    votos, faltam = votar(exemplos, textos, Anotador(provedor, args.k), gold, vetores)
    print(f"{len(votos)} pares anotados, {faltam} sem resposta; chamadas novas: {provedor.chamadas}, "
          f"do cache: {provedor.acertos_cache}")


def cmd_avaliar(args):
    from clinical_kg.hibrido.transferencia import prever_validando_hibrido

    from .fontes import FonteLLM, prever_cascata, prever_llm, rotular_com_llm

    textos = textos_dos_casos()
    gold = carregar_exemplos()
    provedor = provedor_padrao()
    vetores = None
    resultados = {"p1": medir(prever_validando(gold, textos, "p1")),
                  "p2_completo": medir(prever_validando(gold, textos, "completo"))}
    colecoes, faltas = {}, {}
    for nome, conjunto, few, k in configuracoes():
        if conjunto != "gold":
            continue
        if few and vetores is None:
            try:
                from clinical_kg.hibrido.embeddings import Codificador, vetores_dos_exemplos
                vetores = vetores_dos_exemplos(gold, textos, Codificador())
            except ImportError:
                faltas[nome] = len(gold)
                continue
        votos, faltam = votar(gold, textos, Anotador(provedor, k), gold, vetores if few else None)
        if faltam:
            faltas[nome] = faltam
        else:
            colecoes[nome] = votos

    if "zero_k1" in colecoes:
        v = colecoes["zero_k1"]
        resultados["p3a_zero"] = medir(prever_llm(gold, v))
        resultados["p3c_completo"] = medir(prever_validando_hibrido(
            gold, textos, VARIANTES["completo"], fabricar_fontes=lambda: [FonteLLM(v)]))
        cascata, consultas = prever_cascata(prever_validando(gold, textos, "p1"),
                                            prever_validando(gold, textos, "completo"), v)
        resultados["p3e_cascata"] = medir(cascata)
        print(f"P3-E consultou o LLM em {consultas} de {len(gold)} pares")
    for nome, chave in (("few_k1", "p3b_few"), ("few_k5", "p3b_few_k5")):
        if nome in colecoes:
            resultados[chave] = medir(prever_llm(gold, colecoes[nome]))

    fracos = carregar_candidatos_fracos()
    votos_fracos, faltam = votar(fracos, textos, Anotador(provedor, 1))
    if faltam:
        faltas["zero_k1_fracos"] = faltam
    else:
        from clinical_kg.learning.modelo import ModeloAprendido
        rotulados = rotular_com_llm(fracos, votos_fracos)
        saida = []
        for caso in sorted({e.caso for e in gold}):
            modelo = ModeloAprendido(**VARIANTES["completo"]).ajustar(
                [e for e in rotulados if e.caso != caso], [t for c, t in textos.items() if c != caso])
            saida += [(e, modelo.prever_exemplo(e)) for e in gold if e.caso == caso]
        resultados["p3f_fraco_llm"] = medir(saida)

    for nome, n in faltas.items():
        print(f"(votos {nome}: faltam {n} respostas no cache; rode `anotar`)")
    linhas = []
    print(f"\n{'variante':<16}{'prec':>7}{'cob':>7}{'F1':>7}{'tipo':>7}   vs P2 completo [IC 95%]")
    for nome, m in resultados.items():
        r = bootstrap_pareado(m["contagem_por_caso"], resultados["p2_completo"]["contagem_por_caso"])
        print(f"{nome:<16}{m['p']:>7.3f}{m['r']:>7.3f}{m['f1']:>7.3f}{m['tipo']:>7.3f}"
              f"   {r['diferenca']:+.3f} [{r['ic_inf']:+.3f}, {r['ic_sup']:+.3f}]")
        linhas.append((nome, f"{m['p']:.4f}", f"{m['r']:.4f}", f"{m['f1']:.4f}", f"{m['tipo']:.4f}",
                       f"{r['diferenca']:+.4f}", f"{r['ic_inf']:+.4f}", f"{r['ic_sup']:+.4f}"))
    gravar_csv(RESULTADOS_GOLD / "p3_metricas_llm.csv",
               ["variante", "precisao", "cobertura", "f1", "tipo_acertado", "diferenca_vs_p2",
                "ic95_inf", "ic95_sup"], linhas)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):      # console do Windows (cp1252) não tem "−" nem "±"
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="comando", required=True)
    sub.add_parser("estimar", help="chamadas ao LLM que cada variante custa").set_defaults(func=cmd_estimar)
    a = sub.add_parser("anotar", help="pede ao LLM as respostas que faltam no cache")
    a.add_argument("--conjunto", choices=("gold", "fracos"), default="gold")
    a.add_argument("--modo", choices=("zero", "few"), default="zero")
    a.add_argument("--k", type=int, default=1, help="amostras por par (autoconsistência)")
    a.add_argument("--limite", type=int, default=0, help="só os N primeiros pares (teste rápido)")
    a.set_defaults(func=cmd_anotar)
    sub.add_parser("avaliar", help="tabela das variantes com LLM, só com o cache").set_defaults(func=cmd_avaliar)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
