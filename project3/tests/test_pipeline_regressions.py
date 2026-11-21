"""Regressões de proveniência, métricas tipadas e artefatos do pipeline."""

from contextlib import ExitStack, redirect_stdout
import csv
from functools import partial
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from clinical_kg.graph.model import build_graph
from clinical_kg.learning import pipeline as p
from clinical_kg.learning.modelo import Exemplo, ModeloAprendido, NENHUMA
from clinical_kg.relations.tools.evaluate import score


def exemplos():
    return [Exemplo(
        caso=f"c{c}", chave=(c, i), estruturais={"tratamento": float(i)},
        entre=["treated", "with"] if i else ["of", "the"],
        eh_relacao=i, relacao="TREATED_WITH" if i else NENHUMA,
        extras={"p1_score": 10.0 if i else -10.0, "p1_relacao": "TREATED_WITH"},
    ) for c in range(4) for i in (0, 1)]


class ProvenienciaTests(unittest.TestCase):
    def test_duplicata_p2_nao_inventa_participacao_p1(self):
        r = dict(case_id="c", head_start="0", tail_start="10", relation="LOCATED_IN")
        for p1, esperado in (([], "p2"), ([r], "ambos")):
            with self.subTest(esperado=esperado):
                saida = p.combinar_relacoes(p1, [r, r])
                self.assertEqual(len(saida), 1)
                self.assertEqual(saida[0]["origem"], esperado)
                self.assertNotIn("origem", r)


class AvaliacaoTipadaTests(unittest.TestCase):
    def test_troca_de_tipo_conta_fp_e_fn_sem_mudar_deteccao(self):
        gold = {"par": dict(is_relation="1", gold_relation="TREATED_WITH",
                            gold_assertion="affirmed")}
        pred = {"par": dict(score=10, relation="LOCATED_IN", assertion_status="affirmed")}
        m = score(gold, pred, 0)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["typed_tp"]), (1, 0, 0, 0))
        self.assertEqual(dict(m["per_relation"]),
                         {"TREATED_WITH": [0, 0, 1], "LOCATED_IN": [0, 1, 0]})

    def test_acerto_rejeicao_e_relacao_ausente(self):
        gold = {
            "certo": dict(is_relation="1", gold_relation="TREATED_WITH", gold_assertion="affirmed"),
            "perdido": dict(is_relation="1", gold_relation="LOCATED_IN", gold_assertion="affirmed"),
            "falso": dict(is_relation="0", gold_relation="", gold_assertion=""),
        }
        pred = {k: dict(score=10, relation="TREATED_WITH", assertion_status="affirmed")
                for k in ("certo", "falso")}
        m = score(gold, pred, 0)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["typed_tp"]), (1, 1, 1, 1))
        self.assertEqual(dict(m["per_relation"]),
                         {"TREATED_WITH": [1, 1, 0], "LOCATED_IN": [0, 0, 1]})


class ArtefatosPipelineTests(unittest.TestCase):
    def test_extrair_aceita_saida_absoluta_externa_e_relativa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modelo = root / "modelo.json"
            modelo.write_text(json.dumps(ModeloAprendido(False, False)
                                         .ajustar(exemplos(), []).para_dict()), encoding="utf-8")
            caso = dict(case_id="c", article_id="a", age="", gender="Unknown", case_text="")
            for relativo in (False, True):
                with self.subTest(relativo=relativo), ExitStack() as stack:
                    saida = root / f"saida_{relativo}.csv"
                    caminho = Path(os.path.relpath(saida)) if relativo else saida
                    for nome, valor in {
                        "ler_casos": [caso], "textos_dos_casos": {"c": ""},
                        "entidades_lf": {"c": []}, "load_frequencies": {},
                    }.items():
                        stack.enter_context(patch.object(p, nome, return_value=valor))
                    stack.enter_context(patch.object(p, "DEFAULT_MEASUREMENTS", root / "ausente.csv"))
                    stack.enter_context(patch.object(p, "DEFAULT_RELATIONS", root / "ausente.csv"))
                    graph = stack.enter_context(patch("clinical_kg.graph.model.build_graph", wraps=build_graph))
                    log = stack.enter_context(redirect_stdout(io.StringIO()))
                    p.cmd_extrair(SimpleNamespace(modelo=modelo, casos=None, entidades=None, saida=caminho))
                    with saida.open() as f:
                        self.assertEqual(next(csv.reader(f)), p.CSV_COLUMNS)
                    graph.assert_called_once()
                    self.assertIn("grafo:", log.getvalue())

    def test_treino_invalida_fraco_pulado_e_atualiza_fraco_executado(self):
        for sem_fraco, anterior in ((True, True), (True, False), (False, True)):
            with self.subTest(sem_fraco=sem_fraco, anterior=anterior), \
                    tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                fraco = root / "metricas_fraco.csv"
                if anterior:
                    fraco.write_text("resultado de uma execucao antiga\n", encoding="utf-8")
                for nome, valor in {
                    "SAIDA": root, "MODELO_JSON": root / "modelo.json", "PROJECT_ROOT": root,
                }.items():
                    stack.enter_context(patch.object(p, nome, valor))
                for nome, valor in {
                    "carregar_exemplos": exemplos(), "carregar_candidatos_fracos": exemplos(),
                    "textos_dos_casos": {f"c{i}": "treated with of the" for i in range(4)},
                    "sklearn_disponivel": False,
                }.items():
                    stack.enter_context(patch.object(p, nome, return_value=valor))
                stack.enter_context(patch.object(p, "bootstrap_pareado",
                                                 partial(p.bootstrap_pareado, repeticoes=20)))
                stack.enter_context(redirect_stdout(io.StringIO()))
                p.cmd_treinar(SimpleNamespace(casos=None, entidades=None, gold=None, sem_fraco=sem_fraco))
                with (root / "metricas_validacao.csv").open() as f:
                    variantes = {r["variante"] for r in csv.DictReader(f)}
                self.assertEqual(fraco.exists(), not sem_fraco)
                self.assertEqual(any(v.startswith("fraco_") for v in variantes), not sem_fraco)
                if not sem_fraco:
                    with fraco.open() as f:
                        self.assertEqual({r["variante"] for r in csv.DictReader(f)}, set(p.VARIANTES_FRACAS))
