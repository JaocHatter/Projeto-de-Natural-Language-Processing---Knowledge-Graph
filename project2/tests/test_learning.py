"""Testes do Projeto 2. Rodar com: python -m unittest discover -s tests"""

import math
import unittest

from clinical_kg.learning.modelo import (NENHUMA, RegressaoLogistica, UmContraTodos,
                                         UmContraTodosSklearn, atributos,
                                         atributos_associacao, carregar_exemplos,
                                         sklearn_disponivel)
from clinical_kg.learning.ngramas import (Associacoes, ModeloLinguagem,
                                          contar_ngramas, tokenizar)


class TestNgramas(unittest.TestCase):
    def test_tokenizar_minusculas_e_ignora_pontuacao(self):
        self.assertEqual(tokenizar("Treated with Aspirin, 40 mg."),
                         ["treated", "with", "aspirin", "40", "mg"])

    def test_contar_ngramas(self):
        c = contar_ngramas(["a", "b", "a", "b"], n_max=2)
        self.assertEqual(c[("a",)], 2)
        self.assertEqual(c[("a", "b")], 2)
        self.assertEqual(c[("b", "a")], 1)
        self.assertNotIn(("a", "b", "a"), c)


class TestAssociacoes(unittest.TestCase):
    def setUp(self):
        # "x y" sempre juntos; "p" e "q" aparecem sempre separados.
        self.textos = ["x y p"] * 6 + ["q z"] * 6 + ["p q"] * 6
        self.a = Associacoes(self.textos, contagem_minima=5)

    def test_pmi_calculado_a_mao(self):
        # 24 bigramas, 42 palavras; c(x,y)=6, c(x)=6, c(y)=6
        esperado = math.log2((6 / 24) / ((6 / 42) * (6 / 42)))
        self.assertAlmostEqual(self.a.pmi("x", "y"), esperado, places=6)

    def test_pmi_zero_abaixo_da_contagem_minima(self):
        b = Associacoes(["x y"], contagem_minima=5)
        self.assertEqual(b.pmi("x", "y"), 0.0)

    def test_par_inexistente_vale_zero(self):
        self.assertEqual(self.a.pmi("x", "q"), 0.0)
        self.assertEqual(self.a.llr("x", "q"), 0.0)

    def test_llr_alto_para_par_associado_e_nulo_para_independente(self):
        associado = Associacoes(["x y"] * 10 + ["z w"] * 10)
        independente = Associacoes(["a b"] * 6 + ["a c"] * 6 + ["d b"] * 6 + ["d c"] * 6)
        self.assertGreater(associado.llr("x", "y"), 10.0)
        self.assertAlmostEqual(independente.llr("a", "b"), 0.0, places=6)

    def test_ranking_ordenado(self):
        pmis = [r[3] for r in self.a.ranking("pmi")]
        self.assertEqual(pmis, sorted(pmis, reverse=True))


class TestModeloLinguagem(unittest.TestCase):
    def test_probabilidades_somam_um(self):
        m = ModeloLinguagem().ajustar([["treated", "with", "aspirin"], ["caused", "by"]])
        vocab = set(m.unigramas) | {"__nova__"}
        # A massa reservada a palavras novas faz a soma ficar um pouco abaixo de 1.
        soma = sum(m.probabilidade("treated", w) for w in vocab)
        self.assertLessEqual(soma, 1.0 + 1e-9)
        self.assertGreater(soma, 0.5)

    def test_texto_nunca_visto_tem_logprob_finita(self):
        m = ModeloLinguagem().ajustar([["treated", "with"]])
        self.assertTrue(math.isfinite(m.log_prob_media(["xyzzy", "frobnicate"])))

    def test_prefere_o_que_viu(self):
        m = ModeloLinguagem().ajustar([["treated", "with"]] * 5)
        self.assertGreater(m.log_prob_media(["treated", "with"]),
                           m.log_prob_media(["with", "treated"]))


class TestRegressaoLogistica(unittest.TestCase):
    def test_separa_dados_linearmente_separaveis(self):
        X = [{"a": 1.0}] * 20 + [{"b": 1.0}] * 20
        y = [1] * 20 + [0] * 20
        m = RegressaoLogistica(epocas=30).ajustar(X, y)
        self.assertEqual(m.prever({"a": 1.0}), 1)
        self.assertEqual(m.prever({"b": 1.0}), 0)
        self.assertGreater(m.pesos["a"], 0)
        self.assertLess(m.pesos["b"], 0)

    def test_probabilidade_entre_zero_e_um_mesmo_com_pesos_grandes(self):
        m = RegressaoLogistica()
        m.pesos, m.vies = {"a": 800.0}, 0.0
        self.assertEqual(m.probabilidade({"a": 1.0}), 1.0)
        m.pesos = {"a": -800.0}
        self.assertEqual(m.probabilidade({"a": 1.0}), 0.0)

    def test_peso_da_classe_rara_faz_o_modelo_disparar(self):
        # 3 positivos contra 60 negativos, todos com o mesmo atributo ambíguo.
        X = [{"x": 1.0}] * 63
        y = [1] * 3 + [0] * 60
        sem = RegressaoLogistica(peso_classe_rara=False).ajustar(X, y)
        com = RegressaoLogistica(peso_classe_rara=True).ajustar(X, y)
        self.assertEqual(sem.prever({"x": 1.0}), 0)
        self.assertGreater(com.probabilidade({"x": 1.0}), sem.probabilidade({"x": 1.0}))

    def test_resultado_reprodutivel_com_a_mesma_semente(self):
        X = [{"a": 1.0}, {"b": 1.0}] * 10
        y = [1, 0] * 10
        p1 = RegressaoLogistica(semente=3).ajustar(X, y).pesos
        p2 = RegressaoLogistica(semente=3).ajustar(X, y).pesos
        self.assertEqual(p1, p2)


class TestUmContraTodos(unittest.TestCase):
    def test_tres_classes(self):
        X = [{"a": 1.0}] * 15 + [{"b": 1.0}] * 15 + [{"c": 1.0}] * 15
        rotulos = ["TRATA"] * 15 + ["LOCAL"] * 15 + [NENHUMA] * 15
        m = UmContraTodos(epocas=30).ajustar(X, rotulos)
        self.assertEqual(m.prever({"a": 1.0}), "TRATA")
        self.assertEqual(m.prever({"b": 1.0}), "LOCAL")
        self.assertEqual(m.tem_relacao({"c": 1.0}), 0)
        self.assertEqual(m.tem_relacao({"a": 1.0}), 1)
        self.assertAlmostEqual(sum(m.probabilidades({"a": 1.0}).values()), 1.0)


@unittest.skipUnless(sklearn_disponivel(), "scikit-learn não instalado")
class TestSklearn(unittest.TestCase):
    def test_mesma_interface_e_mesmas_respostas_em_dados_simples(self):
        X = [{"a": 1.0}] * 15 + [{"b": 1.0}] * 15 + [{"c": 1.0}] * 15
        rotulos = ["TRATA"] * 15 + ["LOCAL"] * 15 + [NENHUMA] * 15
        m = UmContraTodosSklearn().ajustar(X, rotulos)
        self.assertEqual(m.prever({"a": 1.0}), "TRATA")
        self.assertEqual(m.prever({"b": 1.0}), "LOCAL")
        self.assertEqual(m.tem_relacao({"c": 1.0}), 0)
        self.assertAlmostEqual(sum(m.probabilidades({"a": 1.0}).values()), 1.0)

    def test_concorda_com_a_versao_a_mao_no_gold(self):
        ex = carregar_exemplos()
        X = [atributos(e) for e in ex]
        r = [e.relacao for e in ex]
        a_mao = UmContraTodos().ajustar(X, r)
        sk = UmContraTodosSklearn().ajustar(X, r)
        concordam = sum(a_mao.tem_relacao(x) == sk.tem_relacao(x) for x in X) / len(X)
        self.assertGreater(concordam, 0.85)


class TestAtributos(unittest.TestCase):
    def test_atributos_de_associacao(self):
        assoc = Associacoes(["x y"] * 10 + ["z w"] * 10, contagem_minima=5)
        f = atributos_associacao(["x", "y"], assoc)
        self.assertGreater(f["pmi_max"], 0)
        self.assertEqual(set(f), {"pmi_max", "pmi_medio", "llr_max"})
        self.assertEqual(atributos_associacao(["x"], assoc), {})

    def test_exemplos_do_gold(self):
        ex = carregar_exemplos()
        self.assertEqual(len(ex), 413)
        self.assertEqual(len({e.caso for e in ex}), 10)
        self.assertEqual(sum(e.eh_relacao for e in ex), 221)
        self.assertTrue(all(e.relacao == NENHUMA for e in ex if not e.eh_relacao))
        nomes = {n for e in ex for n in atributos(e)}
        # Nada que venha das regras do Projeto 1 (gatilho, categoria, score).
        self.assertFalse(any(n.startswith(("gatilho", "trigger", "score", "p1")) for n in nomes))
        self.assertIn("mesma_clausula", nomes)


if __name__ == "__main__":
    unittest.main()
