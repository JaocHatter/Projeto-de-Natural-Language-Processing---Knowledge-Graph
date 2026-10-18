"""Testes do Projeto 2. Rodar com: python -m unittest discover -s tests"""

import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
