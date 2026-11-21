"""Testes do Projeto 3: MACCROBAT, asserção, LLM (sem rede), modelo híbrido e paciente virtual.

Tudo com dados sintéticos: nenhum teste precisa do gazetteer UMLS, de rede ou de chave de API.
O teste da aplicação só roda se um corpus já construído existir em data/processed/app/.
"""

import collections
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_kg.hibrido.assercao import propagar_coordenacao  # noqa: E402
from clinical_kg.learning.modelo import NENHUMA, Exemplo  # noqa: E402
from clinical_kg.maccrobat.avaliacao import casar_entidades  # noqa: E402
from clinical_kg.maccrobat.brat import ler_documento  # noqa: E402
from clinical_kg.maccrobat.mapeamento import entidades_p1, papel_no_p1, rotulos_por_par  # noqa: E402

TEXTO_BRAT = "Fever and cough. CT of the chest revealed pneumonia secondary to aspiration.\n"
ANN_BRAT = "\n".join([
    "T1\tSign_symptom 0 5\tFever",
    "E1\tSign_symptom:T1",
    "T2\tSign_symptom 10 15\tcough",
    "T3\tDiagnostic_procedure 17 19\tCT",
    "T4\tBiological_structure 27 32\tchest",
    "T5\tDisease_disorder 42 51\tpneumonia",
    "T6\tDisease_disorder 65 75\taspiration",
    "T7\tDisease_disorder 42 75\tpneumonia secondary to aspiration",
    "R1\tCAUSE Arg1:T6 Arg2:T5",
    "R2\tMODIFY Arg1:T4 Arg2:T3",
    "A1\tPOLARITY E1 NEG",
    "*\tOVERLAP E1 T2",
])


class MaccrobatTests(unittest.TestCase):
    def setUp(self):
        self.doc = ler_documento("123", TEXTO_BRAT, ANN_BRAT)

    def test_leitor_resolve_eventos_e_atributos(self):
        self.assertEqual(self.doc.trechos["T5"].texto, "pneumonia")
        self.assertEqual(self.doc.atributos["T1"]["POLARITY"], "NEG")      # E1 -> T1
        self.assertEqual(self.doc.simultaneos, [["T1", "T2"]])
        self.assertIn(("CAUSE", "T6", "T5"), self.doc.relacoes)

    def test_mapeamento_de_papeis(self):
        self.assertEqual(papel_no_p1("CAUSE", "Disease_disorder", "Disease_disorder"), "CAUSED_BY")
        self.assertEqual(papel_no_p1("MODIFY", "Biological_structure", "Diagnostic_procedure"), "LOCATED_IN")
        self.assertEqual(papel_no_p1("MODIFY", "Lab_value", "Diagnostic_procedure"), "VALOR_DE_EXAME")
        self.assertEqual(papel_no_p1("MODIFY", "Dosage", "Medication"), "DOSE_DE_MEDICAMENTO")
        self.assertEqual(papel_no_p1("BEFORE", "Sign_symptom", "Disease_disorder"), "TEMPORAL")
        self.assertEqual(papel_no_p1("SUB_PROCEDURE", "Diagnostic_procedure", "Diagnostic_procedure"), "")

    def test_entidades_sem_sobreposicao_fica_a_mais_longa(self):
        entidades, descartes = entidades_p1(self.doc)
        textos = [e["surface_text"] for e in entidades]
        self.assertIn("pneumonia secondary to aspiration", textos)
        self.assertNotIn("pneumonia", textos)
        self.assertEqual(descartes["sobreposto"], 2)
        self.assertEqual(next(e for e in entidades if e["maccrobat_id"] == "T1")["assertion_status"], "negated")
        self.assertEqual(next(e for e in entidades if e["maccrobat_id"] == "T3")["entity_type"], "Exam")

    def test_casos_maccrobat_para_a_aplicacao(self):
        from clinical_kg.paciente.corpus import casos_maccrobat, idade_maccrobat, sexo_maccrobat
        texto = "A 28-year-old woman had fever."
        ann = "T1\tAge 2 13\t28-year-old\nT2\tSex 14 19\twoman\nT3\tSign_symptom 24 29\tfever"
        caso = casos_maccrobat([ler_documento("999", texto, ann), self.doc])
        self.assertEqual(caso[0], {"case_id": "MACCROBAT_999", "article_id": "999", "age": "28",
                                   "gender": "Female", "case_text": texto})
        self.assertEqual((caso[1]["age"], caso[1]["gender"]), ("", ""))     # sem trechos Age/Sex
        self.assertEqual(idade_maccrobat("6-month-old"), "0")
        self.assertEqual(idade_maccrobat("elderly"), "")
        self.assertEqual(sexo_maccrobat("Man"), "Male")

    def test_rotulos_por_par(self):
        r = rotulos_por_par(self.doc)
        self.assertEqual(r[frozenset(("T5", "T6"))], "CAUSED_BY")
        self.assertEqual(r[frozenset(("T3", "T4"))], "LOCATED_IN")

    def test_casamento_de_entidades_um_para_um(self):
        gold = [(0, 5, "A"), (10, 15, "A"), (20, 30, "B")]
        previstas = [(0, 15, "A"), (20, 30, "A")]
        g, p = casar_entidades(gold, previstas, "sobreposicao")
        self.assertEqual((len(g), len(p)), (2, 2))       # a longa casa com UM gold só
        g, p = casar_entidades(gold, previstas, "sobreposicao_tipo")
        self.assertEqual((len(g), len(p)), (1, 1))
        g, p = casar_entidades(gold, previstas, "exato")
        self.assertEqual((len(g), len(p)), (1, 1))


class AssercaoTests(unittest.TestCase):
    def ents(self, texto, palavras):
        saida = []
        for w in palavras:
            i = texto.index(w)
            saida.append({"start": i, "end": i + len(w)})
        return saida

    def test_negacao_atravessa_a_coordenacao(self):
        t = "She denied nausea, fever and chest pain."
        e = self.ents(t, ["nausea", "fever", "chest pain"])
        self.assertEqual(propagar_coordenacao(t, e, ["negated", "affirmed", "affirmed"]), ["negated"] * 3)

    def test_negacao_para_em_but_e_em_artigo(self):
        t = "No fever, but abdominal pain. CT showed no ascites and a liver mass."
        e = self.ents(t, ["fever", "abdominal pain", "ascites", "liver mass"])
        estados = propagar_coordenacao(t, e, ["negated", "affirmed", "negated", "affirmed"])
        self.assertEqual(estados, ["negated", "affirmed", "negated", "affirmed"])


def exemplo(caso, chave, tipo_cab="Diagnosis", tipo_cau="BodyPart", relacao=NENHUMA, cab="pain", cau="abdomen"):
    return Exemplo(caso=caso, chave=chave, estruturais={f"tipo_cabeca:{tipo_cab}": 1.0, f"tipo_cauda:{tipo_cau}": 1.0},
                   entre=["in", "the"], eh_relacao=int(relacao != NENHUMA), relacao=relacao, cabeca=cab, cauda=cau)


class LLMTests(unittest.TestCase):
    def setUp(self):
        from clinical_kg.llm.cache import CacheLLM
        self.pasta = tempfile.TemporaryDirectory()
        self.cache = CacheLLM(Path(self.pasta.name) / "cache.jsonl")
        self.texto = "She had pneumonia secondary to aspiration. Pain in the abdomen was severe."
        i = self.texto.index("pneumonia")
        j = self.texto.index("aspiration")
        self.par_causal = exemplo("c1", ("c1", i, i + 9, j, j + 10), "Diagnosis", "Diagnosis",
                                  cab="pneumonia", cau="aspiration")

    def tearDown(self):
        self.pasta.cleanup()

    def test_tipos_permitidos(self):
        from clinical_kg.llm.anotador import tipos_permitidos
        self.assertIn("LOCATED_IN", tipos_permitidos("Diagnosis", "BodyPart"))
        self.assertNotIn("LOCATED_IN", tipos_permitidos("Diagnosis", "Treatment"))
        self.assertEqual(set(tipos_permitidos("Paciente", "Symptom")),
                         {"HAS_SYMPTOM", "HAS_DIAGNOSIS", "HAS_FINDING", "TREATED_WITH"})

    def test_sentenca_marcada(self):
        from clinical_kg.llm.anotador import sentenca_marcada
        marcada, limpa = sentenca_marcada(self.texto, self.par_causal.chave)
        self.assertEqual(marcada, "She had [E1]pneumonia[/E1] secondary to [E2]aspiration[/E2].")
        self.assertEqual(limpa, "She had pneumonia secondary to aspiration.")

    def test_interpretar_exige_evidencia_literal_e_tipo_admissivel(self):
        from clinical_kg.llm.anotador import descrever_par, interpretar
        par = descrever_par(self.par_causal, self.texto)
        ok = '{"relacao": "CAUSED_BY", "evidencia": "pneumonia secondary to aspiration"}'
        self.assertEqual(interpretar("Answer: " + ok, par), "CAUSED_BY")
        self.assertIsNone(interpretar('{"relacao": "CAUSED_BY", "evidencia": "due to smoking"}', par))
        self.assertIsNone(interpretar('{"relacao": "LOCATED_IN", "evidencia": "pneumonia"}', par))
        self.assertIsNone(interpretar("not json at all", par))
        self.assertEqual(interpretar('{"relacao": "NENHUMA", "evidencia": ""}', par), NENHUMA)

    def test_autoconsistencia_e_cache(self):
        from clinical_kg.llm.anotador import Anotador, descrever_par
        from clinical_kg.llm.provedor import ProvedorCache, ProvedorFalso, RespostaAusente
        respostas = iter(['{"relacao": "CAUSED_BY", "evidencia": "secondary to aspiration"}'] * 3
                         + ['{"relacao": "COORDINATE_WITH", "evidencia": "pneumonia"}',
                            '{"relacao": "CAUSED_BY", "evidencia": "invented"}'])
        provedor = ProvedorFalso(lambda *a: next(respostas), cache=self.cache)
        par = descrever_par(self.par_causal, self.texto)
        voto = Anotador(provedor, k=5).anotar(par)
        self.assertEqual(voto["relacao"], "CAUSED_BY")
        self.assertEqual(voto["validas"], 4)                     # a evidência inventada foi descartada
        self.assertAlmostEqual(voto["confianca"], 0.75)
        self.assertEqual(provedor.chamadas, 5)
        # A segunda vez sai do cache, inclusive para um provedor sem modelo.
        de_novo = Anotador(ProvedorCache("falso", self.cache), k=5).anotar(par)
        self.assertEqual(de_novo, voto)
        with self.assertRaises(RespostaAusente):
            Anotador(ProvedorCache("outro-modelo", self.cache), k=1).anotar(par)

    def test_cascata_so_consulta_onde_discordam(self):
        from clinical_kg.llm.fontes import prever_cascata
        es = [exemplo("c", (i,)) for i in range(3)]
        p1 = [(es[0], "LOCATED_IN"), (es[1], NENHUMA), (es[2], "CAUSED_BY")]
        p2 = [(es[0], "LOCATED_IN"), (es[1], "LOCATED_IN"), (es[2], NENHUMA)]
        votos = {(1,): {"relacao": NENHUMA}, (2,): {"relacao": "TREATED_WITH"}}
        saida, consultas = prever_cascata(p1, p2, votos)
        self.assertEqual(consultas, 2)
        self.assertEqual([r for _, r in saida], ["LOCATED_IN", NENHUMA, NENHUMA])


class HibridoTests(unittest.TestCase):
    def test_texto_do_par(self):
        from clinical_kg.hibrido.embeddings import texto_do_par
        t = "History. Pain in the abdomen."
        self.assertEqual(texto_do_par(t, ("c", 9, 13, 21, 28)), "Pain in the abdomen")
        self.assertEqual(texto_do_par(t, ("c", -1, -1, 21, 28)), "Pain in the abdomen")

    def test_fonte_empilhada_entra_nos_atributos(self):
        from clinical_kg.hibrido.transferencia import ModeloHibrido

        class Fonte:
            def ajustar(self, treino):
                self.vistos = len(treino)
                return self

            def atributo_treino(self, e):
                return {"pista": 1.0 if e.relacao == "LOCATED_IN" else 0.0}

            def atributo_teste(self, e):
                return {"pista": 1.0 if e.cauda == "abdomen" else 0.0}

        treino = [exemplo(f"c{i % 3}", (i,), relacao="LOCATED_IN" if i % 2 else NENHUMA,
                          cau="abdomen" if i % 2 else "x") for i in range(30)]
        modelo = ModeloHibrido(usar_pmi=False, usar_lm=False).ajustar(treino, [], fontes=[Fonte()])
        self.assertEqual(modelo.fontes[0].vistos, 30)
        self.assertEqual(modelo.prever_exemplo(exemplo("t", ("t",), cau="abdomen")), "LOCATED_IN")
        self.assertEqual(modelo.prever_exemplo(exemplo("t", ("t2",), cau="x")), NENHUMA)


def corpus_sintetico():
    """Três diagnósticos com perfis distintos, em 12 pacientes."""
    from clinical_kg.paciente.corpus import Caso, Corpus
    nomes = {"D1": "acute pancreatitis", "D2": "cholecystitis", "D3": "appendicitis",
             "S1": "epigastric pain", "S2": "vomiting", "S3": "fever", "S4": "right lower quadrant pain",
             "F1": "jaundice", "E1": "lipase", "E2": "ultrasonography", "T1": "cholecystectomy",
             "T2": "appendectomy", "X1": "patients"}
    categorias = {"S1": "Symptom", "S2": "Symptom", "S3": "Symptom", "S4": "Symptom", "F1": "Finding",
                  "E1": "Exam", "E2": "Exam", "T1": "Treatment", "T2": "Treatment", "X1": "Finding",
                  "D1": "Diagnosis", "D2": "Diagnosis", "D3": "Diagnosis"}
    fatos_por_dx = {"D1": ["S1", "S2", "E1"], "D2": ["S3", "F1", "E2", "T1"], "D3": ["S4", "S3", "S2", "T2"]}
    casos = {}
    for k in range(12):
        dx = ["D1", "D2", "D3"][k % 3]
        cid = f"caso{k}"
        ents = [{"cui": c, "entity_type": categorias[c], "assertion_status": "affirmed",
                 "surface_text": nomes[c]} for c in fatos_por_dx[dx] + ["X1"]]
        rels = [{"relation": "HAS_DIAGNOSIS", "assertion_status": "affirmed", "head_cui": "",
                 "tail_cui": dx}]
        if dx == "D2":
            rels.append({"relation": "HAS_FINDING", "assertion_status": "affirmed", "head_cui": "",
                         "tail_cui": "F1"})
        if dx == "D1":
            rels.append({"relation": "REVEALED_BY", "assertion_status": "affirmed", "head_cui": "E1",
                         "tail_cui": "S1"})
        meds = [{"linked_entity_cui": "E1", "surface_text": "1572 U/L", "unit_category": "lab_concentration"}] \
            if dx == "D1" else []
        casos[cid] = Caso(cid, str(30 + k), "Female" if k % 2 else "Male", ents, meds, rels)
    # Pacientes sem nenhum desses diagnósticos: num corpus real, cada fato aparece numa fração
    # pequena dos casos; só X1 ("patients") aparece em todos e deve sair como genérico.
    for k in range(24):
        cid = f"outro{k}"
        casos[cid] = Caso(cid, "50", "Male", [{"cui": "X1", "entity_type": "Finding",
                                              "assertion_status": "affirmed", "surface_text": "patients"}])
    tuis = {c: "T047" for c in ("D1", "D2", "D3")}
    return Corpus("sintetico", casos, nomes, categorias, tuis)


class PacienteTests(unittest.TestCase):
    def setUp(self):
        from clinical_kg.paciente.perfil import montar_perfis
        self.corpus = corpus_sintetico()
        self.perfis = montar_perfis(self.corpus, minimo_casos=2)

    def test_perfis_e_genericos(self):
        self.assertEqual(set(self.perfis), {"D1", "D2", "D3"})
        p = self.perfis["D1"]
        self.assertEqual(len(p.casos), 4)
        self.assertNotIn("X1", p.fatos)                        # em todos os casos: genérico
        self.assertEqual(p.fatos["E1"].revela["S1"], 4)
        self.assertEqual(p.fatos["E1"].valores[0][1], "1572 U/L")

    def test_ficha_deterministica_e_negados_dos_concorrentes(self):
        from clinical_kg.paciente.ficha import montar_ficha
        a = montar_ficha(self.perfis["D1"], self.perfis, "dificil", 7, self.corpus.nomes)
        b = montar_ficha(self.perfis["D1"], self.perfis, "dificil", 7, self.corpus.nomes)
        self.assertEqual(a.para_dict(), b.para_dict())
        negados = {i.cui for i in a.negados()}
        self.assertTrue(negados)
        self.assertFalse(negados & set(self.perfis["D1"].fatos))
        lipase = next(i for i in a.itens if i.cui == "E1")
        self.assertEqual(lipase.resultado, ["epigastric pain"])
        self.assertEqual(lipase.valores, ["1572 U/L"])
        self.assertFalse(montar_ficha(self.perfis["D1"], self.perfis, "facil", 7, self.corpus.nomes).negados())

    def test_narrador_e_dialogo_so_usam_a_ficha(self):
        from clinical_kg.paciente.dialogo import Atendimento
        from clinical_kg.paciente.ficha import montar_ficha
        from clinical_kg.paciente.narrador import NarradorModelo
        f = montar_ficha(self.perfis["D1"], self.perfis, "medio", 1, self.corpus.nomes)
        texto = NarradorModelo().completo(f)
        for i in f.itens:
            self.assertIn(i.nome, texto.lower() if i.nome.islower() else texto)
        self.assertIn("acute pancreatitis", texto)
        self.assertNotIn("acute pancreatitis", NarradorModelo().apresentacao(f))
        at = Atendimento(f)
        self.assertTrue(at.perguntar("Do you have epigastric pain?").texto.startswith("Yes"))
        self.assertIn("haven't noticed", at.perguntar("Any headache?").texto)
        negado = f.negados()[0]
        self.assertTrue(at.perguntar(f"Do you have {negado.nome}?").texto.startswith("No, I don't"))
        self.assertIn(negado.cui, at.negados_revelados())
        self.assertIn("no abnormality", at.pedir_exame("colonoscopy").texto)

    def test_feedback_diferencial(self):
        from clinical_kg.paciente.dialogo import Atendimento
        from clinical_kg.paciente.feedback import avaliar, diferencial, subgrafo
        from clinical_kg.paciente.ficha import montar_ficha
        self.assertEqual(diferencial({"S4", "S3"}, set(), self.perfis)[0]["cui"], "D3")
        self.assertEqual(diferencial({"F1"}, set(), self.perfis)[0]["cui"], "D2")
        f = montar_ficha(self.perfis["D2"], self.perfis, "facil", 0, self.corpus.nomes)
        at = Atendimento(f)
        at.queixa_principal()
        r = avaliar(f, at, "D3", self.perfis)
        self.assertFalse(r["correto"])
        self.assertIn("jaundice", [p["nome"] for p in r["perdidos"]])
        g = subgrafo(f, at, self.perfis, r["diferencial"])
        ids = {n["id"] for n in g["nodes"]}
        self.assertTrue(all(e["source"] in ids and e["target"] in ids for e in g["edges"]))

    def test_filtros_de_qualidade(self):
        from clinical_kg.paciente.corpus import exames_de_laboratorio, limpar_nome
        from clinical_kg.paciente.perfil import fato_util
        self.assertEqual(limpar_nome("a hemoglobin level"), "hemoglobin level")
        self.assertEqual(limpar_nome("magnetic resonance cholangiopancreatography (mrcp"),
                         "magnetic resonance cholangiopancreatography")
        for ruim in ("5 mm", "day 7", "detected", "past medical history", "decreasing", "severely"):
            self.assertFalse(fato_util(ruim), ruim)
        for bom in ("jaundice", "bleeding", "weight loss", "nausea and vomiting"):
            self.assertTrue(fato_util(bom), bom)
        unidades = {"lipase": collections.Counter(lab_concentration=9, time=1),
                    "prednisone": collections.Counter(mass=8, lab_concentration=1)}
        self.assertEqual(exames_de_laboratorio(unidades), {"lipase"})

    def test_ida_volta_relacoes_esperadas(self):
        from clinical_kg.paciente.ficha import montar_ficha
        from clinical_kg.paciente.ida_volta import PACIENTE, relacoes_esperadas
        f = montar_ficha(self.perfis["D1"], self.perfis, "facil", 0, self.corpus.nomes)
        esperadas = relacoes_esperadas(f)
        self.assertIn((frozenset((PACIENTE, "D1")), "HAS_DIAGNOSIS"), esperadas)
        self.assertIn((frozenset(("E1", "S1")), "REVEALED_BY"), esperadas)


try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipUnless(AppTest is not None and (ROOT / "data/processed/app/multicare50/entidades.csv").exists(),
                     "Requer Streamlit e um corpus da aplicação já construído")
class AppProjeto3Tests(unittest.TestCase):
    def test_atendimento_completo(self):
        at = AppTest.from_file(str(ROOT / "src/clinical_kg/app/projeto3.py"), default_timeout=180).run()
        self.assertFalse(at.exception)
        self.assertEqual([t.label for t in at.tabs], ["Atendimento", "Feedback", "Ficha (professor)", "Resultados do P3"])
        ficha = at.session_state["ficha"]

        def botao(rotulo):
            return next(b for b in at.button if b.label == rotulo)

        botao("Queixa principal").click().run()
        botao("Exame físico").click().run()
        exame = next(s for s in at.selectbox if s.label == "Pedir exame")
        exame.set_value(exame.options[0]).run()
        botao("Pedir").click().run()
        next(s for s in at.selectbox if s.label == "Diagnóstico").set_value(ficha.dx_cui).run()
        botao("Propor diagnóstico").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(at.session_state["resultado"]["correto"])
        self.assertGreaterEqual(len(at.chat_message), 6)


if __name__ == "__main__":
    unittest.main()
