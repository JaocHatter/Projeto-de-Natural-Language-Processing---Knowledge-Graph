"""Run with python3 -m unittest discover -s tests (stdlib only)."""

import sys
import unittest
from pathlib import Path

from clinical_kg.relations import features as rf
from clinical_kg.relations.crf import (NEG_INF, decode_triggers, emissions,
                                       transition, viterbi)
from clinical_kg.relations.extract import (analyze_case, analyze_sentence,
                                           assertion_states,
                                           coordinate_siblings)
from clinical_kg.relations.pos import tag_sequence, tag_token
from clinical_kg.relations.text_layer import (clause_spans, segment,
                                              sentence_spans, token_spans)


def entity(text, surface, kind, cui="C1"):
    start = text.index(surface)
    return {"start": start, "end": start + len(surface), "surface_text": surface,
            "entity_type": kind, "cui": cui, "tui": "T000"}


class TextLayerTests(unittest.TestCase):
    def test_decimal_never_splits(self):
        text = "Pressure was 105/51 mmHg (1 mm Hg = 0.133 kPa). The next sentence."
        self.assertEqual(len(sentence_spans(text)), 2)

    def test_figure_abbreviation_never_splits(self):
        text = "Imaging was done (Fig. 1) and showed a mass. Then he improved."
        spans = sentence_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertIn("mass", text[spans[0][0]:spans[0][1]])

    def test_closing_quote_still_splits(self):
        text = 'She reported "chest pain for two weeks." She then attended the ER.'
        self.assertEqual(len(sentence_spans(text)), 2)

    def test_offsets_index_original_text(self):
        text = "A 44-year-old female presented. Fig. 2 showed a 0.5 cm lesion."
        for start, end in sentence_spans(text):
            self.assertEqual(text[start:end], text[start:end])
            self.assertTrue(text[start:end].strip())
        for token, start, end in token_spans(text):
            self.assertEqual(text[start:end], token)

    def test_clauses_exclude_their_separators(self):
        text = "He had fever and cough, then improved."
        clauses = clause_spans(text, 0, len(text))
        self.assertTrue(all(text[s:e].strip() not in ("and", "then", ",")
                            for s, e in clauses))
        self.assertGreaterEqual(len(clauses), 3)


class PosTests(unittest.TestCase):
    def test_plural_noun_is_not_a_verb(self):
        # The whole reason the lexicon exists: suffix rules alone read
        # "areas of", "episodes of", "doses of" as VERB+PREP.
        for word in ("areas", "episodes", "months", "doses", "features", "days"):
            self.assertEqual(tag_token(word), "NOUN", word)

    def test_content_verbs_are_verbs(self):
        for word in ("treated", "revealed", "showed", "underwent", "performed"):
            self.assertEqual(tag_token(word), "VERB", word)

    def test_ing_is_a_noun_unless_under_an_auxiliary(self):
        self.assertEqual(tag_token("bleeding"), "NOUN")
        self.assertEqual(tag_token("showing", prev_tag="AUX"), "VERB")

    def test_prepositions_beat_suffix_rules(self):
        for word in ("during", "following", "including", "according"):
            self.assertEqual(tag_token(word), "PREP", word)


class CrfTests(unittest.TestCase):
    def test_bio_validity_o_to_i_is_impossible(self):
        self.assertEqual(transition("O", "I-TRIG", 0), NEG_INF)
        self.assertEqual(transition("NEG", "I-TRIG", 0), NEG_INF)
        self.assertGreater(transition("B-TRIG", "I-TRIG", 1), 0)

    def test_trigger_length_is_capped(self):
        cap = rf.WEIGHTS["trans.max_trigger_len"]
        self.assertEqual(transition("I-TRIG", "I-TRIG", cap), NEG_INF)

    def test_viterbi_never_emits_an_invalid_sequence(self):
        text = "The patient was treated with aspirin and improved after surgery."
        sent = segment(text)[0]
        tokens = [t[0] for t in sent["tokens"]]
        emit, _ = emissions(tokens, tag_sequence(tokens),
                            [False] * len(tokens), {})
        tags = viterbi(emit)
        self.assertEqual(len(tags), len(tokens))
        for prev, cur in zip(tags, tags[1:]):
            if cur == "I-TRIG":
                self.assertIn(prev, ("B-TRIG", "I-TRIG"))

    def test_multi_token_trigger_survives_decoding(self):
        text = "She presented with severe chest pain."
        sent = segment(text)[0]
        tokens = [t[0] for t in sent["tokens"]]
        emit, _ = emissions(tokens, tag_sequence(tokens), [False] * len(tokens), {})
        triggers = decode_triggers(tokens, sent["tokens"], viterbi(emit))
        self.assertIn(["presented", "with"], [t["words"] for t in triggers])

    def test_a_noun_is_never_a_trigger_head(self):
        # "presented to the emergency department with pain" must not decode
        # "department with" as a trigger just to collect the B->I bonus.
        text = "He presented to the emergency department with abdominal pain."
        sent = segment(text)[0]
        tokens = [t[0] for t in sent["tokens"]]
        emit, _ = emissions(tokens, tag_sequence(tokens), [False] * len(tokens), {})
        triggers = decode_triggers(tokens, sent["tokens"], viterbi(emit))
        self.assertNotIn("department", [t["words"][0] for t in triggers])


class AssertionTests(unittest.TestCase):
    def states_for(self, text):
        """(token, status) pairs, in order -- a dict would collapse repeats."""
        sent = segment(text)[0]
        tokens = [t[0] for t in sent["tokens"]]
        states = assertion_states(tokens, sent["tokens"], sent["clauses"], text)
        return list(zip([t.lower() for t in tokens], states))

    def first(self, text, token):
        return next(s for t, s in self.states_for(text) if t == token)

    def test_negation_scope_terminates_at_but(self):
        # "tenderness" appears on both sides of the "but": the first is
        # asserted, the second is inside the negation scope.
        text = ("Examination revealed tenderness in the RIF but no rebound "
                "tenderness was found.")
        states = self.states_for(text)
        occurrences = [s for t, s in states if t == "tenderness"]
        self.assertEqual(occurrences, ["affirmed", "negated"])
        self.assertEqual(self.first(text, "rebound"), "negated")

    def test_history_is_historical_not_affirmed(self):
        text = "A female with a history of cesarean section presented."
        self.assertEqual(self.first(text, "cesarean"), "historical")

    def test_family_history_outranks_negation(self):
        text = "Family history of diabetes was noted."
        self.assertEqual(self.first(text, "diabetes"), "family")

    def test_hedging_is_detected(self):
        text = "Findings were consistent with pulmonary adenocarcinoma."
        self.assertEqual(self.first(text, "adenocarcinoma"), "hedged")


class CoordinationTests(unittest.TestCase):
    def test_comma_list_becomes_siblings(self):
        # The original plan discarded any pair with a comma between it; this
        # is the pattern that rule destroys.
        text = ("He presented with fatigue, swollen abdomen, decreased appetite, "
                "and weight loss.")
        ents = [entity(text, s, "Symptom") for s in
                ("fatigue", "swollen abdomen", "decreased appetite", "weight loss")]
        head_of = coordinate_siblings(ents, text)
        self.assertEqual(set(head_of), {1, 2, 3})
        self.assertEqual(set(head_of.values()), {0})

    def test_siblings_inherit_the_head_relation(self):
        text = ("The patient presented with fatigue, swollen abdomen, decreased "
                "appetite, and weight loss.")
        ents = [entity(text, s, "Symptom") for s in
                ("fatigue", "swollen abdomen", "decreased appetite", "weight loss")]
        rels = analyze_sentence(text, segment(text)[0], ents, {})
        symptoms = {r["tail"]["surface_text"] for r in rels
                    if r["relation"] == "HAS_SYMPTOM"}
        self.assertEqual(len(symptoms), 4, symptoms)

    def test_different_types_do_not_coordinate(self):
        text = "He underwent surgery and aspirin."
        ents = [entity(text, "surgery", "Treatment"), entity(text, "aspirin", "Diagnosis")]
        self.assertEqual(coordinate_siblings(ents, text), {})


class OrientationTests(unittest.TestCase):
    def test_exam_is_the_head_of_revealed_by(self):
        head = {"entity_type": "BodyPart"}
        tail = {"entity_type": "Exam"}
        new_head, new_tail, relation, swapped = rf.orient(head, tail, "REVEALED_BY")
        self.assertTrue(swapped)
        self.assertIs(new_head, tail)
        self.assertIs(new_tail, head)

    def test_correct_orientation_is_left_alone(self):
        head = {"entity_type": "Exam"}
        tail = {"entity_type": "Finding"}
        _, _, _, swapped = rf.orient(head, tail, "REVEALED_BY")
        self.assertFalse(swapped)

    def test_person_head_is_never_swapped(self):
        _, _, _, swapped = rf.orient(None, {"entity_type": "Exam"}, "REVEALED_BY")
        self.assertFalse(swapped)


class EndToEndTests(unittest.TestCase):
    def test_offsets_round_trip_and_fields_are_valid(self):
        text = ("A 44-year-old female presented with chest pain. Echocardiography "
                "showed a mildly thickened septum but no pericardial effusion.")
        ents = [entity(text, "chest pain", "Symptom"),
                entity(text, "Echocardiography", "Exam"),
                entity(text, "septum", "BodyPart"),
                entity(text, "pericardial effusion", "Finding")]
        rels = analyze_case(text, ents, {})
        self.assertTrue(rels)
        for rel in rels:
            self.assertIn(rel["relation"], rf.RELATIONS)
            self.assertIn(rel["assertion_status"], rf.ASSERTIONS)
            tail = rel["tail"]
            self.assertEqual(text[tail["start"]:tail["end"]], tail["surface_text"])
            if rel["head"]:
                head = rel["head"]
                self.assertEqual(text[head["start"]:head["end"]], head["surface_text"])
            self.assertTrue(rel["rule_path"])

    def test_negated_finding_is_marked_not_dropped(self):
        text = ("Echocardiography showed a thickened septum but no pericardial "
                "effusion was seen.")
        ents = [entity(text, "Echocardiography", "Exam"),
                entity(text, "pericardial effusion", "Finding")]
        rels = analyze_case(text, ents, {})
        effusion = [r for r in rels if r["tail"]["surface_text"] == "pericardial effusion"]
        self.assertTrue(effusion)
        self.assertEqual(effusion[0]["assertion_status"], "negated")


if __name__ == "__main__":
    unittest.main()


class GraphIntegrationTests(unittest.TestCase):
    """Typed relations must survive the trip into the knowledge graph."""

    def setUp(self):
        from clinical_kg.graph.model import build_graph
        self.build_graph = build_graph
        self.text = ("Echocardiography showed a thickened septum but no "
                     "pericardial effusion was seen.")
        self.meta = {"case_id": "t1", "article_id": "a", "age": "44",
                     "gender": "Female"}
        self.entities = []
        for surface, kind, cui in (("Echocardiography", "Exam", "C1"),
                                   ("septum", "BodyPart", "C2"),
                                   ("pericardial effusion", "Finding", "C3")):
            start = self.text.index(surface)
            self.entities.append({"start": start, "end": start + len(surface),
                                  "surface_text": surface, "cui": cui,
                                  "tui": "T000", "entity_type": kind,
                                  "vocabulary": "V", "tty": "PT",
                                  "preferred_term": surface})

    def relation(self, head, tail, relation, status):
        head_ent = next(e for e in self.entities if e["surface_text"] == head)
        tail_ent = next(e for e in self.entities if e["surface_text"] == tail)
        return {"case_id": "t1", "relation": relation, "assertion_status": status,
                "head_kind": "Concept", "head_start": head_ent["start"],
                "head_end": head_ent["end"], "tail_start": tail_ent["start"],
                "tail_end": tail_ent["end"], "score": "3.0",
                "trigger_text": "showed", "trigger_category": "REVEAL"}

    def test_typed_edges_reach_the_graph_with_assertion_status(self):
        rels = [self.relation("Echocardiography", "septum", "LOCATED_IN", "affirmed"),
                self.relation("Echocardiography", "pericardial effusion",
                              "REVEALED_BY", "negated")]
        graph = self.build_graph(self.meta, self.text, self.entities, [],
                                 relations=rels)
        typed = {e["relation"]: e for e in graph["edges"]
                 if e["relation"] in ("LOCATED_IN", "REVEALED_BY")}
        self.assertEqual(set(typed), {"LOCATED_IN", "REVEALED_BY"})
        self.assertEqual(typed["REVEALED_BY"]["assertion_status"], "negated")
        self.assertTrue(typed["REVEALED_BY"]["heuristic"])

    def test_person_headed_relation_anchors_to_the_case_node(self):
        tail = self.entities[2]
        rels = [{"case_id": "t1", "relation": "HAS_FINDING",
                 "assertion_status": "negated", "head_kind": "Person",
                 "head_start": -1, "head_end": -1, "tail_start": tail["start"],
                 "tail_end": tail["end"], "score": "2.0"}]
        graph = self.build_graph(self.meta, self.text, self.entities, [],
                                 relations=rels)
        edge = next(e for e in graph["edges"] if e["relation"] == "HAS_FINDING")
        self.assertTrue(edge["source"].startswith("person:"))

    def test_unresolved_span_warns_instead_of_dangling(self):
        rels = [{"case_id": "t1", "relation": "HAS_FINDING",
                 "assertion_status": "affirmed", "head_kind": "Person",
                 "head_start": -1, "head_end": -1,
                 "tail_start": 9999, "tail_end": 10000, "score": "1.0"}]
        graph = self.build_graph(self.meta, self.text, self.entities, [],
                                 relations=rels)
        self.assertTrue(any("unresolved tail" in w for w in graph["warnings"]))

    def test_unknown_relation_is_rejected(self):
        rels = [self.relation("Echocardiography", "septum", "NOT_A_RELATION", "affirmed")]
        with self.assertRaises(ValueError):
            self.build_graph(self.meta, self.text, self.entities, [], relations=rels)

    def test_repeated_relation_aggregates_rather_than_colliding(self):
        rels = [self.relation("Echocardiography", "septum", "LOCATED_IN", "affirmed")] * 2
        graph = self.build_graph(self.meta, self.text, self.entities, [],
                                 relations=rels)
        edge = next(e for e in graph["edges"] if e["relation"] == "LOCATED_IN")
        self.assertEqual(edge["count"], 2)


class CliTests(unittest.TestCase):
    """The CLI must actually dispatch to every stage.

    The previous layout had five thin src/*.py shims; restructuring once
    dropped `import sys` from one of them and every unit test still passed,
    because none of them touched a CLI. These run the real dispatcher in a
    subprocess so that cannot recur.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def run_cli(self, *args):
        import os
        import subprocess
        env = {**os.environ, "PYTHONPATH": str(self.ROOT / "src")}
        return subprocess.run([sys.executable, "-m", "clinical_kg", *args],
                              capture_output=True, text=True, timeout=300,
                              cwd=self.ROOT, env=env)

    def test_dispatcher_lists_every_command(self):
        from clinical_kg.cli import COMMANDS
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in COMMANDS:
            self.assertIn(command, result.stdout)

    def test_every_command_resolves_to_a_real_main(self):
        import importlib
        from clinical_kg.cli import COMMANDS
        for command, (module_name, _) in COMMANDS.items():
            module = importlib.import_module(module_name)
            self.assertTrue(callable(getattr(module, "main", None)), command)

    def test_unknown_command_fails_loudly(self):
        result = self.run_cli("not-a-command")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown command", result.stderr)

    def test_help_never_runs_a_writing_stage(self):
        # `clean-cases --help` once had no argparse and rewrote the corpus.
        corpus = self.ROOT / "data/interim/cases_clean.csv"
        before = corpus.stat().st_mtime_ns if corpus.exists() else None
        result = self.run_cli("clean-cases", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        if before is not None:
            self.assertEqual(corpus.stat().st_mtime_ns, before,
                             "--help must never write the corpus")

    def test_help_usage_names_the_subcommand(self):
        # A usage line without the subcommand is not copy-pasteable.
        result = self.run_cli("extract-relations", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("clinical-kg extract-relations", result.stdout)

    def test_annotate_lists_gold_cases(self):
        result = self.run_cli("annotate-relations", "--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.strip().splitlines()), 9)

    def test_evaluate_reports_a_missing_gold_set(self):
        result = self.run_cli("evaluate-relations", "--gold", "/nonexistent.csv")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no gold set", result.stderr)

    def test_extract_explains_a_single_case(self):
        result = self.run_cli("extract-relations", "--explain", "PMC10106591_01")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-->", result.stdout)
