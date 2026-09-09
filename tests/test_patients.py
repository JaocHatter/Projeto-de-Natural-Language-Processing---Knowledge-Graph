"""Regression coverage for the cleaned-patient ontology and data source."""

import csv
import sqlite3
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from clinical_kg.graph.model import build_graph, filter_graph, validate_graph, MENTION_RELATIONS
from clinical_kg.corpus.patients import new_patient, validate_annotations
from clinical_kg.paths import DEFAULT_CASES, DEFAULT_DB
from clinical_kg.extraction.measurements import analyze_case


class PatientTests(unittest.TestCase):
    def test_newborn_and_unknown_demographics(self):
        text = "A 39-week-old newborn girl presented with fever."
        meta = new_patient(text)
        graph = build_graph(meta, text, [], [])
        age = next(n for n in graph["nodes"] if n["kind"] == "Age")
        self.assertEqual(age["value"], "0")
        self.assertEqual(age["unit"], "years")
        self.assertEqual(age["provenance"]["method"], "newborn-keyword")
        self.assertEqual({e["relation"] for e in graph["edges"]}, {"HAS_AGE", "HAS_SEX"})
        unknown = build_graph(new_patient("xyzzy"), "xyzzy", [], [])
        self.assertEqual([n["kind"] for n in unknown["nodes"]], ["Person"])

    def test_merged_identity_and_upstream_audit(self):
        meta = {"case_id": "merged_P1", "source_case_ids": "fragment_01|fragment_02", "age": "0",
                "age_upstream": "39", "age_method": "newborn-keyword", "gender": "Female"}
        graph = build_graph(meta, "text", [], [])
        people = [n for n in graph["nodes"] if n["kind"] == "Person"]
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["id"], "person:merged_P1")
        self.assertEqual(people[0]["source_case_ids"], ["fragment_01", "fragment_02"])
        age = next(n for n in graph["nodes"] if n["kind"] == "Age")
        self.assertEqual(age["value"], "0")
        self.assertEqual(age["provenance"]["upstream_value"], "39")

    def test_every_clinical_category_has_typed_relation(self):
        text = " ".join(MENTION_RELATIONS)
        entities = [{"start": text.index(kind), "end": text.index(kind)+len(kind),
                     "surface_text": kind, "cui": f"C{i}", "entity_type": kind}
                    for i, kind in enumerate(MENTION_RELATIONS)]
        graph = build_graph({"case_id": "test"}, text, entities, [])
        self.assertEqual({e["relation"] for e in graph["edges"]}, set(MENTION_RELATIONS.values()))
        self.assertTrue(all(e["assertion_status"] == "not_assessed" for e in graph["edges"]))

    def test_stale_raw_annotations_fail(self):
        cases = [{"case_id": "merged_P1", "case_text": "pain"}]
        with self.assertRaisesRegex(ValueError, "outside the selected corpus"):
            validate_annotations(cases, [{"case_id": "fragment_01"}], "measurements.csv")
        with self.assertRaisesRegex(ValueError, "offsets do not match"):
            validate_annotations(cases, [{"case_id": "merged_P1", "start": "1", "end": "3",
                                           "surface_text": "pain"}], "entities.csv")

    @unittest.skipUnless(DEFAULT_CASES.exists() and DEFAULT_DB.exists(), "Requires local corpus and gazetteer")
    def test_all_cleaned_patients_and_merged_links(self):
        with DEFAULT_CASES.open(newline="", encoding="utf-8") as f:
            cases = list(csv.DictReader(f))
        self.assertEqual(len(cases), 50)
        ids = {c["case_id"] for c in cases}
        self.assertTrue({"PMC6083636_P1", "PMC11259348_P1"} <= ids)
        self.assertFalse({"PMC9815523_01", "PMC10710131_01", "PMC9650410_01", "PMC6083636_02"} & ids)
        totals = Counter()
        con = sqlite3.connect(DEFAULT_DB.as_uri()+"?mode=ro", uri=True)
        try:
            for case in cases:
                entities, measurements = analyze_case(case["case_text"], con)
                totals["entities"] += len(entities)
                totals["measurements"] += len(measurements)
                for aggregate in (True, False):
                    graph = build_graph(case, case["case_text"], entities, measurements, aggregate=aggregate)
                    self.assertFalse(graph["warnings"], case["case_id"])
                    self.assertEqual(sum(n["kind"] == "Person" for n in graph["nodes"]), 1)
                    self.assertEqual(sum(n["kind"] == "Measurement" for n in graph["nodes"]), len(measurements))
                    validate_graph(filter_graph(graph, entity_types=["Exam"], link_methods=["same_sentence"], max_nodes=20))
                    if case["case_id"] == "PMC6083636_P1":
                        # Later merged fragments use offsets into concatenated text.
                        self.assertTrue(any(n["evidence"] and n["evidence"][0]["start"] > 3000
                                            for n in graph["nodes"]))
            self.assertEqual(totals["entities"], 3134)
            self.assertEqual(totals["measurements"], 535)
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
