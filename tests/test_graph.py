"""Run with python3 -m unittest discover -s tests (stdlib only)."""

import copy
import csv
import io
import json
import sqlite3
import sys
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from clinical_kg.graph.model import build_graph, filter_graph, group_graph, interaction_views, validate_graph
from clinical_kg.extraction.measurements import analyze_case, link_measurement
from clinical_kg.extraction.entities import extract
from clinical_kg.graph.export import to_json, to_graphml, to_csv_zip, GRAPHML_NS
from clinical_kg.gazetteer.build import create_schema


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.text = "Lipase 850 U/L. Lipase 20 U/L. 3-day"
        self.meta = {"case_id": "test", "article_id": "a", "age": "0", "gender": "Unknown"}
        self.entities = [self.entity(0, 6), self.entity(16, 22),
                         self.entity(31, 36, cui="C2", kind="Finding")]
        self.measurements = []
        self.entities[2]["surface_text"] = "3-day"
        for surface, ent in [("850 U/L", self.entities[0]), ("20 U/L", self.entities[1])]:
            start = self.text.index(surface)
            self.measurements.append({"start": start, "end": start+len(surface),
                                      "surface_text": surface, "value": surface.split()[0],
                                      "unit": "U/L", "link_method": "same_sentence",
                                      "linked_entity_start": ent["start"],
                                      "linked_entity_end": ent["end"], "linked_entity_cui": "C1"})
        self.measurements.append({"start": 31, "end": 36, "surface_text": "3-day",
                                  "value": "3", "unit": "day", "link_method": "none"})

    def entity(self, start, end, cui="C1", kind="Exam"):
        return {"start": start, "end": end, "surface_text": "Lipase",
                "cui": cui, "tui": "T059", "entity_type": kind,
                "preferred_term": "Lipase", "vocabulary": "LNC", "tty": "PT"}

    def build(self, **kwargs):
        return build_graph(self.meta, self.text, self.entities, self.measurements, **kwargs)

    def test_on_demand_measurements_and_exports(self):
        graph = filter_graph(self.build(), entity_types=["Exam", "Finding"], link_methods=["same_sentence", "none"])
        views = interaction_views(graph, on_demand=True, max_nodes=100)
        self.assertFalse(any(n["kind"] == "Measurement" for n in views[""]["nodes"]))
        focused = views["concept:C1"]
        measured = [n for n in focused["nodes"] if n["kind"] == "Measurement"]
        self.assertEqual(len(measured), 2)
        self.assertIs(views[measured[0]["id"]], focused)
        person = next(n["id"] for n in graph["nodes"] if n["kind"] == "Person")
        self.assertEqual([n["link_method"] for n in views[person]["nodes"] if n["kind"] == "Measurement"], ["none"])
        self.assertEqual(len(json.loads(to_json(focused))["nodes"]), len(focused["nodes"]))
        for view in interaction_views(graph, on_demand=True, max_nodes=4).values():
            validate_graph(view)
            self.assertLessEqual(len(view["nodes"]), 4)
        occurrences = filter_graph(self.build(aggregate=False), entity_types=["Exam", "Finding"], link_methods=["same_sentence"])
        occurrence_views = interaction_views(occurrences, on_demand=True, max_nodes=100)
        for node in occurrences["nodes"]:
            if node.get("cui") == "C1":
                self.assertEqual(sum(n["kind"] == "Measurement" for n in occurrence_views[node["id"]]["nodes"]), 1)

    def test_is_a_multitype_and_occurrence_views(self):
        self.entities[1]["entity_type"] = "Treatment"
        for aggregate in (True, False):
            graph = group_graph(self.build(aggregate=aggregate), ["Exam", "Treatment", "Finding"])
            validate_graph(graph)
            edges = [e for e in graph["edges"] if e["relation"] == "IS_A"]
            self.assertEqual(len(edges), 3)
            self.assertEqual({e["target"] for e in edges}, {"type:Exam", "type:Treatment", "type:Finding"})
            filtered = filter_graph(graph, entity_types=["Exam"], link_methods=["same_sentence"])
            grouped = group_graph(filtered, ["Exam"])
            self.assertEqual([n["label"] for n in grouped["nodes"] if n["kind"] == "SemanticType"], ["Exam"])
            for view in interaction_views(grouped, on_demand=True, max_nodes=20).values():
                validate_graph(view)

    def test_aggregation_preserves_occurrences_and_measurements(self):
        graph = self.build()
        concept = next(n for n in graph["nodes"] if n["id"] == "concept:C1")
        self.assertEqual(concept["count"], 2)
        self.assertEqual([e["start"] for e in concept["evidence"]], [0, 16])
        edges = [e for e in graph["edges"] if e["relation"] == "ASSOCIATED_WITH_MEASUREMENT"]
        self.assertEqual(len(edges), 2)
        self.assertEqual({tuple(e["linked_entity_span"]) for e in edges}, {(0, 6), (16, 22)})
        # Same-span Finding and Measurement must remain separate nodes.
        self.assertEqual(sum(n["kind"] == "Measurement" for n in graph["nodes"]), 3)
        self.assertTrue(any(n["id"] == "concept:C2" for n in graph["nodes"]))
        self.assertEqual(next(n for n in graph["nodes"] if n["kind"] == "Person")["age"], "0")

    def test_ids_and_output_are_deterministic(self):
        graph = self.build()
        other = build_graph(self.meta, self.text, self.entities[::-1], self.measurements[::-1])
        self.assertEqual(graph, other)
        self.assertEqual(len({n["id"] for n in graph["nodes"]}), len(graph["nodes"]))

    def test_occurrence_mode_attaches_to_exact_mention(self):
        graph = self.build(aggregate=False)
        edges = [e for e in graph["edges"] if e["relation"] == "ASSOCIATED_WITH_MEASUREMENT"]
        self.assertEqual({e["source"] for e in edges}, {"mention:test:0:6:C1", "mention:test:16:22:C1"})

    def test_missing_link_is_retained_but_not_guessed_by_cui(self):
        self.measurements[0]["linked_entity_start"] = 999
        graph = self.build()
        self.assertEqual(len(graph["warnings"]), 1)
        self.assertEqual(sum(e["relation"] == "CONTAINS_UNLINKED_MEASUREMENT" for e in graph["edges"]), 2)

    def test_legacy_csv_link_and_boolean(self):
        row = self.measurements[0]
        row.pop("linked_entity_cui")
        row["start"], row["end"], row["is_range"] = str(row["start"]), str(row["end"]), "False"
        graph = self.build()
        self.assertFalse(graph["warnings"])
        self.assertFalse(next(n for n in graph["nodes"] if n["kind"] == "Measurement")["is_range"])

    def test_invalid_provenance_fails(self):
        for field, value in [("surface_text", "Wrong"), ("case_id", "another")]:
            with self.subTest(field=field):
                entities = copy.deepcopy(self.entities)
                entities[0][field] = value
                with self.assertRaises(ValueError):
                    build_graph(self.meta, self.text, entities, self.measurements)
        self.measurements[0]["linked_entity_cui"] = "C2"
        with self.assertRaises(ValueError):
            self.build()

    def test_filters_and_cap_never_leave_dangling_edges(self):
        graph = self.build()
        for limit in (2, 3, 4, 100):
            view = filter_graph(graph, entity_types=["Exam"], link_methods=["same_sentence"], max_nodes=limit)
            validate_graph(view)
            self.assertLessEqual(len(view["nodes"]), limit)
            self.assertFalse(any(n.get("entity_type") == "Finding" for n in view["nodes"]))
            self.assertFalse(any(n.get("link_method") == "none" for n in view["nodes"]))
        view = filter_graph(graph, entity_types=[], link_methods=["same_sentence", "none"])
        self.assertEqual(len(view["nodes"]), 3)  # person, known age, unlinked measurement

    def test_empty_and_article(self):
        graph = build_graph(self.meta, "", [], [], article={"article_id": "a", "title": "Article"})
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertTrue(any(e["relation"] == "REPORTS_PERSON" for e in graph["edges"]))

    def test_shared_pipeline_without_database(self):
        entities, measurements = analyze_case("Lipase 850 U/L", None)
        self.assertEqual(entities, [])
        self.assertEqual(measurements[0]["link_method"], "none")
        self.assertEqual(measurements[0]["linked_entity_cui"], "")

    def test_link_contract_includes_cui_and_offsets(self):
        link = link_measurement({"start": 7, "end": 14}, self.entities, [0])
        self.assertEqual(link["linked_entity_id"], "concept:C1")
        self.assertEqual(link["linked_entity_start"], 0)

    def test_preferred_term_and_live_pipeline(self):
        with sqlite3.connect(":memory:") as con:
            create_schema(con)
            con.execute("INSERT INTO terms VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        ("lipase", "lipase", 1, "C1", "T059", "Lipase assay", "LNC", 0, 0, "PT"))
            entities, measurements = analyze_case("lipase 850 U/L", con)
            self.assertEqual(entities[0]["preferred_term"], "Lipase assay")
            self.assertEqual(measurements[0]["linked_entity_cui"], entities[0]["cui"])
            self.assertEqual(entities, extract("lipase 850 U/L", con, 6))

    def test_exports_preserve_evidence_and_escaped_labels(self):
        self.entities[0]["preferred_term"] = '<test & "label">'
        graph = self.build()
        self.assertEqual(json.loads(to_json(graph)), graph)
        root = ET.fromstring(to_graphml(graph))
        ns = {"g": GRAPHML_NS}
        keys = {k.attrib["id"]: k.attrib["attr.name"] for k in root.findall("g:key", ns)}
        nodes = root.findall("g:graph/g:node", ns)
        self.assertEqual(len(nodes), len(graph["nodes"]))
        concept = next(n for n in nodes if n.attrib["id"] == "concept:C1")
        attributes = {keys[d.attrib["key"]]: d.text for d in concept}
        self.assertEqual(json.loads(attributes["evidence"]), next(n for n in graph["nodes"]
                                                                if n["id"] == "concept:C1")["evidence"])
        archive_bytes = to_csv_zip(graph)
        self.assertEqual(archive_bytes, to_csv_zip(graph))
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            rows = list(csv.DictReader(io.StringIO(archive.read("nodes.csv").decode())))
            self.assertEqual(len(rows), len(nodes))
            self.assertEqual(json.loads(archive.read("metadata.json"))["text_sha256"], graph["text_sha256"])

    def test_lower_confidence_is_preserved(self):
        self.measurements[0]["link_method"] = "window_fallback"
        graph = self.build()
        edge = next(e for e in graph["edges"] if e.get("link_method") == "window_fallback")
        self.assertTrue(edge["lower_confidence"])
        self.assertTrue(edge["heuristic"])


if __name__ == "__main__":
    unittest.main()
