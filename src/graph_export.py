"""Export evidence-backed graphs as JSON, GraphML, or a CSV ZIP (stdlib only).

Example:
    python3 src/graph_export.py --case-id PMC5137649_01 --out /tmp/case.graphml
"""

import argparse
import csv
import io
import json
import sqlite3
from contextlib import closing
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from graph_builder import build_graph, validate_graph
from project_paths import DEFAULT_CASES, DEFAULT_DB, DEFAULT_METADATA
from extract_measurements import analyze_case
from patient_data import validate_annotations

ROOT = Path(__file__).resolve().parents[1]
GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


def to_json(graph: dict) -> str:
    validate_graph(graph)
    return json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _scalar(value) -> str:
    if isinstance(value, (list, dict)) or value is None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def to_graphml(graph: dict) -> bytes:
    """GraphML attributes are strings; lists/dicts are JSON-encoded losslessly."""
    validate_graph(graph)
    ET.register_namespace("", GRAPHML_NS)
    tag = lambda name: f"{{{GRAPHML_NS}}}{name}"
    root = ET.Element(tag("graphml"))
    metadata = {k: v for k, v in graph.items() if k not in ("nodes", "edges")}
    scopes = {"graph": [metadata], "node": graph["nodes"], "edge": graph["edges"]}
    key_ids = {}
    for scope, rows in scopes.items():
        for name in sorted({k for row in rows for k in row}):
            key_id = f"d{len(key_ids)}"
            key_ids[(scope, name)] = key_id
            ET.SubElement(root, tag("key"), {"id": key_id, "for": scope,
                                            "attr.name": name, "attr.type": "string"})

    def attributes(element, scope, row):
        for name, value in sorted(row.items()):
            ET.SubElement(element, tag("data"), {"key": key_ids[(scope, name)]}).text = _scalar(value)

    element = ET.SubElement(root, tag("graph"), {"id": "case_graph", "edgedefault": "directed"})
    attributes(element, "graph", metadata)
    for node in graph["nodes"]:
        attributes(ET.SubElement(element, tag("node"), {"id": node["id"]}), "node", node)
    for edge in graph["edges"]:
        attributes(ET.SubElement(element, tag("edge"), {k: edge[k] for k in ("id", "source", "target")}),
                   "edge", edge)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _csv(rows: list[dict], leading: list[str]) -> str:
    columns = leading + sorted({k for row in rows for k in row} - set(leading))
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    writer.writerows({k: _scalar(v) for k, v in row.items()} for row in rows)
    return buf.getvalue()


def to_csv_zip(graph: dict) -> bytes:
    validate_graph(graph)
    buf = io.BytesIO()
    files = {"nodes.csv": _csv(graph["nodes"], ["id", "kind", "label"]),
             "edges.csv": _csv(graph["edges"], ["id", "source", "target", "relation"]),
             "metadata.json": json.dumps({k: v for k, v in graph.items() if k not in ("nodes", "edges")},
                                         ensure_ascii=False, indent=2)}
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content.encode("utf-8"))
    return buf.getvalue()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--from-csv", action="store_true", help="Use aligned processed CSVs instead of live extraction")
    ap.add_argument("--entities", type=Path, help="Use this entities CSV (implies --from-csv)")
    ap.add_argument("--measurements", type=Path, help="Use this measurements CSV (implies --from-csv)")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--include-article", action="store_true")
    ap.add_argument("--occurrences", action="store_true")
    ap.add_argument("--out", type=Path, required=True, help="Output .json, .graphml, or .zip")
    ap.add_argument("--force", action="store_true", help="Replace an existing export")
    args = ap.parse_args()
    writers = {".json": lambda g: to_json(g).encode("utf-8"), ".graphml": to_graphml, ".zip": to_csv_zip}
    if args.out.suffix.lower() not in writers:
        ap.error("--out must end with .json, .graphml, or .zip")
    if args.out.exists() and not args.force:
        ap.error("Output exists; choose another path or use --force")
    try:
        cases = _read_csv(args.cases)
        case = next((c for c in cases if c["case_id"] == args.case_id), None)
        if case is None:
            ap.error(f"Unknown case_id: {args.case_id}")
        article = None
        if args.include_article:
            article = next((a for a in _read_csv(args.metadata) if a["article_id"] == case["article_id"]), None)
        from_csv = args.from_csv or args.entities is not None or args.measurements is not None
        if from_csv:
            entities = _read_csv(args.entities or ROOT / "data/processed/entities.csv")
            measurements = _read_csv(args.measurements or ROOT / "data/processed/measurements.csv")
            validate_annotations(cases, entities, "entities.csv")
            validate_annotations(cases, measurements, "measurements.csv")
            entities = [r for r in entities if r["case_id"] == args.case_id]
            measurements = [r for r in measurements if r["case_id"] == args.case_id]
        else:
            with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as con:
                entities, measurements = analyze_case(case["case_text"], con)
        case["data_source"] = str(args.cases)
        graph = build_graph(case, case["case_text"], entities, measurements,
                            aggregate=not args.occurrences, article=article)
        if from_csv and graph["warnings"]:
            raise ValueError("Annotation CSVs have unresolved occurrence links. Regenerate them from the same corpus.")
        payload = writers[args.out.suffix.lower()](graph)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("wb" if args.force else "xb") as f:
            f.write(payload)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        ap.error(str(exc))
    print(f'{len(graph["nodes"])} nodes, {len(graph["edges"])} edges → {args.out}')
    for warning in graph["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
