"""Build evidence-backed, per-case graphs without UI or third-party dependencies.

Concepts are aggregated by CUI; occurrences and measurement links retain their
exact text offsets. No edge asserts a clinical diagnosis or causal relation.
"""

import hashlib
import json
from collections import defaultdict
from urllib.parse import quote

LINK_METHODS = ("same_sentence", "same_sentence_forward", "window_fallback", "none")
MENTION_RELATIONS = {
    "Diagnosis": "MENTIONS_DIAGNOSIS", "Treatment": "MENTIONS_TREATMENT",
    "Exam": "MENTIONS_EXAM", "Symptom": "MENTIONS_SYMPTOM",
    "Finding": "MENTIONS_FINDING", "BodyPart": "MENTIONS_BODY_PART",
}
# Extracted clinical relations. These are heuristic text relations from
# extract_relations.py, carrying an assertion status -- still not clinical
# assertions, but no longer only "this case mentions X".
CLINICAL_RELATIONS = {
    "HAS_SYMPTOM": "has symptom", "HAS_DIAGNOSIS": "has diagnosis",
    "HAS_FINDING": "has finding", "TREATED_WITH": "treated with",
    "REVEALED_BY": "revealed", "LOCATED_IN": "located in",
    "CAUSED_BY": "caused by", "COORDINATE_WITH": "co-occurs with",
}
ASSERTION_STATUSES = ("affirmed", "negated", "hedged", "historical", "family",
                      "not_assessed")
RELATION_LABELS = {
    "IS_A": "is a",
    **CLINICAL_RELATIONS,
    **{relation: "mentions " + kind.lower() for kind, relation in MENTION_RELATIONS.items()},
    "HAS_AGE": "has age", "HAS_SEX": "has sex", "REPORTS_PERSON": "reports patient",
    "ASSOCIATED_WITH_MEASUREMENT": "associated measurement",
    "CONTAINS_UNLINKED_MEASUREMENT": "unlinked measurement",
}


def _id(kind: str, *parts) -> str:
    return ":".join([kind, *(quote(str(p), safe="") for p in parts)])


def _evidence(row: dict, case_id: str, text: str) -> dict:
    start, end = int(row["start"]), int(row["end"])
    if not 0 <= start < end <= len(text) or text[start:end] != row["surface_text"]:
        raise ValueError(f"Invalid evidence offsets for {case_id}: {start}:{end}")
    if row.get("case_id", case_id) != case_id:
        raise ValueError("Cannot mix cases in a per-case graph")
    return {"case_id": case_id, "start": start, "end": end,
            "surface_text": row["surface_text"]}


def build_graph(meta: dict, text: str, entities: list[dict], measurements: list[dict],
                *, aggregate: bool = True, article: dict | None = None,
                relations: list[dict] | None = None) -> dict:
    """Return a deterministic JSON-compatible graph; accept live or legacy CSV rows.

    A measurement joins an exact occurrence, even in the aggregated view. A
    supplied CUI must agree with that occurrence. Unresolved links are retained
    as unlinked measurements with a warning, never attached to an arbitrary CUI.
    """
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    case_id = str(meta.get("case_id") or f"pasted-{text_hash[:16]}")
    case_node = _id("person", case_id)
    source_ids = meta.get("source_case_ids") or case_id
    if isinstance(source_ids, str):
        source_ids = source_ids.split("|")
    nodes = [{"id": case_node, "kind": "Person", "label": f"Person · {case_id}",
              "case_id": case_id, "article_id": meta.get("article_id") or "",
              "age": meta.get("age", ""), "gender": meta.get("gender", ""),
              "source_case_ids": source_ids,
              **{k: meta.get(k, "") for k in ("age_upstream", "gender_upstream", "age_method", "gender_method")},
              "count": 1, "evidence": []}]
    edges, warnings = [], []

    def edge(source, target, relation, **attrs):
        edges.append({"id": _id("edge", relation, source, target),
                      "source": source, "target": target, "relation": relation,
                      "case_id": case_id, **attrs})

    # These values belong to the cleaned patient row. In particular, newborn
    # age 0 is known; missing age must never be coerced to zero or upstream age.
    for kind, field, unit in (("Age", "age", "years"), ("Sex", "gender", "")):
        value = meta.get(field)
        if value is None or str(value).strip().lower() in ("", "unknown", "nan"):
            continue
        value = str(value).strip()
        node_id = _id(kind.lower(), case_id)
        provenance = {"field": field, "method": meta.get(f"{field}_method") or "provided",
                      "upstream_value": meta.get(f"{field}_upstream", ""),
                      "data_source": meta.get("data_source") or "patient_metadata",
                      "source_case_ids": source_ids}
        nodes.append({"id": node_id, "kind": kind, "entity_type": kind,
                      "label": f"{value} {unit}".strip(), "value": value, "unit": unit,
                      "case_id": case_id, "count": 1, "evidence": [], "provenance": provenance})
        edge(case_node, node_id, f"HAS_{kind.upper()}", provenance=provenance)

    if article is not None and meta.get("article_id"):
        article_id = str(meta["article_id"])
        if str(article.get("article_id")) != article_id:
            raise ValueError("Article metadata does not match the case")
        article_node = _id("article", article_id)
        nodes.append({"id": article_node, "kind": "Article",
                      "label": article.get("title") or article_id,
                      **{k: article.get(k, "") for k in ("article_id", "title", "year", "doi")},
                      "count": 1, "evidence": []})
        edge(article_node, case_node, "REPORTS_PERSON")

    groups = defaultdict(list)
    occurrences = {}
    for row in sorted(entities, key=lambda e: (int(e["start"]), int(e["end"]), e["cui"])):
        ev = _evidence(row, case_id, text)
        if not row.get("cui"):
            raise ValueError("An entity must have a CUI")
        span = (ev["start"], ev["end"])
        node_id = (_id("concept", row["cui"]) if aggregate else
                   _id("mention", case_id, *span, row["cui"]))
        if span in occurrences:
            raise ValueError(f"Duplicate entity occurrence: {span}")
        occurrences[span] = (node_id, row["cui"])
        groups[node_id].append((row, ev))

    for node_id, rows in sorted(groups.items()):
        evidence = [{**ev, **{k: row.get(k, "") for k in
                              ("cui", "tui", "entity_type", "vocabulary", "tty", "preferred_term")}}
                    for row, ev in rows]
        # term_pref is the matched gazetteer term, not necessarily a global CUI
        # preferred name. Prefer PT within observed matches, then lexical order.
        label_row = min((row for row, _ in rows), key=lambda r:
                        (r.get("tty") != "PT", r.get("preferred_term") or r["surface_text"]))
        node = {"id": node_id, "kind": "Concept" if aggregate else "Mention",
                "label": label_row.get("preferred_term") or label_row["surface_text"],
                "preferred_term": label_row.get("preferred_term") or label_row["surface_text"],
                "cui": label_row["cui"], "entity_type": label_row["entity_type"],
                "tuis": sorted({r.get("tui", "") for r, _ in rows}),
                "entity_types": sorted({r["entity_type"] for r, _ in rows}),
                "vocabularies": sorted({r.get("vocabulary", "") for r, _ in rows}),
                "count": len(rows), "evidence": evidence}
        nodes.append(node)
        for kind in node["entity_types"]:
            typed_rows = [(r, ev) for r, ev in rows if r["entity_type"] == kind]
            edge(case_node, node_id, MENTION_RELATIONS[kind], count=len(typed_rows),
                 entity_type=kind, assertion_status="not_assessed",
                 spans=[[ev["start"], ev["end"]] for _, ev in typed_rows],
                 surface_forms=sorted({r["surface_text"] for r, _ in typed_rows}))

    # Extracted relations. Endpoints are resolved through the same occurrence
    # index the measurements use, so a relation can only reference a span that
    # actually produced an entity node. A negated relation is retained and
    # labelled, never silently dropped -- "no rebound tenderness" is a finding
    # about the patient too.
    relation_edges: dict[str, dict] = {}
    for row in sorted(relations or [], key=lambda r: (int(r["tail_start"]),
                                                      int(r["tail_end"]),
                                                      r["relation"])):
        if row.get("case_id", case_id) != case_id:
            raise ValueError("Cannot mix cases in a per-case graph")
        relation = row["relation"]
        if relation not in CLINICAL_RELATIONS:
            raise ValueError(f"Unknown relation: {relation}")
        status = row.get("assertion_status") or "not_assessed"
        if status not in ASSERTION_STATUSES:
            raise ValueError(f"Unknown assertion status: {status}")
        tail = occurrences.get((int(row["tail_start"]), int(row["tail_end"])))
        if tail is None:
            warnings.append(f"Relation {relation} dropped: unresolved tail span.")
            continue
        if str(row.get("head_kind", "")) == "Person" or row.get("head_start") in ("", None, -1, "-1"):
            source = case_node
        else:
            head = occurrences.get((int(row["head_start"]), int(row["head_end"])))
            if head is None:
                warnings.append(f"Relation {relation} dropped: unresolved head span.")
                continue
            source = head[0]
        if source == tail[0]:
            continue          # aggregation collapsed both arguments onto one concept
        edge_id = _id("edge", relation, source, tail[0])
        span = [int(row["tail_start"]), int(row["tail_end"])]
        score = float(row.get("score") or 0.0)
        existing = relation_edges.get(edge_id)
        if existing is None:
            relation_edges[edge_id] = {
                "id": edge_id, "source": source, "target": tail[0],
                "relation": relation, "case_id": case_id,
                "assertion_status": status, "count": 1,
                "score": score, "heuristic": True,
                "trigger_text": row.get("trigger_text", ""),
                "trigger_category": row.get("trigger_category", ""),
                "spans": [span],
            }
        else:
            existing["count"] += 1
            existing["spans"].append(span)
            # Keep the best-supported reading of a repeated relation.
            if score > existing["score"]:
                existing.update(score=score, assertion_status=status,
                                trigger_text=row.get("trigger_text", ""),
                                trigger_category=row.get("trigger_category", ""))
    edges.extend(relation_edges.values())

    measurement_ids = set()
    for row in sorted(measurements, key=lambda m: (int(m["start"]), int(m["end"]))):
        ev = _evidence(row, case_id, text)
        node_id = _id("measurement", case_id, ev["start"], ev["end"])
        if node_id in measurement_ids:
            raise ValueError(f"Duplicate measurement: {node_id}")
        measurement_ids.add(node_id)
        method = row.get("link_method", "none")
        if method not in LINK_METHODS:
            raise ValueError(f"Unknown link method: {method}")
        target = None
        linked_span = []
        if method != "none":
            try:
                linked_span = [int(row["linked_entity_start"]), int(row["linked_entity_end"])]
                target = occurrences.get(tuple(linked_span))
            except (KeyError, ValueError, TypeError):
                pass
            if target and row.get("linked_entity_cui") not in (None, "", target[1]):
                raise ValueError(f"Measurement CUI disagrees with its evidence: {node_id}")
            if not target:
                warnings.append(f"Unresolved entity link for {node_id}; retained as unlinked.")
        effective_method = method if target else "none"
        nodes.append({"id": node_id, "kind": "Measurement", "entity_type": "Measurement",
                      "label": row["surface_text"], "count": 1, "evidence": [ev],
                      **{k: row.get(k, "") for k in
                         ("value", "range_low", "range_high", "unit", "unit_norm", "unit_category")},
                      "is_range": str(row.get("is_range", False)).lower() == "true",
                      "link_method": effective_method, "original_link_method": method,
                      "linked_entity_cui": target[1] if target else "",
                      "linked_entity_span": linked_span})
        if target:
            edge(target[0], node_id, "ASSOCIATED_WITH_MEASUREMENT",
                 link_method=method, linked_entity_span=linked_span,
                 heuristic=True, lower_confidence=method == "window_fallback")
        else:
            edge(case_node, node_id, "CONTAINS_UNLINKED_MEASUREMENT", link_method="none")

    graph = {"schema_version": 3, "case_id": case_id, "text_sha256": text_hash,
             "source_case_ids": source_ids, "data_source": meta.get("data_source", "patient_metadata"),
             "mode": "concepts" if aggregate else "occurrences",
             "semantics": "Text mentions, heuristic measurement associations and "
                          "heuristic extracted relations with assertion status; "
                          "not clinical assertions.",
             "nodes": sorted(nodes, key=lambda n: n["id"]),
             "edges": sorted(edges, key=lambda e: e["id"]), "warnings": warnings}
    validate_graph(graph)
    return graph


def validate_graph(graph: dict) -> None:
    ids = [n["id"] for n in graph["nodes"]]
    edge_ids = [e["id"] for e in graph["edges"]]
    if len(ids) != len(set(ids)) or len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Graph contains duplicate IDs")
    known = set(ids)
    if any(e["source"] not in known or e["target"] not in known for e in graph["edges"]):
        raise ValueError("Graph contains dangling edges")


def filter_graph(graph: dict, *, entity_types: list[str], link_methods: list[str],
                 show_measurements: bool = True, show_article: bool = False,
                 max_nodes: int = 100, show_demographics: bool = True) -> dict:
    """Project a graph without relabelling filtered links as unlinked evidence.

    Retain context nodes first, then concepts by mention count, with their
    measurements immediately afterwards. The cap includes all visible nodes.
    """
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least 2")
    by_id = {n["id"]: n for n in graph["nodes"]}
    children = defaultdict(list)
    for edge in graph["edges"]:
        if by_id[edge["target"]]["kind"] == "Measurement":
            children[edge["source"]].append(by_id[edge["target"]])
    ordered = [n for n in graph["nodes"] if n["kind"] == "Person"]
    ordered += [n for n in graph["nodes"] if
                (show_demographics and n["kind"] in ("Age", "Sex")) or
                (show_article and n["kind"] == "Article")]
    concepts = [n for n in graph["nodes"] if n["kind"] in ("Concept", "Mention")
                and set(n["entity_types"]) & set(entity_types)]
    for node in sorted(concepts, key=lambda n: (-n["count"], n["label"], n["id"])):
        ordered.append(node)
        if show_measurements:
            ordered.extend(n for n in children[node["id"]] if n["link_method"] in link_methods)
    if show_measurements and "none" in link_methods:
        ordered.extend(n for n in graph["nodes"] if n["kind"] == "Measurement"
                       and n["link_method"] == "none")
    nodes = ordered[:max_nodes]
    ids = {n["id"] for n in nodes}
    result = {**graph, "nodes": nodes,
              "edges": [e for e in graph["edges"] if e["source"] in ids and e["target"] in ids
                        and ("entity_type" not in e or e["entity_type"] in entity_types)],
              "view": {"eligible_nodes": len(ordered), "hidden_by_limit": max(0, len(ordered)-len(nodes)),
                       "entity_types": entity_types, "link_methods": link_methods,
                       "show_measurements": show_measurements, "show_article": show_article,
                       "show_demographics": show_demographics,
                       "max_nodes": max_nodes}}
    validate_graph(result)
    return result


def graph_fingerprint(graph: dict) -> str:
    return hashlib.sha256(json.dumps(graph, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def group_graph(graph: dict, entity_types: list[str]) -> dict:
    """Add project semantic classes, not an inferred clinical/UMLS hierarchy."""
    nodes, edges = list(graph["nodes"]), list(graph["edges"])
    classes = set()
    for node in graph["nodes"]:
        for kind in sorted(set(node.get("entity_types", [])) & set(entity_types)):
            classes.add(kind)
            edges.append({"id": _id("is-a", node["id"], kind), "source": node["id"],
                          "target": _id("type", kind), "relation": "IS_A",
                          "entity_type": kind, "classification_source": "project_entity_type"})
    nodes.extend({"id": _id("type", kind), "kind": "SemanticType", "entity_type": kind,
                  "label": kind, "count": 1, "evidence": []} for kind in sorted(classes))
    result = {**graph, "nodes": nodes, "edges": edges}
    validate_graph(result)
    return result


def interaction_views(graph: dict, *, on_demand: bool, max_nodes: int) -> dict[str, dict]:
    """Precompute bounded projections shared by the canvas and visible exports.

    Clicking Person exposes only unlinked measurements. A measurement or its
    association edge keeps the owning concept's projection active.
    """
    nodes = {n["id"]: n for n in graph["nodes"]}
    children = defaultdict(list)
    owner = {}
    for edge in graph["edges"]:
        if nodes[edge["target"]]["kind"] == "Measurement":
            children[edge["source"]].append(edge["target"])
            owner[edge["target"]] = edge["source"]
            owner[edge["id"]] = edge["source"]
    context = [n["id"] for n in graph["nodes"] if n["kind"] in ("Person", "Age", "Sex", "Article", "SemanticType")]
    clinical = [n["id"] for n in graph["nodes"] if n["kind"] in ("Concept", "Mention")]

    def project(focus=""):
        if on_demand:
            ordered = list(dict.fromkeys(context + ([focus] if focus else [])
                                        + children.get(focus, []) + clinical))
        else:
            ordered = context + [n["id"] for n in graph["nodes"] if n["id"] not in context]
        ids = set(ordered[:max_nodes])
        return {**graph, "nodes": [nodes[i] for i in ordered[:max_nodes]],
                "edges": [e for e in graph["edges"] if e["source"] in ids and e["target"] in ids],
                "view": {**graph["view"], "max_nodes": max_nodes,
                         "measurements_on_demand": on_demand, "measurement_owner": focus,
                         "eligible_nodes": len(ordered), "hidden_by_limit": max(0, len(ordered)-max_nodes)}}

    views = {"": project()}
    if on_demand:
        # Only initially visible owners can be expanded, never hidden concepts.
        available = {n["id"] for n in views[""]["nodes"]}
        for source in children.keys() & available:
            views[source] = project(source)
        for selection, source in owner.items():
            if source in views and selection in {
                item["id"] for item in [*views[source]["nodes"], *views[source]["edges"]]
            }:
                views[selection] = views[source]
    return views
