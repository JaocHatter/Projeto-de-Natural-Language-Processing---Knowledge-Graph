"""Local Cytoscape.js renderer for Streamlit's bidirectional components v2."""

import html
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

from graph_builder import graph_fingerprint, RELATION_LABELS

ASSETS = Path(__file__).resolve().parent / "assets"
HTML = """
<div class="kg-root">
  <div class="kg-controls">
    <input class="kg-search" type="search" placeholder="Find a concept…" aria-label="Find a graph node" />
    <select class="kg-picker" aria-label="Select a graph node"></select>
    <button class="kg-fit" type="button">Fit graph</button>
    <button class="kg-layout" type="button">Rearrange</button>
    <button class="kg-clear" type="button">Clear selection</button>
    <button class="kg-focus" type="button">Focus selection</button>
    <button class="kg-zoom-in" type="button" aria-label="Zoom in">+</button>
    <button class="kg-zoom-out" type="button" aria-label="Zoom out">−</button>
  </div>
  <div class="kg-surface">
    <div class="kg-canvas" role="img" aria-label="Case knowledge graph; use the node selector to inspect evidence"></div>
    <div class="kg-tooltip" hidden></div>
  </div>
  <div class="kg-legend" aria-label="Node categories"></div>
  <div class="kg-status" role="status">Click a node or edge to inspect evidence.</div>
</div>
"""


@st.cache_resource
def _component(asset_revision):
    vendor = (ASSETS / "cytoscape" / "cytoscape.min.js").read_text(encoding="utf-8")
    # A CommonJS scope keeps the UMD library local to this ES module.
    js = ("const cytoscape = (() => { const module = {exports: {}}; const exports = module.exports;\n"
          + vendor + "\nreturn module.exports; })();\n"
          + (ASSETS / "graph.js").read_text(encoding="utf-8"))
    return components.component("clinical_knowledge_graph", html=HTML,
                                css=(ASSETS / "graph.css").read_text(encoding="utf-8"), js=js,
                                # Cytoscape's pointer hit testing needs the real
                                # DOM event target, not Shadow DOM retargeting.
                                isolate_styles=False)


def cytoscape_elements(graph: dict, palette: dict) -> list[dict]:
    elements = []
    for node in graph["nodes"]:
        kind = node.get("entity_type", node["kind"])
        colour = palette.get(kind, ("#334155",))[0]
        tooltip = f'{node["kind"]} · {node["label"]}'
        if node.get("cui"):
            tooltip += f'\n{node["cui"]} · {", ".join(node["tuis"])} · {", ".join(node["vocabularies"])}'
            tooltip += f'\n{node["count"]} text mention(s); clinical assertion not assessed'
        if node["kind"] == "Measurement":
            tooltip += f'\nAssociation: {node["link_method"]} (heuristic)'
        elements.append({"data": {**node, "color": colour,
                                  "size": 28 + min(30, 7 * (node["count"] - 1) ** 0.5),
                                  "tooltip": tooltip}})
    for edge in graph["edges"]:
        label = RELATION_LABELS.get(edge["relation"], edge["relation"])
        method = edge.get("link_method", "")
        elements.append({"data": {**edge, "label": label,
                                  "line_style": "dashed" if method == "window_fallback" else "solid",
                                  "tooltip": f'{edge["relation"]}\n{method}'}})
    return elements


STYLE = [
    {"selector": "node", "style": {
        "label": "data(label)", "background-color": "data(color)",
        "shape": "ellipse", "width": "data(size)", "height": "data(size)", "font-size": 13,
        "color": "#1f2937", "text-wrap": "ellipsis", "text-max-width": "160px",
        "text-valign": "bottom", "text-margin-y": 8, "text-background-color": "#ffffff",
        "text-background-opacity": 0.9, "text-background-padding": "3px", "border-width": 2, "border-color": "#ffffff",
        "overlay-padding": 7,
    }},
    {"selector": 'node[kind = "Person"]', "style": {"width": 56, "height": 56, "border-width": 5, "border-style": "double", "border-color": "#64748b"}},
    {"selector": 'node[kind = "Age"], node[kind = "Sex"]', "style": {"width": 22, "height": 22}},
    {"selector": 'node[kind = "Article"]', "style": {"shape": "hexagon"}},
    {"selector": 'node[kind = "SemanticType"]', "style": {"width": 68, "height": 68, "font-weight": "bold", "font-size": 16, "border-width": 5, "border-opacity": 0.4}},
    {"selector": 'node[kind = "Measurement"]', "style": {"shape": "round-rectangle", "width": 100, "height": 30, "text-valign": "center", "text-margin-y": 0, "color": "#ffffff", "text-background-opacity": 0, "text-max-width": "90px", "font-size": 11}},
    {"selector": "edge", "style": {
        "curve-style": "bezier", "target-arrow-shape": "triangle", "arrow-scale": 0.65, "width": 1,
        "opacity": 0.45, "line-color": "#94a3b8", "target-arrow-color": "#94a3b8", "line-style": "data(line_style)",
    }},
    {"selector": "edge:selected", "style": {
        "label": "data(label)", "font-size": 11, "text-background-color": "#ffffff",
        "text-background-opacity": 1, "line-color": "#2563eb", "target-arrow-color": "#2563eb", "width": 2.5, "opacity": 1,
    }},
    {"selector": "node:selected", "style": {"border-width": 4, "border-color": "#111827"}},
    {"selector": ".kg-muted", "style": {"opacity": 0.14, "text-opacity": 0.1}},
    {"selector": ".kg-hover", "style": {"overlay-color": "#60a5fa", "overlay-opacity": 0.15}},
    {"selector": "node.kg-overview", "style": {"text-opacity": 0}},
    {"selector": 'node.kg-overview[kind = "SemanticType"], node.kg-overview[kind = "Person"], node:selected, node.kg-hover', "style": {"text-opacity": 1}},
]


def render_graph(graph: dict, palette: dict, layout: str = "cose", *, show_labels: bool = True,
                 views: dict | None = None, grouped: bool = False) -> tuple[dict | None, dict]:
    views = views or {"": graph}
    fingerprint = graph_fingerprint(graph) + ":" + layout + ":" + str(show_labels) + ":" + graph_fingerprint(views[""])
    # A revisit must not resurrect an expanded measurement from an old canvas.
    if st.session_state.get("kg_active_view") != fingerprint:
        st.session_state["kg_view_epoch"] = st.session_state.get("kg_view_epoch", 0) + 1
        st.session_state["kg_active_view"] = fingerprint
    fingerprint += ":" + str(st.session_state["kg_view_epoch"])
    style = STYLE + ([{"selector": "edge", "style": {"label": "data(label)", "font-size": 10,
                       "text-rotation": "autorotate", "text-background-color": "#f8fafc",
                       "text-background-opacity": 0.85}}] if show_labels else [])
    if grouped:
        style += [{"selector": 'edge[relation ^= "MENTIONS_"]', "style": {"opacity": 0.12}},
                  {"selector": 'edge[relation = "IS_A"]', "style": {"opacity": 0.65}},
                  {"selector": "edge:selected", "style": {"opacity": 1}},
                  {"selector": "edge.kg-muted", "style": {"opacity": 0.06}}]
    revision = tuple((ASSETS / name).stat().st_mtime_ns for name in ("graph.js", "graph.css"))
    result = _component(revision)(data={"elements": cytoscape_elements(graph, palette),
                                "views": {key: [item["id"] for item in [*view["nodes"], *view["edges"]]]
                                          for key, view in views.items()},
                                "grouped": grouped,
                                "style": style, "layout": layout, "fingerprint": fingerprint},
                          key=f"kg-{fingerprint}", default={"selection": None},
                          on_selection_change=lambda: None, height="content")
    selection = result.selection
    if not isinstance(selection, dict) or selection.get("fingerprint") != fingerprint:
        return None, views[""]
    visible = views.get(selection.get("id"), views[""])
    return next((item for item in [*visible["nodes"], *visible["edges"]]
                 if item["id"] == selection.get("id")), None), visible


def selection_evidence(selection: dict | None, graph: dict) -> list[dict]:
    if not selection:
        return []
    nodes = {n["id"]: n for n in graph["nodes"]}
    if selection.get("relation") == "IS_A":
        return [ev for ev in nodes[selection["source"]]["evidence"]
                if ev.get("entity_type") == selection["entity_type"]]
    if "evidence" in selection:
        evidence = list(selection["evidence"])
        if selection["kind"] == "Measurement":
            for edge in graph["edges"]:
                if edge["target"] == selection["id"] and edge.get("linked_entity_span"):
                    evidence += [ev for ev in nodes[edge["source"]]["evidence"]
                                 if [ev["start"], ev["end"]] == edge["linked_entity_span"]]
        return evidence
    evidence = list(nodes[selection["target"]]["evidence"])
    if selection.get("spans"):
        evidence = [ev for ev in evidence if [ev["start"], ev["end"]] in selection["spans"]]
    span = selection.get("linked_entity_span")
    if span:
        evidence += [ev for ev in nodes[selection["source"]]["evidence"]
                     if [ev["start"], ev["end"]] == span]
    return evidence


def render_inspector(selection: dict | None, graph: dict, text: str) -> list[dict]:
    evidence = selection_evidence(selection, graph)
    if not selection:
        st.info("Select a node or edge to see its metadata and source text.")
        return evidence
    st.subheader("Selected evidence")
    st.json({k: v for k, v in selection.items() if k != "evidence"}, expanded=False)
    if selection.get("provenance"):
        provenance = selection["provenance"]
        st.caption(f'Source: {provenance["data_source"]} · field: {provenance["field"]} · '
                   f'method: {provenance["method"]} · upstream: {provenance["upstream_value"] or "unknown"}')
    for ev in evidence:
        start, end = ev["start"], ev["end"]
        excerpt = (html.escape(text[max(0, start-100):start]) + "<mark>"
                   + html.escape(text[start:end]) + "</mark>" + html.escape(text[end:end+100]))
        st.caption(f'Characters {start}–{end} · {ev["case_id"]}')
        st.markdown(f'<div class="ner-text">{excerpt}</div>', unsafe_allow_html=True)
    return evidence
