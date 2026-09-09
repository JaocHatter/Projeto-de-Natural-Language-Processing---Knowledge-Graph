"""Patient graph controls and exports; application text/tables stay in app.py."""

import streamlit as st

from graph_builder import build_graph, filter_graph, group_graph, interaction_views, LINK_METHODS
from graph_export import to_json, to_graphml, to_csv_zip
from graph_view import render_graph, render_inspector


def graph_panel(meta: dict, text: str, entities: list[dict], measurements: list[dict],
                palette: dict, article: dict | None = None) -> tuple[dict, dict | None, list[dict]]:
    st.caption("One Person per cleaned patient row. Age and Sex come from the cleaning pipeline. "
               "Clinical edges describe text mentions; measurement associations are heuristic.")
    clinical_types = [kind for kind in palette if kind not in ("Measurement", "Person", "Age", "Sex")]
    with st.expander("Graph filters", expanded=True):
        left, right = st.columns(2)
        with left:
            aggregate = st.radio("Representation", ["Aggregated concepts", "All occurrences"],
                                 horizontal=True) == "Aggregated concepts"
            types = st.multiselect("Entity types", clinical_types, default=clinical_types)
            measurement_mode = st.selectbox("Measurements", ["On node selection", "Hide all"],
                                            key="kg_measurements_v2")
            grouped = st.checkbox("Group by IS_A category", value=False,
                                  help="Arrange concepts around their semantic class. Multi-type concepts retain all IS_A links.")
            demographics = st.checkbox("Show Age and Sex", value=True)
        with right:
            methods = st.multiselect("Measurement association methods", LINK_METHODS, default=list(LINK_METHODS),
                                     help="All methods are shown initially. Dashed edges indicate window fallback; "
                                          "same-sentence associations also require review.")
            limit = st.slider("Maximum visible nodes", 20, 300, 100, step=20)
            layout = st.selectbox("Layout", ["cose", "breadthfirst", "circle"],
                                  disabled=grouped,
                                  format_func={"cose": "Force-directed", "breadthfirst": "Hierarchical",
                                               "circle": "Circular"}.get)
            labels = st.checkbox("Show relation labels", value=False)
            show_article = st.checkbox("Show source article", value=False, disabled=article is None)
    graph = build_graph(meta, text, entities, measurements, aggregate=aggregate, article=article)
    if grouped:
        graph = group_graph(graph, clinical_types)
    candidates = filter_graph(graph, entity_types=types, link_methods=methods,
                              show_measurements=measurement_mode != "Hide all",
                              show_article=show_article, show_demographics=demographics,
                              max_nodes=max(2, len(graph["nodes"])))
    if grouped:
        # Rebuild only the classes of the filtered clinical nodes.
        candidates = group_graph(candidates, types)
    views = interaction_views(candidates, on_demand=True, max_nodes=limit)
    st.caption("Click a concept to reveal its measurements; click Person for unlinked measurements. "
               "Click the background to hide them again. IS_A groups are project categories, not clinical assertions.")
    selection, visible = render_graph(candidates, palette, layout, show_labels=labels, views=views, grouped=grouped)
    st.caption(f'{len(visible["nodes"])} visible nodes · {len(visible["edges"])} edges · '
               f'{len(graph["nodes"])} nodes in the complete patient graph')
    if visible["view"]["hidden_by_limit"]:
        st.warning(f'{visible["view"]["hidden_by_limit"]} matching nodes hidden by the node limit. '
                   'Narrow the filters or increase the limit; complete exports retain them.')
    for warning in graph["warnings"]:
        st.warning(warning)
    st.caption("Circles: clinical concepts · Double ring: Person · Small circles: Age/Sex · "
               "Pills: measurements · Dashed edge: window fallback.")
    st.caption("Select a node to emphasize its direct relations. Click the background to clear the selection.")
    evidence = render_inspector(selection, visible, text)
    scope = st.radio("Graph export scope", ["Visible graph", "Complete patient graph"], horizontal=True)
    exported = visible if scope == "Visible graph" else graph
    suffix = "visible" if scope == "Visible graph" else "complete"
    name = f'graph_{meta["case_id"]}_{graph["mode"]}_{suffix}'
    for col, label, payload, ext, mime in zip(st.columns(3),
            ["Download graph JSON", "Download GraphML", "Download graph CSVs"],
            [to_json(exported), to_graphml(exported), to_csv_zip(exported)],
            ["json", "graphml", "zip"], ["application/json", "application/graphml+xml", "application/zip"]):
        with col:
            st.download_button(label, payload, f"{name}.{ext}", mime)
    return graph, selection, evidence
