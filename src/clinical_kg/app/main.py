#!/usr/bin/env python3
"""
Combined viewer: UMLS gazetteer NER (app.py) + value/unit extraction
(extract_measurements.py), overlaid on the same case text in one view.

Requires data/interim/gazetteer.db (build it first: clinical-kg build-gazetteer).

Run:
    streamlit run src/clinical_kg/app/main.py
"""

import csv
import html
import io
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st

from clinical_kg.extraction.entities import (
    DEFAULT_DB, DEFAULT_MAX_N, CSV_COLUMNS,
)
from clinical_kg.extraction.measurements import (
    analyze_case, CSV_COLUMNS as MEASUREMENT_COLUMNS,
)
from clinical_kg.paths import DEFAULT_CASES, DEFAULT_METADATA
from clinical_kg.corpus.patients import new_patient
from clinical_kg.graph.ui import graph_panel

# Same 6 slots as app.py (validated palette), plus one new category for this
# project's own extraction. Not independently contrast-validated like the
# original 6 -- chosen to be visually distinct from all of them.
PALETTE = {
    "Diagnosis":   ("#2a78d6", "#3987e5"),   # slot 1 blue
    "Treatment":   ("#eb6834", "#d95926"),   # slot 2 orange
    "Exam":        ("#1baf7a", "#199e70"),   # slot 3 aqua
    "Finding":     ("#eda100", "#c98500"),   # slot 4 yellow
    "Symptom":     ("#e87ba4", "#d55181"),   # slot 5 magenta
    "BodyPart":    ("#008300", "#008300"),   # slot 6 green
    "Measurement": ("#64748b", "#94a3b8"),   # slot 7 slate -- value/unit
    "Person": ("#334155", "#334155"),
    "Age": ("#8b5cf6", "#8b5cf6"),
    "Sex": ("#0891b2", "#0891b2"),
}
ENTITY_ORDER = list(PALETTE)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

def file_revision(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data
def load_cases(revision: tuple) -> list[dict]:
    with open(DEFAULT_CASES, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


@st.cache_data
def analyze(text: str, max_n: int, db_revision: tuple):
    """Full pipeline: gazetteer entities + measurements linked to them."""
    with closing(sqlite3.connect(DEFAULT_DB.as_uri() + "?mode=ro", uri=True)) as con:
        return analyze_case(text, con, max_n)


@st.cache_data
def load_articles(revision: tuple) -> dict:
    with DEFAULT_METADATA.open(newline="", encoding="utf-8") as f:
        return {r["article_id"]: r for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _colour_css() -> str:
    # Fixed light/gray backdrop, on purpose -- the point is to make the
    # highlight colours read consistently regardless of the viewer's OS
    # theme, so this deliberately does not follow prefers-color-scheme.
    # The --ent-* vars live on :root (not scoped to .ner-root) because
    # Streamlit renders each st.markdown/st.columns call into its own
    # container -- a div opened in one call and closed in a later one does
    # not actually nest them in the DOM, so anything scoped to .ner-root
    # would stop resolving the moment render_bars() lands in a sibling
    # container instead of a descendant of it.
    colours = "\n".join(f"  --ent-{k.lower()}: {v[0]};" for k, v in PALETTE.items())
    return f"""
<style>
:root {{
{colours}
  --ner-border: #cbd5e1;
}}
.ner-root {{
  background: #e5e7eb;
  padding: 1rem;
  border-radius: 12px;
  color: #1f2937;
}}
.ner-text {{
  white-space: pre-wrap; line-height: 2.05; font-size: 0.95rem;
  background: #ffffff; color: #1f2937;
  border: 1px solid var(--ner-border); border-radius: 8px;
  padding: 1rem 1.15rem; max-height: 520px; overflow-y: auto;
}}
.ner-text mark {{ padding: 0.08em 0.22em; border-radius: 3px; color: inherit; background: none; }}
.ner-legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem; margin-bottom: 0.6rem; color: #1f2937; }}
.ner-legend span {{ display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.83rem; }}
.ner-legend i {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
.ner-bars {{ display: flex; flex-direction: column; gap: 2px; background: #ffffff;
             border: 1px solid var(--ner-border); border-radius: 8px; padding: 0.75rem 1rem; }}
.ner-bar-row {{ display: grid; grid-template-columns: 6.5rem 1fr 2.6rem; align-items: center; gap: 0.5rem; }}
.ner-bar-row b {{ font-size: 0.82rem; font-weight: 500; color: #1f2937; }}
.ner-bar-row u {{ height: 15px; border-radius: 0 4px 4px 0; text-decoration: none; display: block; }}
.ner-bar-row s {{ font-size: 0.82rem; text-decoration: none; font-variant-numeric: tabular-nums; color: #1f2937; }}
</style>
"""


def render_legend(present: set[str]) -> str:
    items = "".join(
        f'<span><i style="background: var(--ent-{k.lower()})"></i>{k}</span>'
        for k in ENTITY_ORDER if k in present
    )
    return f'<div class="ner-legend">{items}</div>'


def render_highlighted(text: str, spans: list[dict]) -> str:
    """`spans`: dicts with start, end, kind (a PALETTE key), tooltip.
    Sorted by start; an overlap (rare) keeps whichever sorts first and skips
    the other rather than producing malformed nested <mark> tags."""
    spans = sorted(spans, key=lambda s: s["start"])
    out, cursor = [], 0
    for sp in spans:
        if sp["start"] < cursor:
            continue
        out.append(html.escape(text[cursor:sp["start"]]))
        colour = f'var(--ent-{sp["kind"].lower()})'
        tip = html.escape(sp["tooltip"], quote=True)
        out.append(
            f'<mark title="{tip}" style="background:{_rgba(PALETTE[sp["kind"]][0], 0.17)};'
            f'box-shadow: inset 0 -2px 0 {colour};'
            f'outline:{"2px solid #111827" if sp.get("selected") else "none"}">'
            f'{html.escape(text[sp["start"]:sp["end"]])}</mark>'
        )
        cursor = sp["end"]
    out.append(html.escape(text[cursor:]))
    return f'<div class="ner-text">{"".join(out)}</div>'


def render_bars(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    top = max(counts.values())
    rows = []
    for name in ENTITY_ORDER:
        n = counts.get(name, 0)
        if not n:
            continue
        rows.append(
            f'<div class="ner-bar-row" title="{name}: {n}">'
            f'<b>{name}</b>'
            f'<u style="width:{100 * n / top:.1f}%; background: var(--ent-{name.lower()})"></u>'
            f'<s>{n}</s></div>'
        )
    return f'<div class="ner-bars">{"".join(rows)}</div>'


def _entity_tooltip(e: dict) -> str:
    return f'{e["entity_type"]} · {e["cui"]} · {e["tui"]} · {e["vocabulary"]}'


def _measurement_tooltip(m: dict) -> str:
    value = m["value"] or f'{m["range_low"]}–{m["range_high"]}'
    if m["linked_entity_surface"]:
        link = f' → {m["linked_entity_surface"]} [{m["linked_entity_type"]}] ({m["link_method"]})'
    else:
        link = " → unlinked"
    return f'{value} {m["unit"]} · {m["unit_category"]}{link}'


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Clinical NER + Measurements", page_icon="🧬", layout="wide")
    st.title("🧬 Clinical Knowledge-Graph Extraction Viewer")
    st.caption(
        "UMLS gazetteer NER + value/unit extraction, on the same case text. "
        "Hover a highlight for details."
    )

    if not DEFAULT_DB.exists():
        st.error(f"Gazetteer not found at {DEFAULT_DB}\n\n"
                 "Build it first: `clinical-kg build-gazetteer`")
        st.stop()

    source = st.sidebar.radio("Source", ["Corpus case", "New case"])
    st.sidebar.markdown("---")

    text, meta = "", {}
    if source == "Corpus case":
        if not DEFAULT_CASES.exists():
            st.error("Cleaned corpus not found. Run: clinical-kg clean-cases")
            st.stop()
        cases = load_cases(file_revision(DEFAULT_CASES))
        if not cases:
            st.error("The cleaned corpus contains no patients.")
            st.stop()
        ids = [c["case_id"] for c in cases]
        chosen = st.sidebar.selectbox(f"Case ({len(ids)} available)", ids)
        case = next(c for c in cases if c["case_id"] == chosen)
        text = case.get("case_text") or ""
        meta = {k: v for k, v in case.items() if k != "case_text"}
        meta["data_source"] = "data/interim/cases_clean.csv"
        st.sidebar.markdown(
            f"**Patient** `{meta['case_id']}`\n\n"
            f"Age: **{meta['age'] or '—'}**  \n"
            f"Sex: **{meta['gender'] or '—'}**  \n"
            f"Article: `{meta['article_id']}`"
        )
        with st.sidebar.expander("Patient provenance"):
            st.json({k: meta.get(k, "") for k in ("source_case_ids", "age_method", "gender_method",
                                                   "age_upstream", "gender_upstream")})
    else:
        upload = st.file_uploader("Upload a case report (.txt)", type=["txt"])
        pasted = st.text_area("…or paste the case text here", height=200,
                              placeholder="An 83-year-old woman presented with…")
        if upload is not None:
            text = upload.getvalue().decode("utf-8", errors="replace")
            st.caption(f"Loaded **{upload.name}** ({len(text):,} characters)")
        else:
            text = pasted
        meta = new_patient(text)
        st.sidebar.write(f'Age: {meta["age"] or "unknown"} · Sex: {meta["gender"]}')

    if not text.strip():
        st.info("Select a corpus case, or paste / upload text to annotate.")
        st.stop()

    entities, measurements = analyze(text, DEFAULT_MAX_N, file_revision(DEFAULT_DB))

    counts = {
        **{e: sum(1 for x in entities if x["entity_type"] == e) for e in
           ["Diagnosis", "Treatment", "Exam", "Finding", "Symptom", "BodyPart"]},
        "Measurement": len(measurements),
    }
    counts = {k: v for k, v in counts.items() if v}

    spans = (
        [{"start": e["start"], "end": e["end"], "kind": e["entity_type"], "tooltip": _entity_tooltip(e)}
         for e in entities]
        + [{"start": m["start"], "end": m["end"], "kind": "Measurement", "tooltip": _measurement_tooltip(m)}
           for m in measurements]
    )

    st.markdown(_colour_css(), unsafe_allow_html=True)
    text_tab, graph_tab, tables_tab = st.tabs(["Annotated text", "Knowledge Graph", "Tables"])
    with graph_tab:
        article = (load_articles(file_revision(DEFAULT_METADATA)).get(meta["article_id"])
                   if DEFAULT_METADATA.exists() and meta.get("article_id") else None)
        st.markdown(render_legend(set(PALETTE)), unsafe_allow_html=True)
        graph, selected, evidence = graph_panel(meta, text, entities, measurements, PALETTE, article)
    selected_spans = {(ev["start"], ev["end"]) for ev in evidence}
    with text_tab:
        if not spans:
            st.warning("Nothing matched in this text.")
        if selected_spans:
            st.caption("Black outlines mark the graph selection. Exact excerpts are in the graph inspector.")
        marked = [{**sp, "selected": (sp["start"], sp["end"]) in selected_spans} for sp in spans]
        if selected_spans:
            marked = [sp for sp in marked if sp["selected"] or not any(
                sp["start"] < end and sp["end"] > start for start, end in selected_spans)]
        st.markdown(render_legend(set(counts)) + render_highlighted(text, marked), unsafe_allow_html=True)
        st.subheader("By type")
        st.markdown(render_bars(counts), unsafe_allow_html=True)
    with tables_tab:
        only_selected = st.checkbox("Only graph selection", disabled=selected is None)
        df = pd.DataFrame([{**meta, **e} for e in entities], columns=CSV_COLUMNS)
        df_meas = pd.DataFrame([{**meta, **m} for m in measurements], columns=MEASUREMENT_COLUMNS)
        for column in ("linked_entity_start", "linked_entity_end"):
            df_meas[column] = pd.to_numeric(df_meas[column], errors="coerce").astype("Int64")
        shown_e, shown_m = df, df_meas
        edges = graph["edges"]
        if only_selected and selected is not None:
            shown_e = df.loc[[(e["start"], e["end"]) in selected_spans for e in entities]]
            shown_m = df_meas.loc[[(m["start"], m["end"]) in selected_spans for m in measurements]]
            edges = [e for e in edges if selected["id"] in (e["id"], e["source"], e["target"])]
        st.subheader("Entities")
        st.dataframe(shown_e, width="stretch", height=260, hide_index=True)
        st.subheader("Measurements")
        st.dataframe(shown_m, width="stretch", height=260, hide_index=True)
        st.subheader("Relations")
        st.dataframe(pd.DataFrame(edges, columns=["source", "relation", "target", "count", "link_method"]),
                     width="stretch", height=260, hide_index=True)
        st.caption("Extraction CSV downloads include the entire patient, regardless of graph filters.")
        for label, frame in (("entities", df), ("measurements", df_meas)):
            buf = io.StringIO()
            frame.to_csv(buf, index=False)
            st.download_button(f"Download {label} (CSV)", buf.getvalue(),
                               file_name=f'{label}_{meta["case_id"]}.csv', mime="text/csv")


if __name__ == "__main__":
    main()
