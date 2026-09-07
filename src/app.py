#!/usr/bin/env python3
"""
Combined viewer: UMLS gazetteer NER (app.py) + value/unit extraction
(extract_measurements.py), overlaid on the same case text in one view.

Requires data/interim/gazetteer.db (build it first: python3 src/build_gazetteer.py).

Run:
    streamlit run src/app_full.py
"""

import csv
import html
import io
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_entities import (  # noqa: E402
    DEFAULT_DB, DEFAULT_MAX_N, extract,
)
from extract_measurements import (  # noqa: E402
    find_measurements, link_measurement, split_sentences,
)

# LOCAL OVERRIDE (not committed): point the app at the cleaned/merged corpus
# (src/clean_cases.py) instead of the raw data/raw/cases.csv -- fixes the 2
# bad ages, merges the 2 fragmented case series into one row per real
# patient, and drops the 1 row that's a cohort summary, not a patient. See
# CLAUDE.md "Known data-quality issues" for why.
DEFAULT_CASES = Path(__file__).resolve().parents[1] / "data" / "interim" / "cases_clean.csv"

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
}
ENTITY_ORDER = list(PALETTE)

CSV_COLUMNS = ["case_id", "article_id", "age", "gender", "start", "end",
               "surface_text", "term_norm", "cui", "tui", "entity_type",
               "n_tokens", "vocabulary", "tty"]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(str(DEFAULT_DB), check_same_thread=False)


@st.cache_data
def load_cases() -> list[dict]:
    with open(DEFAULT_CASES, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


@st.cache_data
def analyze(text: str, max_n: int):
    """Full pipeline: gazetteer entities + measurements linked to them."""
    con = get_connection()
    gaz_hits = extract(text, con, max_n)

    entities = [
        {"start": e["start"], "end": e["end"],
         "surface_text": e["surface_text"], "entity_type": e["entity_type"]}
        for e in gaz_hits
    ]
    entities.sort(key=lambda e: e["start"])

    sent_spans = split_sentences(text)
    sent_starts = [s for s, _ in sent_spans]

    measurements = find_measurements(text)
    for m in measurements:
        m.update(link_measurement(m, entities, sent_starts))

    return gaz_hits, measurements


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
            f'box-shadow: inset 0 -2px 0 {colour}">'
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
                 "Build it first: `python3 src/build_gazetteer.py`")
        st.stop()

    source = st.sidebar.radio("Source", ["Corpus case", "New case"])
    st.sidebar.markdown("---")

    text, meta = "", {}
    if source == "Corpus case":
        cases = load_cases()
        ids = [c["case_id"] for c in cases]
        chosen = st.sidebar.selectbox(f"Case ({len(ids)} available)", ids)
        case = next(c for c in cases if c["case_id"] == chosen)
        text = case.get("case_text") or ""
        meta = {"case_id": case.get("case_id"), "article_id": case.get("article_id"),
                "age": case.get("age"), "gender": case.get("gender")}
        st.sidebar.markdown(
            f"**Patient** `{meta['case_id']}`\n\n"
            f"Age: **{meta['age'] or '—'}**  \n"
            f"Sex: **{meta['gender'] or '—'}**  \n"
            f"Article: `{meta['article_id']}`"
        )
    else:
        upload = st.file_uploader("Upload a case report (.txt)", type=["txt"])
        pasted = st.text_area("…or paste the case text here", height=200,
                              placeholder="An 83-year-old woman presented with…")
        if upload is not None:
            text = upload.getvalue().decode("utf-8", errors="replace")
            st.caption(f"Loaded **{upload.name}** ({len(text):,} characters)")
        else:
            text = pasted
        meta = {"case_id": "pasted", "article_id": "", "age": "", "gender": ""}

    if not text.strip():
        st.info("Select a corpus case, or paste / upload text to annotate.")
        st.stop()

    entities, measurements = analyze(text, DEFAULT_MAX_N)

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
    st.markdown('<div class="ner-root">', unsafe_allow_html=True)
    if not spans:
        st.warning("Nothing matched in this text.")
    st.markdown(render_legend(set(counts)) + render_highlighted(text, spans), unsafe_allow_html=True)

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("By type")
        st.markdown(render_bars(counts), unsafe_allow_html=True)
    with right:
        st.subheader("Entities")
        rows = [{**meta, **e} for e in entities]
        df = pd.DataFrame(rows, columns=CSV_COLUMNS) if rows else pd.DataFrame(columns=CSV_COLUMNS)
        st.dataframe(df.drop(columns=["article_id", "age", "gender"]),
                     width="stretch", height=260, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Measurements")
    cols_meas = ["start", "end", "surface_text", "value", "range_low", "range_high",
                 "unit", "unit_category", "linked_entity_surface", "linked_entity_type",
                 "link_method"]
    df_meas = pd.DataFrame(measurements, columns=cols_meas) if measurements else pd.DataFrame(columns=cols_meas)
    st.dataframe(df_meas, width="stretch", height=280, hide_index=True)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "⬇ Download entities (CSV)", buf.getvalue(),
        file_name=f"entities_{meta.get('case_id') or 'pasted'}.csv", mime="text/csv",
    )


if __name__ == "__main__":
    main()
