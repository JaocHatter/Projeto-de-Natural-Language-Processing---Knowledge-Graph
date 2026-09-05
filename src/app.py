#!/usr/bin/env python3
"""
Streamlit viewer for gazetteer-based NER over clinical case reports.

Run:
    /home/jaoc/anaconda3/envs/agents/bin/python3.14 -m streamlit run src/app.py

Shows entity spans highlighted in context, for either one of the 56 corpus
cases or arbitrary text pasted / uploaded by the user. Extraction is live
(~35 ms per case), so both paths use the identical code path as the CLI.
"""

import csv
import html
import io
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Same idiom as extract_entities.py: guarantee the sibling modules import
# regardless of how Streamlit resolves the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_entities import (  # noqa: E402
    DEFAULT_CASES, DEFAULT_DB, DEFAULT_MAX_N, extract,
)

# Categorical palette, slots 1-6 of the validated default theme. Validated with
# scripts/validate_palette.js: ALL CHECKS PASS in both modes (worst adjacent CVD
# dE 9.1 light / 8.4 dark; normal-vision 19.6 / 19.3). The light-mode contrast
# WARN invokes the relief rule, satisfied here by direct bar labels, the text
# legend, and the entity table -- identity is never carried by colour alone.
PALETTE = {
    "Diagnosis": ("#2a78d6", "#3987e5"),   # slot 1 blue
    "Treatment": ("#eb6834", "#d95926"),   # slot 2 orange
    "Exam":      ("#1baf7a", "#199e70"),   # slot 3 aqua
    "Finding":   ("#eda100", "#c98500"),   # slot 4 yellow
    "Symptom":   ("#e87ba4", "#d55181"),   # slot 5 magenta
    "BodyPart":  ("#008300", "#008300"),   # slot 6 green
}
ENTITY_ORDER = list(PALETTE)

CSV_COLUMNS = ["case_id", "article_id", "age", "gender", "start", "end",
               "surface_text", "term_norm", "cui", "tui", "entity_type",
               "n_tokens", "vocabulary", "tty"]


# ---------------------------------------------------------------------------
# Cached resources -- Streamlit re-runs this whole script on every interaction
# ---------------------------------------------------------------------------

@st.cache_resource
def get_connection() -> sqlite3.Connection:
    # check_same_thread=False: Streamlit runs re-runs on a ScriptRunner thread,
    # and Python's sqlite3 enforces its own same-thread check independently of
    # the module's threadsafety level.
    return sqlite3.connect(str(DEFAULT_DB), check_same_thread=False)


@st.cache_data
def load_cases() -> list[dict]:
    with open(DEFAULT_CASES, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


@st.cache_data
def run_extraction(text: str, max_n: int) -> list[dict]:
    return extract(text, get_connection(), max_n)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _colour_css() -> str:
    """Per-entity custom properties, with dark-mode steps from the same theme."""
    light = "\n".join(f"  --ent-{k.lower()}: {v[0]};" for k, v in PALETTE.items())
    dark = "\n".join(f"  --ent-{k.lower()}: {v[1]};" for k, v in PALETTE.items())
    return f"""
<style>
.ner-root {{
{light}
  --ner-border: rgba(128,128,128,0.32);
}}
@media (prefers-color-scheme: dark) {{
  .ner-root {{
{dark}
  }}
}}
.ner-text {{
  white-space: pre-wrap; line-height: 2.05; font-size: 0.95rem;
  border: 1px solid var(--ner-border); border-radius: 8px;
  padding: 1rem 1.15rem; max-height: 520px; overflow-y: auto;
}}
.ner-text mark {{
  padding: 0.08em 0.22em; border-radius: 3px;
  color: inherit; background: none;
}}
.ner-legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem; margin-bottom: 0.6rem; }}
.ner-legend span {{ display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.83rem; }}
.ner-legend i {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
.ner-bars {{ display: flex; flex-direction: column; gap: 2px; }}
.ner-bar-row {{ display: grid; grid-template-columns: 5.5rem 1fr 2.6rem; align-items: center; gap: 0.5rem; }}
.ner-bar-row b {{ font-size: 0.82rem; font-weight: 500; }}
.ner-bar-row u {{ height: 15px; border-radius: 0 4px 4px 0; text-decoration: none; display: block; }}
.ner-bar-row s {{ font-size: 0.82rem; text-decoration: none; font-variant-numeric: tabular-nums; }}
</style>
"""


def render_legend(present: set[str]) -> str:
    items = "".join(
        f'<span><i style="background: var(--ent-{e.lower()})"></i>{e}</span>'
        for e in ENTITY_ORDER if e in present
    )
    return f'<div class="ner-legend">{items}</div>'


def render_highlighted(text: str, entities: list[dict]) -> str:
    """Wrap each span in a <mark>. Spans from extract() are non-overlapping and
    already in document order, so a single cursor walk suffices."""
    out, cursor = [], 0
    for ent in entities:
        out.append(html.escape(text[cursor:ent["start"]]))
        colour = f'var(--ent-{ent["entity_type"].lower()})'
        tip = html.escape(
            f'{ent["entity_type"]} · {ent["cui"]} · {ent["tui"]} · {ent["vocabulary"]}',
            quote=True,
        )
        out.append(
            f'<mark title="{tip}" style="background:{_rgba(PALETTE[ent["entity_type"]][0], 0.17)};'
            f'box-shadow: inset 0 -2px 0 {colour}">'
            f'{html.escape(ent["surface_text"])}</mark>'
        )
        cursor = ent["end"]
    out.append(html.escape(text[cursor:]))
    return f'<div class="ner-text">{"".join(out)}</div>'


def render_bars(counts: dict[str, int]) -> str:
    """Horizontal bars. Every bar is directly labelled, which is also the
    relief the palette's light-mode contrast WARN requires."""
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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Clinical NER Viewer", page_icon="🧬", layout="wide")
    st.title("🧬 Clinical NER Viewer")
    st.caption(
        "UMLS gazetteer + longest-match extraction. "
        "Hover a highlight for its CUI, semantic type and source vocabulary."
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

    entities = run_extraction(text, DEFAULT_MAX_N)
    counts = {e: sum(1 for x in entities if x["entity_type"] == e) for e in ENTITY_ORDER}
    counts = {k: v for k, v in counts.items() if v}

    if meta.get("case_id") != "pasted":
        cols = st.columns(4)
        cols[0].metric("Case", meta["case_id"])
        cols[1].metric("Age", meta["age"] or "—")
        cols[2].metric("Sex", meta["gender"] or "—")
        cols[3].metric("Entities", f"{len(entities):,}")
    else:
        cols = st.columns(3)
        cols[0].metric("Characters", f"{len(text):,}")
        cols[1].metric("Entities", f"{len(entities):,}")
        cols[2].metric("Unique concepts", f"{len({e['cui'] for e in entities}):,}")

    st.markdown(_colour_css(), unsafe_allow_html=True)
    st.markdown('<div class="ner-root">', unsafe_allow_html=True)

    if not entities:
        st.warning("No entities matched in this text.")

    st.markdown(
        render_legend(set(counts)) + render_highlighted(text, entities),
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("By entity type")
        st.markdown(render_bars(counts), unsafe_allow_html=True)
    with right:
        st.subheader("Entities")
        rows = [{**meta, **e} for e in entities]
        df = pd.DataFrame(rows, columns=CSV_COLUMNS) if rows else pd.DataFrame(columns=CSV_COLUMNS)
        st.dataframe(df.drop(columns=["article_id", "age", "gender"]),
                     width="stretch", height=340, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "⬇ Download annotations (CSV)", buf.getvalue(),
        file_name=f"entities_{meta.get('case_id') or 'pasted'}.csv", mime="text/csv",
    )


if __name__ == "__main__":
    main()
