"""Knowledge graph of clinical case reports.

Layers, in dependency order -- imports only ever point down this list:

    paths        shared corpus, gazetteer and output locations
    corpus       case cleaning and per-patient records
    gazetteer    the UMLS term dictionary (SQLite)
    extraction   entity and measurement extraction
    relations    typed clinical relations (hand-weighted linear-chain CRF)
    graph        graph construction, export and viewer
    app          Streamlit front end
    cli          one dispatcher over every stage

Only `app` and `graph.ui` require third-party packages; the pipeline itself is
stdlib-only.
"""

__version__ = "0.1.0"
