#!/usr/bin/env python3
"""Compatibility entry point for the knowledge-graph export CLI."""

from knowledge_graph.export import GRAPHML_NS, main, to_csv_zip, to_graphml, to_json

__all__ = ["GRAPHML_NS", "to_json", "to_graphml", "to_csv_zip", "main"]


if __name__ == "__main__":
    main()
