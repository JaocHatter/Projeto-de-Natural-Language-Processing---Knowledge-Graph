#!/usr/bin/env python3
"""One entry point for every pipeline stage.

Each subcommand delegates to that module's own `main()`, which keeps its own
argparse definition. The dispatcher therefore adds no flags of its own and
cannot drift from what the stages actually accept -- pass `--help` after a
subcommand to see it.

    clinical-kg extract-relations --explain PMC10106591_01
    python3 -m clinical_kg extract-relations --explain PMC10106591_01

The modules are imported lazily so that a stdlib-only stage still runs when the
Streamlit dependencies of the viewer are not installed.
"""

import importlib
import sys

# subcommand -> module providing main(). Ordered as the pipeline runs.
COMMANDS = {
    "clean-cases":          ("clinical_kg.corpus.clean",         "clean the raw case corpus"),
    "build-gazetteer":      ("clinical_kg.gazetteer.build",      "build the UMLS SQLite gazetteer"),
    "extract-entities":     ("clinical_kg.extraction.entities",  "extract entities from the corpus"),
    "extract-measurements": ("clinical_kg.extraction.measurements", "extract and link measurements"),
    "pos-lexicon":          ("clinical_kg.relations.pos",        "dump the POS lexicon and coverage"),
    "extract-relations":    ("clinical_kg.relations.extract",    "extract typed clinical relations"),
    "annotate-relations":   ("clinical_kg.relations.annotate",   "annotate the gold relation set"),
    "evaluate-relations":   ("clinical_kg.relations.evaluate",   "score relations against the gold set"),
    "export-graph":         ("clinical_kg.graph.export",         "export graphs as JSON/GraphML/CSV"),
}

USAGE = "usage: clinical-kg <command> [options]\n\ncommands:\n" + "\n".join(
    f"  {name:<22} {help_text}" for name, (_, help_text) in COMMANDS.items()
) + "\n\nRun 'clinical-kg <command> --help' for a command's own options.\n"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return
    command = argv[0]
    if command not in COMMANDS:
        print(f"clinical-kg: unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
        sys.exit(2)

    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    # The stage parses sys.argv itself, so present it as if invoked directly.
    # argparse derives its program name from sys.argv[0], except that since
    # 3.14 it reports "python3 -m <pkg>" when __main__ was started with -m,
    # which would print usage without the subcommand -- copying that line would
    # give a command that does nothing. Clearing __spec__ restores argv[0].
    sys.argv = [f"clinical-kg {command}", *argv[1:]]
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        main_module.__spec__ = None
    module.main()


if __name__ == "__main__":
    main()
