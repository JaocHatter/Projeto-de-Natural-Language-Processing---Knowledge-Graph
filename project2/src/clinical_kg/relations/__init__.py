"""Clinical relation extraction with a hand-weighted linear-chain CRF.

Layout:

    core/       the algorithm: text_layer, pos, features, crf, coreference
    extract     candidate generation, attachment, typing, assertion, CLI --
                the one module that runs in production
    ontology    UMLS's own relations (MRREL.RRF), a second evidence source
    tools/      annotate, evaluate -- developer-facing, not part of the
                production pipeline

The weights are set by hand, not learned: this project has no labeled relation
data. See README.md in this directory for the process and the reasoning.
"""

from .core.features import ASSERTIONS, RELATIONS, WEIGHTS
from .extract import analyze_case, analyze_sentence

__all__ = ["RELATIONS", "ASSERTIONS", "WEIGHTS", "analyze_case", "analyze_sentence"]
