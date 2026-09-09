"""Clinical relation extraction with a hand-weighted linear-chain CRF.

Layers, in dependency order:

    text_layer  sentences, clauses and tokens, with character offsets
    pos         six coarse part-of-speech tags (lexicon + suffix backoff)
    features    the trigger lexicon, cue lists and the WEIGHTS table
    crf         emission/transition potentials and Viterbi decoding
    extract     candidate generation, attachment, typing, assertion, CLI
    annotate    interactive gold-set annotation (evaluation only)
    evaluate    precision/recall, threshold sweep and ablation

The weights are set by hand, not learned: this project has no labeled relation
data. See README.md in this directory for the process and the reasoning.
"""

from .features import ASSERTIONS, RELATIONS, WEIGHTS
from .extract import analyze_case, analyze_sentence

__all__ = ["RELATIONS", "ASSERTIONS", "WEIGHTS", "analyze_case", "analyze_sentence"]
