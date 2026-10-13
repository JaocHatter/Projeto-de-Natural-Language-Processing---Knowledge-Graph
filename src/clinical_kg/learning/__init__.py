"""Project 2: learn relation extraction from data instead of hand-set rules.

Project 1 scored candidate entity pairs with a linear-chain CRF whose potentials
were written by hand (`relations.core.features.WEIGHTS`, plus a 140-entry trigger
lexicon). This package keeps that candidate generator and replaces only the
*decision function*: a classifier trained on the 413 hand-judged candidates in
`data/interim/gold_relations.csv`, over n-gram and structural features.

Holding the candidate generator fixed is what makes the comparison controlled --
both systems see exactly the same pairs, so any difference in P/R/F1 comes from
how the pair is scored, not from which pairs were considered.

    dataset     candidates -> feature dicts + labels (n-grams, POS, PMI, ...)
    models      Naive Bayes written from scratch; optional scikit-learn models
    experiment  leave-one-case-out evaluation, ablations, comparison with P1
"""
