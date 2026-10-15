#!/usr/bin/env python3
"""N-gram language models over the text between two entities.

Project 1 decided a relation from a hand-written list of 140 trigger words.
This module replaces that list with a question a language model can answer:
*given the words between these two entities, which is more likely -- the way
people write when there IS a relation, or the way they write when there is
not?*

Two models are trained, one per answer, and the difference of their
log-probabilities becomes a single feature. That compression matters. Measured
on this corpus, handing the classifier ~2,900 individual n-gram indicators
makes it worse (F1 0.708 -> 0.677): 413 examples cannot support that many
parameters. The same evidence, condensed into one number, helps instead.

Smoothing is the whole problem here. 81.6% of this corpus's bigram types occur
exactly once, so an unsmoothed model assigns probability zero to almost every
sentence it has not literally seen. Two defences, both standard:

  * add-k smoothing, interpolated with the unigram distribution;
  * interpolation with a second, larger in-domain corpus -- the UMLS
    definitions of the concepts this corpus actually mentions
    (`data/interim/umls_definitions.csv`, 959 definitions, 29,718 words).

The second is the same interpolation idea applied across corpora rather than
across n-gram orders, and it is aimed at a measured gap: "caused by" occurs
once in the 50 case reports and 32 times in the definitions.
"""

import collections
import csv
import math
from pathlib import Path

from clinical_kg.extraction.entities import TOKEN_RE, mask_figure_refs
from clinical_kg.paths import PROJECT_ROOT

DEFAULT_DEFINITIONS = PROJECT_ROOT / "data/interim/umls_definitions.csv"

BOS, EOS = "<s>", "</s>"
# Weight of the bigram estimate against the unigram backoff.
BIGRAM_WEIGHT = 0.7
# add-k: k below 1 is gentler than Laplace, which over-smooths a small corpus.
DEFAULT_K = 0.4


def tokenize(text: str) -> list[str]:
    """Tokens as the rest of the pipeline sees them, bracketed by sentence marks."""
    return [BOS] + [m.group().lower()
                    for m in TOKEN_RE.finditer(mask_figure_refs(text))] + [EOS]


def load_definitions(path: Path = DEFAULT_DEFINITIONS) -> list[list[str]]:
    """UMLS definitions as token lists, or [] when they were never extracted."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [tokenize(row["definition"]) for row in csv.DictReader(f)
                if row.get("definition", "").strip()]


class NgramLM:
    """Bigram model with add-k smoothing, unigram backoff and optional
    interpolation with a background corpus.

        P(w | prev) = lambda * P_foreground(w | prev)
                    + (1 - lambda) * P_background(w | prev)

    `lam` is the knob the experiment sweeps: 1.0 ignores the background corpus
    entirely (the model this project started with), 0.0 ignores the examples.
    """

    def __init__(self, k: float = DEFAULT_K, lam: float = 1.0, background=None):
        self.k, self.lam = k, lam
        self.background = background

    def fit(self, documents):
        self.unigram = collections.Counter(w for d in documents for w in d)
        self.bigram = collections.Counter(b for d in documents
                                          for b in zip(d, d[1:]))
        self.total = sum(self.unigram.values())
        self.vocabulary = len(self.unigram) + 1
        if self.background is not None and self.lam < 1.0:
            self.background.fit_if_needed()
        return self

    def _conditional(self, previous: str, word: str) -> float:
        bigram = ((self.bigram[(previous, word)] + self.k)
                  / (self.unigram[previous] + self.k * self.vocabulary))
        unigram = ((self.unigram[word] + self.k)
                   / (self.total + self.k * self.vocabulary))
        return BIGRAM_WEIGHT * bigram + (1 - BIGRAM_WEIGHT) * unigram

    def log_probability(self, document) -> float:
        """Mean log-probability per token, so length does not dominate."""
        if len(document) < 2:
            return 0.0
        total = 0.0
        for previous, word in zip(document, document[1:]):
            probability = self._conditional(previous, word)
            if self.background is not None and self.lam < 1.0:
                probability = (self.lam * probability
                               + (1 - self.lam) * self.background.conditional(previous, word))
            total += math.log(probability)
        return total / (len(document) - 1)


class BackgroundLM:
    """The UMLS-definition side of the interpolation, fitted once and shared.

    Kept separate from NgramLM because it never changes between folds: the
    definitions are not part of the training split, so reusing them across
    folds leaks nothing about the held-out case.
    """

    def __init__(self, documents, k: float = DEFAULT_K):
        self.documents, self.k = documents, k
        self._fitted = False

    def fit_if_needed(self):
        if self._fitted:
            return self
        self.unigram = collections.Counter(w for d in self.documents for w in d)
        self.bigram = collections.Counter(b for d in self.documents
                                          for b in zip(d, d[1:]))
        self.total = max(sum(self.unigram.values()), 1)
        self.vocabulary = len(self.unigram) + 1
        self._fitted = True
        return self

    def conditional(self, previous: str, word: str) -> float:
        self.fit_if_needed()
        bigram = ((self.bigram[(previous, word)] + self.k)
                  / (self.unigram[previous] + self.k * self.vocabulary))
        unigram = ((self.unigram[word] + self.k)
                   / (self.total + self.k * self.vocabulary))
        return BIGRAM_WEIGHT * bigram + (1 - BIGRAM_WEIGHT) * unigram


class ContextScorer:
    """Trains one language model per answer and scores a context by their ratio.

    `fit` takes the contexts of examples that are relations and those that are
    not; `score` returns a number above zero when the relation model finds the
    context more likely. The value is squashed into [-1, 1] because the
    classifier's other features are indicators.
    """

    CLIP = 3.0

    def __init__(self, lam: float = 1.0, background=None, k: float = DEFAULT_K):
        self.lam, self.background, self.k = lam, background, k

    def fit(self, positive_contexts, negative_contexts):
        self.positive = NgramLM(self.k, self.lam, self.background).fit(positive_contexts)
        self.negative = NgramLM(self.k, self.lam, self.background).fit(negative_contexts)
        return self

    def score(self, context) -> float:
        difference = (self.positive.log_probability(context)
                      - self.negative.log_probability(context))
        return max(min(difference, self.CLIP), -self.CLIP) / self.CLIP
