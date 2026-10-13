#!/usr/bin/env python3
"""Classifiers over the sparse feature dicts produced by `dataset.py`.

`NaiveBayes` and `LogisticRegression` are written from scratch, in the standard
library, for the same reason the rest of this project is: a model whose update
rule is visible in twenty lines can be audited, and the extraction pipeline
keeps running with no third-party dependency. scikit-learn models (SVM, random
forest) are offered through `sklearn_model`, which returns None when the library
is absent so the experiment degrades to the hand-written pair instead of failing.
"""

import collections
import math
import random


class NaiveBayes:
    """Multinomial Naive Bayes over binary/count features, Laplace-smoothed.

    P(class | features) is proportional to P(class) * prod P(feature | class).
    With 413 examples and tens of thousands of n-gram features, the independence
    assumption is wrong but harmless: it is exactly the regime where Naive Bayes
    is known to work well, because only the ranking of the posterior matters.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.log_prior: dict = {}
        self.log_likelihood: dict = {}
        self.vocabulary: set = set()
        self.classes: list = []

    def fit(self, examples, labels):
        counts = collections.Counter(labels)
        total = len(labels)
        self.classes = sorted(counts)
        self.vocabulary = {name for feats in examples for name in feats}
        size = len(self.vocabulary)

        mass = {c: collections.defaultdict(float) for c in self.classes}
        totals = {c: 0.0 for c in self.classes}
        for feats, label in zip(examples, labels):
            for name, value in feats.items():
                if value:
                    mass[label][name] += value
                    totals[label] += value

        self.log_prior = {c: math.log(counts[c] / total) for c in self.classes}
        self.log_likelihood = {}
        for c in self.classes:
            denominator = totals[c] + self.alpha * size
            self.log_likelihood[c] = {
                name: math.log((mass[c][name] + self.alpha) / denominator)
                for name in self.vocabulary
            }
            self.log_likelihood[c]["__unseen__"] = math.log(self.alpha / denominator)
        return self

    def _scores(self, feats):
        out = {}
        for c in self.classes:
            table = self.log_likelihood[c]
            score = self.log_prior[c]
            for name, value in feats.items():
                if value:
                    score += value * table.get(name, table["__unseen__"])
            out[c] = score
        return out

    def predict(self, feats):
        return max(self._scores(feats).items(), key=lambda kv: kv[1])[0]

    def predict_proba(self, feats):
        """Posterior over classes, normalised in log space."""
        scores = self._scores(feats)
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        total = sum(exp.values())
        return {c: v / total for c, v in exp.items()}


class LogisticRegression:
    """Binary logistic regression trained by SGD with L2 regularisation.

    Unlike Naive Bayes it does not assume the features are independent, which
    matters here: the n-gram features overlap heavily by construction ("treated"
    and "treated_with" fire together).
    """

    def __init__(self, epochs: int = 40, lr: float = 0.5, l2: float = 1e-4, seed: int = 0):
        self.epochs, self.lr, self.l2, self.seed = epochs, lr, l2, seed
        self.weights: dict = {}
        self.bias = 0.0

    def fit(self, examples, labels):
        self.weights, self.bias = {}, 0.0
        order = list(range(len(examples)))
        rng = random.Random(self.seed)
        for epoch in range(self.epochs):
            rng.shuffle(order)
            rate = self.lr / (1.0 + epoch)
            for i in order:
                feats, y = examples[i], labels[i]
                error = self._sigmoid(self._raw(feats)) - y
                self.bias -= rate * error
                for name, value in feats.items():
                    if value:
                        weight = self.weights.get(name, 0.0)
                        self.weights[name] = weight - rate * (error * value + self.l2 * weight)
        return self

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp = math.exp(z)
        return exp / (1.0 + exp)

    def _raw(self, feats):
        return self.bias + sum(self.weights.get(n, 0.0) * v for n, v in feats.items() if v)

    def predict(self, feats):
        return 1 if self._sigmoid(self._raw(feats)) >= 0.5 else 0

    def predict_proba(self, feats):
        p = self._sigmoid(self._raw(feats))
        return {0: 1.0 - p, 1: p}

    def top_features(self, n=15):
        ranked = sorted(self.weights.items(), key=lambda kv: -abs(kv[1]))
        return ranked[:n]


def sklearn_model(name: str, seed: int = 0):
    """An sklearn classifier wrapped in the same predict/fit interface, or None
    when scikit-learn is not installed."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.svm import LinearSVC
    except ImportError:
        return None

    builders = {
        "svm": lambda: LinearSVC(C=0.5, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1),
    }
    if name not in builders:
        return None

    class Wrapped:
        def __init__(self):
            # Dense on purpose: the matrix is at most a few hundred rows by a few
            # thousand columns, and DictVectorizer's sparse output uses int64
            # indices, which several sklearn estimators reject.
            self.vectorizer = DictVectorizer(sparse=False)
            self.model = builders[name]()

        def fit(self, examples, labels):
            self.model.fit(self.vectorizer.fit_transform(examples), labels)
            return self

        def predict(self, feats):
            return self.model.predict(self.vectorizer.transform([feats]))[0]

    return Wrapped()
