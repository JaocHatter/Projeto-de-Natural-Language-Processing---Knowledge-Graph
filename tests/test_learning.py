"""Tests for the Project 2 learning package. Run with
python3 -m unittest discover -s tests (standard library only)."""

import unittest

from clinical_kg.learning.dataset import (Associations, DEFAULT_GROUPS,
                                          FEATURE_GROUPS, candidate_features,
                                          context_tokens, load_examples)
from clinical_kg.learning.experiment import (evaluate_binary, folds, prf,
                                             run_p1_baseline, with_lm)
from clinical_kg.learning.models import (LogisticRegression, NaiveBayes,
                                         OneVsRest)
from clinical_kg.learning.ngram import ContextScorer, NgramLM, tokenize


class NgramTests(unittest.TestCase):
    """The corpus is 81.6% hapax bigrams, so smoothing is not optional."""

    def test_unseen_bigram_keeps_a_usable_probability(self):
        model = NgramLM().fit([tokenize("the patient was treated with aspirin")])
        seen = model.log_probability(tokenize("treated with"))
        unseen = model.log_probability(tokenize("xyzzy frobnicate"))
        self.assertLess(unseen, seen)
        self.assertGreater(unseen, float("-inf"))   # never a zero probability

    def test_probabilities_are_a_distribution(self):
        model = NgramLM().fit([tokenize("fever and chills and rash")])
        total = sum(model._conditional("and", word) for word in model.unigram)
        self.assertLessEqual(total, 1.0 + 1e-6)
        self.assertGreater(total, 0.0)

    def test_score_is_the_mean_log_probability_per_bigram(self):
        """Length normalisation is the contract: the score is a mean, not a sum.

        Without it a long context would score lower than a short one purely for
        having more terms, and the inter-entity contexts here range from one
        token to twenty.
        """
        import math
        model = NgramLM().fit([tokenize("fever and chills and rash")])
        document = tokenize("fever and rash")
        expected = sum(math.log(model._conditional(a, b))
                       for a, b in zip(document, document[1:])) / (len(document) - 1)
        self.assertAlmostEqual(model.log_probability(document), expected, places=9)

    def test_empty_context_does_not_crash(self):
        model = NgramLM().fit([tokenize("fever and chills")])
        self.assertEqual(model.log_probability(["<s>"]), 0.0)

    def test_scorer_prefers_the_class_it_was_trained_on(self):
        scorer = ContextScorer().fit(
            [tokenize("was treated with"), tokenize("treated with")],
            [tokenize("of the"), tokenize("in the")])
        self.assertGreater(scorer.score(tokenize("treated with")), 0)
        self.assertLess(scorer.score(tokenize("of the")), 0)

    def test_score_is_bounded(self):
        scorer = ContextScorer().fit([tokenize("a b c")] * 20, [tokenize("x y z")])
        for context in (tokenize("a b c"), tokenize("x y z"), tokenize("q")):
            self.assertLessEqual(abs(scorer.score(context)), 1.0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        # Two features that separate the classes perfectly.
        self.examples = ([{"good": 1.0} for _ in range(20)]
                         + [{"bad": 1.0} for _ in range(20)])
        self.labels = [1] * 20 + [0] * 20

    def test_logistic_regression_learns_a_separable_problem(self):
        model = LogisticRegression().fit(self.examples, self.labels)
        self.assertEqual(model.predict({"good": 1.0}), 1)
        self.assertEqual(model.predict({"bad": 1.0}), 0)
        self.assertGreater(model.weights["good"], model.weights["bad"])

    def test_naive_bayes_learns_the_same_problem(self):
        model = NaiveBayes().fit(self.examples, self.labels)
        self.assertEqual(model.predict({"good": 1.0}), 1)
        self.assertEqual(model.predict({"bad": 1.0}), 0)

    def test_naive_bayes_probabilities_sum_to_one(self):
        model = NaiveBayes().fit(self.examples, self.labels)
        self.assertAlmostEqual(sum(model.predict_proba({"good": 1.0}).values()), 1.0)

    def test_unseen_feature_does_not_crash_either_model(self):
        for model in (LogisticRegression().fit(self.examples, self.labels),
                      NaiveBayes().fit(self.examples, self.labels)):
            self.assertIn(model.predict({"never_seen": 1.0}), (0, 1))

    def test_one_vs_rest_handles_more_than_two_classes(self):
        examples = [{"a": 1.0}] * 10 + [{"b": 1.0}] * 10 + [{"c": 1.0}] * 10
        labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
        model = OneVsRest().fit(examples, labels)
        self.assertEqual(model.classes, ["A", "B", "C"])
        for feature, expected in (("a", "A"), ("b", "B"), ("c", "C")):
            self.assertEqual(model.predict({feature: 1.0}), expected)


class AssociationTests(unittest.TestCase):
    def test_rare_bigram_scores_zero_rather_than_the_maximum(self):
        """Raw PMI peaks on hapaxes; the cutoff is what makes it usable here."""
        texts = ["alpha beta " * 10 + "gamma delta"]      # gamma delta occurs once
        assoc = Associations(texts, background_texts=[])
        self.assertEqual(assoc.pmi("gamma", "delta"), 0.0)
        self.assertGreater(assoc.pmi("alpha", "beta"), 0.0)

    def test_llr_is_signed_and_finite(self):
        assoc = Associations(["alpha beta " * 10], background_texts=[])
        self.assertGreater(assoc.llr("alpha", "beta"), 0.0)
        self.assertEqual(assoc.llr("nothing", "here"), 0.0)


class DatasetTests(unittest.TestCase):
    """These read the committed corpus, so they also guard the data files."""

    @classmethod
    def setUpClass(cls):
        cls.examples = load_examples(("struct", "assoc"))

    def test_every_judged_candidate_is_loaded_exactly_once(self):
        keys = [ex.key for ex in self.examples]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(len(keys), 300)

    def test_labels_are_consistent(self):
        for ex in self.examples:
            self.assertIn(ex.is_relation, (0, 1))
            if ex.is_relation:
                self.assertTrue(ex.relation)
            else:
                self.assertEqual(ex.relation, "")

    def test_every_example_carries_a_context(self):
        for ex in self.examples:
            self.assertGreaterEqual(len(ex.context), 2)   # at least <s> </s>

    def test_feature_groups_are_honoured(self):
        only_struct = load_examples(("struct",))
        names = {name for ex in only_struct for name in ex.features}
        self.assertFalse({n for n in names if n.startswith(("btw:", "pos_btw:"))})
        self.assertTrue({n for n in names if n.startswith("pair:")})

    def test_default_groups_are_the_measured_best(self):
        """lex and pos were measured to hurt; the default must not include them."""
        self.assertEqual(set(DEFAULT_GROUPS), {"struct", "assoc"})
        self.assertTrue(set(DEFAULT_GROUPS) <= set(FEATURE_GROUPS))

    def test_context_tokens_are_bracketed(self):
        tokens = context_tokens("revealed a mass in the spleen", 0, 15)
        self.assertEqual(tokens[0], "<s>")
        self.assertEqual(tokens[-1], "</s>")


class ValidationTests(unittest.TestCase):
    """The protocol itself: folds must not leak a case into its own training."""

    @classmethod
    def setUpClass(cls):
        cls.examples = load_examples(("struct", "assoc"))

    def test_folds_are_one_case_each_and_disjoint(self):
        seen = set()
        for case_id, train, test in folds(self.examples):
            self.assertEqual({ex.case_id for ex in test}, {case_id})
            self.assertNotIn(case_id, {ex.case_id for ex in train})
            self.assertEqual(len(train) + len(test), len(self.examples))
            seen.add(case_id)
        self.assertEqual(seen, {ex.case_id for ex in self.examples})

    def test_language_model_feature_is_generated_out_of_fold(self):
        """Training values must come from models that never read that context.

        Scored in-fold the feature looks near-perfect and the classifier learns
        to over-trust it; measured, that mistake costs more than the feature is
        worth (F1 0.670 against 0.723).
        """
        _, train, test = next(iter(folds(self.examples)))
        train_feats, test_feats = with_lm(train, test, 1.0, None)
        self.assertEqual(len(train_feats), len(train))
        self.assertEqual(len(test_feats), len(test))
        self.assertTrue(all("lm_score" in f for f in train_feats + test_feats))
        scored = [f["lm_score"] for f in train_feats]
        # An in-fold score would put nearly every positive at the ceiling.
        self.assertLess(sum(1 for s in scored if s >= 0.99) / len(scored), 0.5)

    def test_prf_and_counts_agree(self):
        predictions = [(1, 1), (1, 0), (0, 1), (0, 0)]
        self.assertEqual(evaluate_binary(predictions), (1, 1, 1))
        precision, recall, f1 = prf(1, 1, 1)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5)

    def test_project_1_baseline_reproduces_its_own_evaluation(self):
        """The baseline here must be the same system evaluate-relations scores."""
        predictions, _ = run_p1_baseline(self.examples, "binary")
        _, _, f1 = prf(*evaluate_binary(predictions))
        self.assertAlmostEqual(f1, 0.628, places=2)


if __name__ == "__main__":
    unittest.main()
