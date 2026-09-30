import unittest
from portfolio.churn_threshold import choose_threshold, decision_metrics


class ThresholdTests(unittest.TestCase):
    def test_validation_labels_determine_threshold(self):
        threshold, scores = choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(threshold, 0.3)
        self.assertEqual(max(row['f1'] for row in scores), 1.0)

    def test_ties_prefer_default_and_boundary_is_inclusive(self):
        threshold, _ = choose_threshold([0, 1], [0.0, 1.0])
        self.assertEqual(threshold, 0.5)
        self.assertEqual(decision_metrics([0, 1], [0.49, 0.5], 0.5)['confusion_matrix'], [[1, 0], [0, 1]])
