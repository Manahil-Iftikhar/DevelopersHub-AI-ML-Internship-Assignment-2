import unittest
from portfolio.ticket_eval import score


class TicketEvaluationTests(unittest.TestCase):
    def test_invalid_extra_text_is_not_exact_match(self):
        result = score([{'expected': ['Billing'], 'tags': ['Billing'], 'needs_review': True}])
        self.assertEqual(result['exact_match'], 0)
        self.assertEqual(result['review_rate'], 1)

    def test_multilabel_precision_recall_and_empty_output(self):
        result = score([
            {'expected': ['Billing'], 'tags': ['Billing', 'Technical Issue'], 'needs_review': False},
            {'expected': ['Login Problem'], 'tags': [], 'needs_review': True}])
        self.assertEqual(result['micro_precision'], .5)
        self.assertEqual(result['micro_recall'], .5)
        self.assertEqual(result['micro_f1'], .5)
        self.assertEqual(result['exact_match'], 0)

    def test_reject_empty_evaluation(self):
        with self.assertRaises(ValueError):
            score([])
