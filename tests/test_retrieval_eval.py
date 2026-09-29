import unittest
from portfolio.retrieval_eval import evaluate, score_cases


class RetrievalEvaluationTests(unittest.TestCase):
    def test_source_metrics_deduplicate_chunks_and_count_false_accepts(self):
        cases = [{'relevant_sources': ['a', 'b']}, {'relevant_sources': []}, {'relevant_sources': []}]
        predictions = [[{'source': 'x'}, {'source': 'a'}, {'source': 'a'}], [], [{'source': 'b'}]]
        scores = score_cases(cases, predictions)
        self.assertEqual(scores['source_recall_at_k'], 0.5)
        self.assertEqual(scores['source_mrr_at_k'], 0.5)
        self.assertEqual(scores['unanswerable_abstention_rate'], 0.5)
        self.assertEqual(scores['unanswerable_false_accepts'], 1)

    def test_followup_context_and_no_overlap_abstention(self):
        data = {'documents': {'a': 'oranges citrus fruit', 'b': 'engines trucks'}, 'cases': [
            {'question': 'What about those?', 'previous_question': 'oranges citrus', 'relevant_sources': ['a']},
            {'question': 'astronomy', 'relevant_sources': []}]}
        result = evaluate(data)
        self.assertEqual(result['outputs'][0]['retrieved'][0]['source'], 'a')
        self.assertEqual(result['outputs'][1]['retrieved'], [])

    def test_bad_labels_and_mismatched_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            score_cases([{'relevant_sources': []}], [])
        with self.assertRaises(ValueError):
            evaluate({'documents': {'a': 'sample text'}, 'cases': [
                {'question': 'sample', 'relevant_sources': ['missing']}]})
