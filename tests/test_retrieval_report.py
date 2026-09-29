import copy
import unittest
from portfolio.retrieval_eval import verify_report


class RetrievalReportTests(unittest.TestCase):
    def fixture(self):
        return {'dataset_sha256': 'original', 'metrics': {'recall': 1.0},
                'outputs': [{'source': 'a', 'similarity': 0.5}, {'source': 'b', 'similarity': 0.3}],
                'environment': {'python': 'historical'}}

    def test_runtime_versions_and_tiny_float_drift_are_allowed(self):
        original = self.fixture()
        fresh = copy.deepcopy(original)
        fresh['environment'] = {'python': 'current'}
        fresh['outputs'][0]['similarity'] += 1e-12
        verify_report(original, fresh)

    def test_changed_data_metrics_and_rankings_fail(self):
        original = self.fixture()
        for field, value in [('dataset_sha256', 'changed'),
                             ('metrics', {'recall': 0.5}),
                             ('outputs', list(reversed(original['outputs'])))]:
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                with self.assertRaises(ValueError):
                    verify_report(original, changed)

    def test_missing_evidence_and_nonfinite_scores_fail(self):
        original = self.fixture()
        missing = copy.deepcopy(original)
        del missing['outputs']
        with self.assertRaises(ValueError):
            verify_report(original, missing)
        invalid = copy.deepcopy(original)
        invalid['outputs'][0]['similarity'] = float('nan')
        with self.assertRaises(ValueError):
            verify_report(original, invalid)
