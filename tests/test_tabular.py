import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from portfolio.tabular import build_pipeline, prepare_data, train


class TabularTests(unittest.TestCase):
    def frame(self, classification=False):
        rng = np.random.default_rng(12)
        size = rng.normal(100, 20, 180)
        return pd.DataFrame({'Id': np.arange(180), 'area': size,
                             'location': np.where(size > 100, 'city', 'town'),
                             'target': (size > 100).astype(int) if classification else size*30 + rng.normal(0, 15, 180)})

    def test_ids_are_excluded_and_duplicates_removed(self):
        df = self.frame()
        duplicate = df.iloc[[0]].assign(Id=999)
        X, y, removed = prepare_data(pd.concat([df, duplicate]), 'target', 'regression')
        self.assertNotIn('Id', X)
        self.assertNotIn('target', X)
        self.assertEqual(removed, 1)
        self.assertEqual(len(y), 180)

    def test_prediction_does_not_refit_imputer_on_holdout(self):
        train_X = pd.DataFrame({'x': [1.0, 3.0, np.nan], 'category': ['a', 'a', 'b']})
        pipe = build_pipeline(Ridge()).fit(train_X, [2, 6, 4])
        imputer = pipe.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
        self.assertEqual(imputer.statistics_[0], 2.0)
        pred = pipe.predict(pd.DataFrame({'x': [np.nan, 10000.0], 'category': ['unseen', 'a']}))
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(imputer.statistics_[0], 2.0)

    def test_rejects_non_binary_labels(self):
        df = self.frame(True)
        df.loc[0, 'target'] = 2
        with self.assertRaises(ValueError):
            prepare_data(df, 'target', 'classification')

    def test_rejects_invalid_regression_targets(self):
        df = self.frame()
        df.loc[0, 'target'] = np.inf
        with self.assertRaises(ValueError):
            prepare_data(df, 'target', 'regression')

    def test_end_to_end_regression_and_serialization(self):
        df = self.frame()
        model, report, pred = train(df, 'target', 'regression')
        self.assertEqual(report['rows'], {'train': 108, 'validation': 36, 'test': 36})
        self.assertEqual(report['selected_on'], 'validation')
        self.assertEqual(len(pred), 36)
        self.assertLess(report['test']['selected_model']['rmse'], report['test']['baseline']['rmse'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(model.predict(df.drop(columns=['target', 'Id'])),
                                       joblib.load(path).predict(df.drop(columns=['target', 'Id'])))

    def test_end_to_end_binary_classification(self):
        _, report, pred = train(self.frame(True), 'target', 'classification')
        self.assertTrue(pred['probability_class_1'].between(0, 1).all())
        self.assertEqual(sum(map(sum, report['test']['selected_model']['confusion_matrix'])), 36)
        self.assertIn(report['selected_model'], report['validation'])


if __name__ == '__main__':
    unittest.main()
