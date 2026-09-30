"""Exploratory validation-only threshold selection on existing synthetic splits."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from portfolio.churn import make_sample
from portfolio.tabular import build_pipeline, prepare_data

THRESHOLDS = tuple(value / 100 for value in range(10, 91, 5))


def decision_metrics(y, probabilities, threshold):
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {'threshold': threshold, 'accuracy': float(accuracy_score(y, predicted)),
            'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def choose_threshold(y_validation, probabilities):
    scores = [decision_metrics(y_validation, probabilities, threshold) for threshold in THRESHOLDS]
    # Predetermined tie rule: closest to 0.5, then lower threshold.
    winner = min(scores, key=lambda score: (-score['f1'], abs(score['threshold'] - 0.5), score['threshold']))
    return winner['threshold'], scores


def run(output):
    csv = make_sample(7043, 42).to_csv(index=False, lineterminator='\n')
    X, y, _ = prepare_data(pd.read_csv(io.StringIO(csv)), 'Churn', 'classification')
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev)
    model = build_pipeline(RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=1))
    model.fit(X_train, y_train)
    val_probability = model.predict_proba(X_val)[:, 1]
    chosen, validation = choose_threshold(y_val, val_probability)
    # Freeze both model and threshold before computing test predictions.
    probability = model.predict_proba(X_test)[:, 1]
    report = {
        'dataset': {'kind': 'synthetic', 'generation_seed': 42, 'csv_sha256': hashlib.sha256(csv.encode()).hexdigest()},
        'split_seed': 42, 'rows': {'train': len(X_train), 'validation': len(X_val), 'test': len(X_test)},
        'model': 'RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=1)',
        'model_choice': 'Fixed from the previously published seed-42 model-selection result; no new model search.',
        'protocol': 'Fit training rows only; maximize validation F1 over predeclared grid; freeze model without refitting; compare both thresholds on identical test probabilities.',
        'tie_rule': 'Closest to 0.5, then lower threshold',
        'validation_grid': validation, 'chosen_threshold': chosen,
        'test': {'default': decision_metrics(y_test, probability, 0.5),
                 'validation_selected': decision_metrics(y_test, probability, chosen),
                 'roc_auc': float(roc_auc_score(y_test, probability))},
        'environment': {'python': platform.python_version(), **{name: importlib.metadata.version(name)
                        for name in ('numpy', 'pandas', 'scikit-learn')}},
        'limitations': 'Exploratory reuse of previously inspected synthetic holdouts, not fresh confirmatory evidence. F1 is not a business cost function. No refit, so scores differ from the earlier train-plus-validation model. No calibration or real-customer claim.',
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for split, target, values in [('validation', y_val, val_probability), ('test', y_test, probability)]:
        pd.DataFrame({'row_index': target.index, 'actual': target.to_numpy(), 'probability_class_1': values,
                      'default_prediction': (values >= 0.5).astype(int),
                      'selected_prediction': (values >= chosen).astype(int)}).to_csv(output / f'{split}-predictions.csv', index=False)
    print(json.dumps({'chosen_threshold': chosen, 'test': report['test']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/churn-threshold'))
    run(parser.parse_args().output)
