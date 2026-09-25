"""Train a tabular baseline with separate training, validation, and test sets.

Run from the repository root: python -m portfolio.tabular --help
All learned preprocessing is fitted inside the pipeline on training rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(estimator):
    """Handle missing values and unseen categories without fitting on test data."""
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([('preprocess', preprocess), ('model', estimator)])


def prepare_data(frame, target, task, id_columns=('Id', 'id', 'customerID')):
    """Validate a dataset and remove exact duplicate records before splitting."""
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError('Column names must be unique after trimming whitespace.')
    if target not in frame:
        raise ValueError(f'Missing target column: {target}')
    if target in id_columns:
        raise ValueError('The target cannot also be an ID column.')
    frame = frame.drop(columns=list(id_columns), errors='ignore')
    before = len(frame)
    frame = frame.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    if len(frame) < 30:
        raise ValueError('At least 30 distinct labelled rows are needed for this example.')
    X = frame.drop(columns=[target])
    if X.empty:
        raise ValueError('At least one feature column is required.')
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = X[col].fillna('missing').astype(str)
    numeric = X.select_dtypes(include=np.number)
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError('Numeric features contain infinity; clean the input first.')
    if task == 'regression':
        y = pd.to_numeric(frame[target], errors='raise')
        if not np.isfinite(y.to_numpy()).all():
            raise ValueError('Regression targets must be finite numbers.')
    else:
        labels = frame[target].astype(str).str.strip().str.lower()
        mapping = {'0': 0, '0.0': 0, 'no': 0, 'false': 0,
                   '1': 1, '1.0': 1, 'yes': 1, 'true': 1}
        if not set(labels).issubset(mapping):
            raise ValueError('Binary target must use 0/1 or Yes/No labels.')
        y = labels.map(mapping)
        if y.nunique() != 2 or y.value_counts().min() < 10:
            raise ValueError('Each binary class needs at least 10 distinct rows.')
    return X, y, before - len(frame)


def metrics(model, X, y, task):
    prediction = model.predict(X)
    if task == 'regression':
        return {'mae': float(mean_absolute_error(y, prediction)),
                'rmse': float(np.sqrt(mean_squared_error(y, prediction))),
                'r2': float(r2_score(y, prediction))}
    probability = model.predict_proba(X)[:, 1]
    return {'accuracy': float(accuracy_score(y, prediction)),
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probability)),
            'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            'classification_report': classification_report(
                y, prediction, output_dict=True, zero_division=0)}


def train(frame, target, task, *, seed=42, id_columns=('Id', 'id', 'customerID')):
    X, y, removed = prepare_data(frame, target, task, id_columns)
    stratify = y if task == 'classification' else None
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=seed,
        stratify=y_dev if task == 'classification' else None)
    if task == 'regression':
        estimators = {'baseline': DummyRegressor(strategy='median'),
                      'ridge': Ridge(alpha=1.0),
                      'random_forest': RandomForestRegressor(
                          n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)}
    else:
        estimators = {'baseline': DummyClassifier(strategy='prior'),
                      'logistic_regression': LogisticRegression(max_iter=2000, random_state=seed),
                      'random_forest': RandomForestClassifier(
                          n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)}
    models, validation = {}, {}
    for name, estimator in estimators.items():
        models[name] = build_pipeline(estimator).fit(X_train, y_train)
        validation[name] = metrics(models[name], X_val, y_val, task)
    metric = 'rmse' if task == 'regression' else 'roc_auc'
    select = min if task == 'regression' else max
    selected = select(validation, key=lambda name: validation[name][metric])
    final_model = models[selected].fit(X_dev, y_dev)
    baseline = build_pipeline(estimators['baseline']).fit(X_dev, y_dev)
    report = {
        'task': task, 'target': target, 'seed': seed,
        'rows': {'train': len(X_train), 'validation': len(X_val), 'test': len(X_test)},
        'rows_removed_before_split': removed,
        'selection_metric': metric, 'selected_on': 'validation',
        'selected_model': selected, 'validation': validation,
        'test': {'selected_model': metrics(final_model, X_test, y_test, task),
                 'baseline': metrics(baseline, X_test, y_test, task)},
        'feature_columns': X.columns.tolist(),
        'limitations': 'Random holdout; use grouped or temporal splits for related records or time-dependent deployment.',
    }
    predictions = pd.DataFrame({'row_index': y_test.index,
                                'actual': y_test.to_numpy(),
                                'predicted': final_model.predict(X_test)})
    if task == 'classification':
        predictions['probability_class_1'] = final_model.predict_proba(X_test)[:, 1]
    return final_model, report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--task', choices=['regression', 'classification'], required=True)
    parser.add_argument('--data-kind', choices=['external', 'synthetic'], required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tabular'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--id-columns', nargs='*', default=['Id', 'id', 'customerID'])
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    model, report, predictions = train(frame, args.target, args.task,
                                       seed=args.seed, id_columns=args.id_columns)
    report['dataset'] = {'filename': args.csv.name, 'kind': args.data_kind,
                         'sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest()}
    report['environment'] = {name: importlib.metadata.version(name)
                             for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    joblib.dump(model, args.output / 'model.joblib')
    print(json.dumps({'selected_model': report['selected_model'], 'test': report['test'],
                      'data_kind': args.data_kind, 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
