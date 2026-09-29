"""Repeat the unchanged churn selection workflow on one fixed synthetic dataset.

The seed changes both holdout assignments and stochastic estimator initialization.
This is a descriptive sensitivity study, not independent external validation.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import statistics

import pandas as pd
from portfolio.churn import make_sample
from portfolio.tabular import train

SEEDS = (7, 21, 42, 84, 123)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Fix data generation; CSV round-trip matches the existing tabular CLI.
    csv = make_sample(7043, 42).to_csv(index=False, lineterminator='\n')
    frame = pd.read_csv(io.StringIO(csv))
    runs = []
    all_predictions = []
    for seed in SEEDS:
        _, report, predictions = train(frame, 'Churn', 'classification', seed=seed)
        runs.append(report)
        predictions.insert(0, 'seed', seed)
        all_predictions.append(predictions)
        print(f"seed={seed}: {report['selected_model']}, test AUC={report['test']['selected_model']['roc_auc']:.6f}", flush=True)
    summary = {}
    for metric in ('roc_auc', 'f1', 'accuracy'):
        summary[metric] = {}
        for model in ('selected_model', 'baseline'):
            values = [r['test'][model][metric] for r in runs]
            summary[metric][model] = {
                'mean': statistics.mean(values), 'sample_std': statistics.stdev(values),
                'min': min(values), 'max': max(values),
            }
    report = {
        'experiment': 'Fixed synthetic dataset; five predeclared split-and-estimator seeds',
        'dataset': {'kind': 'synthetic', 'rows': 7043, 'generation_seed': 42,
                    'csv_sha256': hashlib.sha256(csv.encode('utf-8')).hexdigest()},
        'seeds': list(SEEDS),
        'protocol': 'Each run selects by validation ROC-AUC, refits on train+validation, then measures its test split. No tuning or best-seed selection.',
        'selected_model_counts': dict(Counter(r['selected_model'] for r in runs)),
        'summary': summary, 'runs': runs,
        'environment': {'python': platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'joblib')}},
        'limitations': 'Overlapping holdouts from one generated dataset; variation combines split and estimator randomness. Sample standard deviation is descriptive, not a confidence interval. No real-customer or external generalization claim.',
    }
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    pd.concat(all_predictions, ignore_index=True).to_csv(output / 'predictions.csv', index=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/churn-stability'))
    args = parser.parse_args()
    run(args.output)
