"""Separate public-data housing case study; not the original Ames assignment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.datasets import fetch_california_housing
from portfolio.tabular import train

FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
SOURCE_URL = 'https://ndownloader.figshare.com/files/5976036'
ARCHIVE_SHA256 = 'aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681'


def validate_frame(frame):
    if list(frame.columns) != FEATURES + ['MedHouseVal']:
        raise ValueError('Unexpected California Housing columns or target.')
    if len(frame) != 20640 or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Expected 20640 finite numeric records.')
    if (frame['MedHouseVal'] <= 0).any():
        raise ValueError('Median house values must be positive.')
    return frame


def run(output, cache, download=False):
    data = fetch_california_housing(data_home=str(cache), as_frame=True,
                                    download_if_missing=download, n_retries=1)
    frame = validate_frame(data.frame)
    csv = frame.to_csv(index=False, lineterminator='\n')
    _, report, predictions = train(frame, 'MedHouseVal', 'regression', seed=42)
    report['dataset'] = {
        'name': 'California Housing', 'kind': 'external', 'source_url': SOURCE_URL,
        'loader': 'sklearn.datasets.fetch_california_housing',
        'source_archive_sha256': ARCHIVE_SHA256,
        'canonical_csv_sha256': hashlib.sha256(csv.encode()).hexdigest(),
        'target_units': '100000 USD; census block-group median house value',
        'target_max': float(frame.MedHouseVal.max()),
        'rows_at_target_max': int((frame.MedHouseVal == frame.MedHouseVal.max()).sum()),
    }
    report['environment'] = {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'joblib')}}
    report['limitations'] = ('Historical 1990 census block groups, not individual listings or current prices. '
        'Random holdouts may share nearby locations and do not test geographic or temporal generalization. '
        'Target ceiling affects high-value error interpretation. No paired images or multimodal model.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    predictions.to_csv(output / 'predictions.csv', index=False)
    print(json.dumps({'selected_model': report['selected_model'], 'rows': report['rows'],
                      'test': report['test'], 'dataset': report['dataset']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Allow public dataset download on first run')
    parser.add_argument('--cache', type=Path, default=Path('artifacts/california-cache'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/california-housing'))
    args = parser.parse_args()
    run(args.output, args.cache, args.download)
