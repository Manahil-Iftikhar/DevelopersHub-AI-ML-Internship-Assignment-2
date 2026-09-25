"""Original synthetic churn-data generator, extracted for reproducible demonstrations.
The generated rows are not real customers and are not an external Telco benchmark.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_sample(n_samples=7043, seed=42):
    rng = np.random.RandomState(seed)


    # Generate synthetic dataset similar to Telco Churn
    data = {
        'gender': rng.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': rng.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': rng.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48]),
        'Dependents': rng.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
        'tenure': rng.exponential(20, n_samples).astype(int),
        'PhoneService': rng.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10]),
        'MultipleLines': rng.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10]),
        'InternetService': rng.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22]),
        'OnlineBackup': rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.43, 0.22]),
        'DeviceProtection': rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22]),
        'TechSupport': rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22]),
        'StreamingTV': rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22]),
        'StreamingMovies': rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.39, 0.22]),
        'Contract': rng.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': rng.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41]),
        'PaymentMethod': rng.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                                         n_samples, p=[0.34, 0.23, 0.22, 0.21]),
        'MonthlyCharges': rng.normal(65, 30, n_samples),
        'TotalCharges': rng.normal(2300, 2200, n_samples)
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Clean data
    df['tenure'] = np.clip(df['tenure'], 0, 72)
    df['MonthlyCharges'] = np.clip(df['MonthlyCharges'], 18, 120)
    df['TotalCharges'] = np.clip(df['TotalCharges'], 18, 8700)

    # Create target variable with realistic churn patterns
    churn_prob = (
        0.1 +  # base probability
        0.3 * (df['Contract'] == 'Month-to-month').astype(int) +
        0.2 * (df['tenure'] < 12).astype(int) +
        0.15 * (df['MonthlyCharges'] > 80).astype(int) +
        0.1 * (df['SeniorCitizen'] == 1).astype(int) +
        0.1 * (df['InternetService'] == 'Fiber optic').astype(int)
    )
    churn_prob = np.clip(churn_prob, 0, 0.8)
    df['Churn'] = rng.binomial(1, churn_prob, n_samples)


    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=7043)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('artifacts/synthetic-churn.csv'))
    args = parser.parse_args()
    if args.rows < 30:
        parser.error('--rows must be at least 30')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_sample(args.rows, args.seed).to_csv(args.output, index=False)
    print(f'Wrote {args.rows} synthetic rows to {args.output}')
