# Run locally in VS Code

Use Python 3.11 or 3.12. Start in the repository root after cloning `DevelopersHub-AI-ML-Internship-Assignment-2`.

## Create and select an environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements-notebook.txt
```

Install the Python and Jupyter VS Code extensions. Open a file in `notebooks/`, choose **Select Kernel**, and select `.venv`. Run cells in order. Model training can require substantial memory and time; the README of each task documents the evidence and remaining requirements.

## Lightweight checks

```bash
python tools/check_workspace.py
python -m unittest discover -s tests -v
```

These checks exercise repository structure and small fixtures. They do not download datasets, validate model-generated health advice, train BERT, or establish real-world model quality.

## Reusable tabular runner

```bash
python -m portfolio.tabular --help
```

The runner requires a CSV, target column, task type, and explicit `--data-kind external` or `--data-kind synthetic`. It writes `metrics.json`, `predictions.csv`, and `model.joblib` under the chosen output directory. The JSON records the data hash, package versions, split sizes, baseline, validation selection, and final test metrics.

Only load joblib/pickle artifacts you trust. Keep downloaded datasets, local models, and credentials out of Git commits.

## Additional environments

```bash
python -m pip install -r requirements-llm.txt
```

These pins follow versions recorded in the original model notebooks where available. The complete model environment was not installed or executed during portfolio maintenance. The first model run needs network access and disk space for public checkpoints. Use a separate environment if your hardware requires a different PyTorch build.

## Data and reproduction limits

External dataset files used in the original submissions were not committed. Supply the original files and record their exact source/version before reporting new benchmark results. The original notebooks remain in `archive/`; the maintained notebooks have cleared outputs so stale results cannot be mistaken for a fresh run.

## Offline synthetic churn example

```bash
python -m portfolio.churn --output artifacts/synthetic-churn.csv
python -m portfolio.tabular --csv artifacts/synthetic-churn.csv --target Churn --task classification --data-kind synthetic --output artifacts/churn
```

These results concern generated data only.

## BERT

Install `requirements-news.txt`, then run the BERT notebook in order. Its default training subset is 10,000 examples, split into 8,000 training and 2,000 validation rows; a separate 2,000-example test subset is used after training. The archive does not contain completed weights or verified test metrics.

## Document assistant

After installing `requirements-llm.txt`:

```bash
python -m streamlit run app.py
```

The app starts locally. Press **Start assistant** to download/load models. A hosted deployment is a separate step.
