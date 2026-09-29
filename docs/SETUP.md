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

## Offline retrieval explorer

After creating and activating the environment, install the lightweight demo dependencies:

```bash
python -m pip install -r requirements-demo.txt
python -m streamlit run offline_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open the local URL printed in the terminal. Search the six included portfolio documents, inspect matched passages, and optionally enter a previous question to try a follow-up. After installing dependencies, retrieval needs no network access, API key, embedding checkpoint or language-model weights.

This mode performs TF-IDF retrieval only. It does not write answers, and similarity is not confidence that a passage answers the question. The measured baseline retrieved context for three of four unanswerable questions. No personal document upload or external service is involved.

Run its in-process interface checks with `python tools/check_offline_app.py`. These checks block outbound socket connections and cover relevant/no-match/blank questions, follow-ups and fresh-session state. They do not establish browser rendering or deployment readiness.

### Launch from VS Code

1. Open the repository folder in VS Code and install the recommended Python and Python Debugger extensions.
2. Use **Python: Select Interpreter** in the Command Palette to select the project's `.venv`. Install `requirements-demo.txt` in that environment using the command above.
3. Open **Run and Debug**, select **Offline retrieval explorer**, and press **F5**. The committed [launch configuration](../.vscode/launch.json) runs Streamlit with your selected interpreter.
4. Open the local URL printed in the integrated terminal. Set a breakpoint in `offline_app.py` or `portfolio/retrieval_eval.py`, then submit a question to inspect retrieval.
5. Use the debugger's stop button when finished.

The launcher binds to `127.0.0.1`, disables usage telemetry, and disables automatic reruns on file save to keep debugging predictable. It does not install dependencies automatically. If Streamlit cannot be imported, check the selected interpreter and install the demo requirements there. If the default port is occupied, stop the previous app process before restarting.

The configuration follows the [official Python debugging guide](https://code.visualstudio.com/docs/python/debugging). Its JSON and repository paths were checked; an interactive VS Code debugging session was not exercised in the maintenance environment.

## Model-backed document assistant

After installing `requirements-llm.txt`:

```bash
python -m streamlit run app.py
```

The app starts locally. Press **Start assistant** to download/load models. A hosted deployment is a separate step.
