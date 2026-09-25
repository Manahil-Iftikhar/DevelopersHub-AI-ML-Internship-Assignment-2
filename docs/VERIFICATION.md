# Verification record

Portfolio maintenance: 2026-09-24.

Validated locally with Python 3.12.14 and the pinned core environment in `requirements.txt`.

| Check | Observed result |
| --- | --- |
| Repository validation | Passed: five maintained notebooks, local Markdown links, Python syntax, cover SVG, and five original notebook blob hashes |
| Focused offline tests | 16 passed |
| Synthetic-churn notebook | Every code cell executed successfully in order with a shared namespace |
| Churn command-line workflow | Generated CSV, trained and selected a model, exported metrics, predictions, and a serialized pipeline |
| Original submissions | All five archived notebook contents match their original Git blob SHA |

The tests cover preprocessing, input validation, serialization, controlled ticket labels, actual few-shot prompt construction, chunk/source preservation, independent conversation histories, and abstention when retrieval returns no evidence. Retrieval and generation are stubbed in offline tests; these tests do not measure language-model quality.

## Recorded synthetic-churn run

The [machine-readable report](../reports/synthetic-churn.json) records the generated CSV SHA-256, package versions, seed, validation scores, and test scores. This is a simulation with 7,043 rows, not the external Telco dataset or real customer data.

- Split: 4,225 training rows, 1,409 validation rows, 1,409 test rows.
- Selected by validation ROC-AUC: random forest; refitted on training plus validation before one final test evaluation.
- Test: accuracy **0.6629**, churn-class F1 **0.6122**, ROC-AUC **0.7186**.
- Majority baseline: accuracy **0.5444**, churn-class F1 **0.0000**, ROC-AUC **0.5000**.

These numbers demonstrate the implemented workflow on one seeded simulation. They do not establish real-world customer performance or uncertainty across repeated splits.

## Repeat the checks and experiment

```bash
python -m pip install -r requirements.txt
python tools/check_workspace.py
python -m unittest discover -s tests -v
python -m portfolio.churn --output artifacts/synthetic-churn.csv
python -m portfolio.tabular --csv artifacts/synthetic-churn.csv --target Churn --task classification --data-kind synthetic --output artifacts/churn-cli
python tools/render_metrics.py
```

`render_metrics.py` plots the committed report. To plot a new run, first replace `reports/synthetic-churn.json` with the new metrics and record the changed environment and data origin. Serialized models and generated datasets remain outside version control.

## Execution limits

The BERT, document-assistant, and ticket-tagging model dependencies and weights were not installed or executed during this maintenance. The Streamlit app and model notebooks were checked for syntax, with offline behavior tests for their reusable utilities. Their full runtime compatibility, output quality, latency, and resource use require model-backed runs. The housing CSV was absent and its image branch is not implemented.

The GitHub Actions workflow runs the offline checks above when GitHub enables it. Local success does not by itself establish a successful hosted CI run. Historical outputs are documented separately in the project guides.
