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

The initial September 24 maintenance did not install or execute BERT, document-assistant or ticket-tagging model weights. A later ticket-tagging run is recorded below; BERT and document-assistant execution remain pending. The Streamlit app and model notebooks were checked for syntax, with offline behavior tests for their reusable utilities. Their full runtime compatibility, output quality, latency, and resource use require model-backed runs. The housing CSV was absent and its image branch is not implemented.

Historical outputs are documented separately in the project guides.

## Hosted CI evidence

[GitHub Actions run 36143914094](https://github.com/Manahil-Iftikhar/DevelopersHub-AI-ML-Internship-Assignment-2/actions/runs/36143914094) completed successfully on **September 25, 2026**, for commit `d67c1ac9240ef68f88ed3a6a2fa75e19aef6c2e1`. The workflow used Python 3.12 and installed the pinned core requirements. Both repository validation and focused offline test steps passed.

The [workflow](../.github/workflows/checks.yml) runs `python tools/check_workspace.py` and `python -m unittest discover -s tests -v` on pushes and pull requests. It does not execute the churn training CLI, the complete model-backed notebooks, or the Streamlit application. The local experiment results above are separate evidence, not outputs of this CI run.


## Model-backed ticket evaluation · September 29, 2026

Ran the unchanged zero-shot and few-shot prompts on a declared 15-case synthetic diagnostic set using pinned FLAN-T5-small weights on CPU. Both modes scored exact match 0.20, equal to the constant-label baseline. [The case study](projects/05-ticket-tagging.md) links raw outputs, measured runtime, provenance and limitations. No prompt tuning was performed after observing the results.

All 19 offline tests passed locally, including three new scoring tests. Core CI exercises those tests without downloading a model; it does not reproduce the inference run. The interactive ticket notebook uses the pinned model revision but was not executed in this evaluation environment; the measured run used the CLI.

## Offline retrieval baseline · September 29, 2026

Executed `python -m portfolio.retrieval_eval` on six authored corpus documents and 12 prelabelled questions. Source recall and mean reciprocal rank at three were 1.00 on eight answerable cases. Only one of four unanswerable cases abstained; three received context that did not contain their answer. The [case study](projects/04-context-assistant.md#recorded-offline-retrieval-baseline) explains these limits and links all retrieved passages and environment details.

All **22 offline tests** and the repository validator passed locally, including three added tests for source-level scoring, false accepts, follow-up context, abstention, and invalid labels. This baseline does not execute embedding or generation models. Existing model-backed document-assistant execution remains pending.

## Automated retrieval-report verification

Portfolio CI now runs `python -m portfolio.retrieval_eval --check`. It recomputes the lexical baseline and verifies the corpus SHA-256, settings, question labels, ranked sources/passages, similarities and aggregate metrics against the committed report. It does not overwrite that report. Numerical comparisons allow relative tolerance 1e-9 and absolute tolerance 1e-12; historical environment metadata is retained rather than required to match the current runner.

A mismatch fails CI. To intentionally update the experiment, run the CLI without `--check`, inspect changed results and limitations, and commit the dataset/code/report changes together. This gate detects stale evidence; it does not establish retrieval quality or run embedding/generation models.
