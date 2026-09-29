# Customer churn pipeline

**Assignment task 2 · Manahil Iftikhar**

Reusable preprocessing, model selection, and serialization.

[Maintained notebook](../../notebooks/02-churn-pipeline.ipynb) · [Original submission](../../archive/Task%202_%20End-to-End%20ML%20Pipeline%20with%20Scikit-learn%20Pipeline%20API) · [Portfolio](../../README.md)

## Current state

Runs offline on explicitly synthetic data.

## Data and model provenance

7,043 generated rows using the original seed-42 simulator. This is synthetic Telco-like data, not the downloaded Telco Customer Churn dataset.

## Evidence in the original submission

Saved original outputs: Logistic Regression accuracy 0.6749 / ROC-AUC 0.7141; Random Forest accuracy 0.6551 / ROC-AUC 0.7342. The original README’s 87% accuracy claim is not supported by those outputs.

These statements describe source and saved outputs from the original commit `8fa5cff63746`. They are not fresh benchmark measurements.

## Improvements and remaining work

The maintained runner selects on a separate validation set and evaluates the final choice on a held-out test set. Synthetic results demonstrate workflow behavior, not real-world customer performance.

## Measured portfolio run · 2026-09-24

The revised seeded simulation selected a random forest by validation ROC-AUC. On 1,409 held-out test rows it achieved accuracy **0.6629**, churn-class F1 **0.6122**, and ROC-AUC **0.7186**. The majority baseline achieved accuracy **0.5444** and ROC-AUC **0.5000**. The [full report](../../reports/synthetic-churn.json) includes environment versions, dataset hash, and validation results.

The confusion matrix contains 267 missed churn cases and 208 false alarms. That trade-off is a reason to study threshold selection and the cost of each error on validation data before any operational use. All examples here are synthetic.

## Reproduce

Follow the [VS Code setup](../SETUP.md), open the maintained notebook, and select the installed environment. Supply the data described above where required. The [verification record](../VERIFICATION.md) distinguishes executed checks from workflows requiring external assets.

## Questions to answer in the next experiment

- What baseline is the method compared against?
- Does the evaluation split represent the intended use?
- Which errors remain, and what evidence explains them?
- Can another person reproduce the result from the documented data and settings?

## Five-seed sensitivity study · 2026-09-29

A second experiment holds the generated dataset fixed (7,043 rows; generator seed 42) and repeats the unchanged workflow with seeds **7, 21, 42, 84, 123**, chosen before execution. Each seed controls both holdout assignment and stochastic estimator initialization. This measures their combined sensitivity, not split variation alone.

Every run uses 4,225 training, 1,409 validation and 1,409 test rows. The same candidate pipelines compete on validation ROC-AUC; the winner is refitted on training plus validation before evaluating that run's test split. No best seed was selected and no hyperparameters were changed after inspecting the results.

| Seed | Validation-selected model | Test ROC-AUC | Test F1 | Test accuracy |
| --- | --- | ---: | ---: | ---: |
| 7 | logistic regression | 0.7366 | 0.6538 | 0.6820 |
| 21 | logistic regression | 0.7309 | 0.6278 | 0.6600 |
| 42 | random forest | 0.7186 | 0.6122 | 0.6629 |
| 84 | logistic regression | 0.7407 | 0.6481 | 0.6778 |
| 123 | logistic regression | 0.7492 | 0.6535 | 0.6891 |

Mean test ROC-AUC was **0.7352**, with sample standard deviation **0.0114** and range **0.7186–0.7492**. The prior baseline scored **0.5000** in every run. Logistic regression won validation selection four times and random forest once, showing that a single selected model is not a stable conclusion. The seed-42 result reproduces the earlier report rather than replacing it with the best score.

[Full metrics and validation results](../../reports/churn-stability/metrics.json) · [All test predictions](../../reports/churn-stability/predictions.csv) · [Runner](../../portfolio/churn_stability.py)

### Reproduce the study

Install the core requirements, then run from the repository root:

```bash
python -m portfolio.churn_stability --output artifacts/churn-stability
```

The runner regenerates the fixed data, records its CSV hash and environment, and exports metrics plus predictions identified by seed and original row index. It reuses the maintained tabular pipeline; no separate model implementation is introduced.

### Limits

The five test sets overlap and come from one synthetic dataset. These results are correlated; the reported sample standard deviation is descriptive, **not a confidence interval**. They do not measure robustness to a new population, external customers, altered generation assumptions, or future time periods. Future tuning needs fresh held-out evidence; these five results should not become a test set repeatedly optimized against. CI checks source and existing tests, but does not rerun this training study.
