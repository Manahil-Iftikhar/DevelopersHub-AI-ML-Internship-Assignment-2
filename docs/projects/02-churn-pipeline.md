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
