# BERT news classification

**Assignment task 1 · Manahil Iftikhar**

Transformer training, text classification, and evaluation.

[Maintained notebook](../../notebooks/01-bert-news.ipynb) · [Original submission](../../archive/Task%201_%20News%20Topic%20Classifier%20Using%20BERT) · [Portfolio](../../README.md)

## Current state

Training code present; completed evaluation not in the archive.

## Data and model provenance

AG News via Hugging Face Datasets; bert-base-uncased. The original selected 10,000 training and 2,000 test examples.

## Evidence in the original submission

The saved training progress stops at 3/1,875 steps. No completed accuracy or F1 result is recorded, so the earlier README’s approximately 92% accuracy claim is not retained.

These statements describe source and saved outputs from the original commit `8fa5cff63746`. They are not fresh benchmark measurements.

## Improvements and remaining work

The maintained notebook reserves validation data from the training pool, keeps test data out of checkpoint selection, handles inference device placement, and makes the demo launch opt-in.

## Reproduce

Follow the [VS Code setup](../SETUP.md), open the maintained notebook, and select the installed environment. Supply the data described above where required. The [verification record](../VERIFICATION.md) distinguishes executed checks from workflows requiring external assets.

## Questions to answer in the next experiment

- What baseline is the method compared against?
- Does the evaluation split represent the intended use?
- Which errors remain, and what evidence explains them?
- Can another person reproduce the result from the documented data and settings?
