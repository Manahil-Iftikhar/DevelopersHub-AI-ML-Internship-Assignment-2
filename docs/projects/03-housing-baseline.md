# Housing: tabular baseline

**Assignment task 3 · Manahil Iftikhar**

Ames-style housing regression; multimodal extension pending.

[Maintained notebook](../../notebooks/03-housing-baseline.ipynb) · [Original submission](../../archive/Task%203_%20Multimodal%20ML%20%E2%80%93%20Housing%20Price%20Prediction%20Using%20Images%20%2B%20Tabular%20Data) · [Portfolio](../../README.md)

## Current state

CSV required; paired image branch not implemented.

## Data and model provenance

Original train.csv and test.csv include SalePrice and Ames-style housing columns. Dataset files and paired property images are absent from the repository.

## Evidence in the original submission

The original source trains Linear Regression on tabular features and records validation RMSE 65,393.11. Its five generated house drawings are illustrations; they are never input to the model. No CNN, fusion model, or image-ablation result is present.

These statements describe source and saved outputs from the original commit `8fa5cff63746`. They are not fresh benchmark measurements.

## Improvements and remaining work

The maintained notebook implements an honest tabular baseline. A true multimodal extension needs property-ID-matched images and a comparison of tabular-only, image-only, and fused models using the same split.

## Reproduce

Follow the [VS Code setup](../SETUP.md), open the maintained notebook, and select the installed environment. Supply the data described above where required. The [verification record](../VERIFICATION.md) distinguishes executed checks from workflows requiring external assets.

## Questions to answer in the next experiment

- What baseline is the method compared against?
- Does the evaluation split represent the intended use?
- Which errors remain, and what evidence explains them?
- Can another person reproduce the result from the documented data and settings?
