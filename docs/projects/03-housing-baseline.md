# Housing: tabular baseline

**Assignment task 3 · Manahil Iftikhar**

Ames-style housing regression; multimodal extension pending.

[Maintained notebook](../../notebooks/03-housing-baseline.ipynb) · [Original submission](../../archive/Task%203_%20Multimodal%20ML%20%E2%80%93%20Housing%20Price%20Prediction%20Using%20Images%20%2B%20Tabular%20Data) · [Portfolio](../../README.md)

## Current state

California Housing case study measured; original CSV and paired images still absent.

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

## Separate California Housing case study · September 30, 2026

To make a tabular regression example reproducible without the missing assignment files, this **separate CLI experiment** uses California Housing through scikit-learn. It does not replace the original Ames-style notebook or claim an image branch.

### Provenance and target

The [official dataset description](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) describes 20,640 census block groups from 1990, with eight numeric predictors. The target is each area's median house value in units of $100,000, not an individual property's sale price. The loader derives room and occupancy averages from the source data.

Data downloads from [the archive used by scikit-learn](https://ndownloader.figshare.com/files/5976036). The pinned scikit-learn 1.8.0 loader checks the downloaded archive against SHA-256 `aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`. The report additionally records the canonical loaded-table hash, package versions, target range and feature columns. Raw data and model files are not committed.

### Protocol and results

The existing tabular pipeline uses seed 42 to split 12,384 training, 4,128 validation and 4,128 test rows. Imputation and scaling fit within each training pipeline. Median, ridge and random-forest candidates are compared by validation RMSE. Random forest wins, is refitted on training plus validation, and is evaluated on the held-out test partition.

| Test metric | Selected random forest | Median baseline |
| --- | ---: | ---: |
| MAE (dataset units) | 0.3264 | 0.8740 |
| RMSE (dataset units) | 0.5044 | 1.1731 |
| R² | 0.8059 | -0.0502 |
| MAE (historical USD equivalent) | $32,642 | $87,404 |
| RMSE (historical USD equivalent) | $50,438 | $117,312 |

Dollar equivalents multiply the dataset-unit errors by 100,000; they are not estimates of today's property-market errors.

[Full metrics and validation scores](../../reports/california-housing/metrics.json) · [4,128 held-out predictions](../../reports/california-housing/predictions.csv) · [CLI source](../../portfolio/california_housing.py)

### Reproduce

Install the pinned core requirements, then:

```bash
python -m portfolio.california_housing --download
```

The explicit flag allows the first public-data download. Later runs can omit it and reuse the cache in `artifacts/california-cache`. Outputs default to `artifacts/california-housing`; use `--output` and `--cache` to choose other directories. A missing cache without `--download` fails rather than downloading silently.

### Interpretation and limits

Nearby areas can appear in both sides of a random split. This experiment does not establish transfer to a new region or future market; a geographic holdout is a separate next test. The loaded target reaches a ceiling of 5.00001 in 965 rows, complicating high-value error interpretation. Block-group aggregates are not individual homes, and the historical data is not a current valuation service. No paired images, image model, multimodal comparison or deployment is included. This experiment provides measured tabular evidence only.
