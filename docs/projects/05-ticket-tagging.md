# Support ticket tagging

**Assignment task 5 · Manahil Iftikhar**

Zero-shot/few-shot prompts and controlled labels.

[Maintained notebook](../../notebooks/05-ticket-tagging.ipynb) · [Original submission](../../archive/Task%205_%20Auto%20Tagging%20Support%20Tickets%20Using%20LLM) · [Portfolio](../../README.md)

## Current state

Prompt/parser checks pass offline; model quality not established.

## Data and model provenance

Three handcrafted support-ticket examples and five candidate labels; google/flan-t5-small. These examples are a demonstration, not a benchmark dataset.

## Evidence in the original submission

The saved original output labels all three examples Login Problem. The original few-shot prompt is defined but never used. No calibrated probabilities or verified top-three ranking are present.

These statements describe source and saved outputs from the original commit `8fa5cff63746`. They are not fresh benchmark measurements.

## Improvements and remaining work

The maintained code actually applies the selected prompting mode, accepts only known labels, removes duplicates, and flags invalid output for human review. Compare prompting modes on a separate labelled dataset before claiming accuracy.

## Reproduce

Follow the [VS Code setup](../SETUP.md), open the maintained notebook, and select the installed environment. Supply the data described above where required. The [verification record](../VERIFICATION.md) distinguishes executed checks from workflows requiring external assets.

## Questions to answer in the next experiment

- What baseline is the method compared against?
- Does the evaluation split represent the intended use?
- Which errors remain, and what evidence explains them?
- Can another person reproduce the result from the documented data and settings?
