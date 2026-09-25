# Document assistant

**Assignment task 4 · Manahil Iftikhar**

Retrieval, source inspection, and conversation state.

[Maintained notebook](../../notebooks/04-context-assistant.ipynb) · [Original submission](../../archive/Task%204_%20Context-Aware%20Chatbot%20Using%20LangChain%20or%20RAG) · [Portfolio](../../README.md)

## Current state

Local app included; model-backed execution requires downloads.

## Data and model provenance

A short, repository-supplied sample corpus; all-MiniLM-L6-v2 embeddings; FAISS retrieval; google/flan-t5-small generation. The maintained splitter is plain Python, so legacy LangChain import paths are unnecessary.

## Evidence in the original submission

The original contains one displayed answer, not a retrieval-quality benchmark. The new app exposes retrieved passages and keeps each session’s history separate. Its similarity cutoff is an uncalibrated heuristic.

These statements describe source and saved outputs from the original commit `8fa5cff63746`. They are not fresh benchmark measurements.

## Improvements and remaining work

Evaluate relevant and unrelated questions, short follow-ups, and incorrect generated answers. Source display makes inspection possible but does not guarantee grounded generation.

## Reproduce

Follow the [VS Code setup](../SETUP.md), open the maintained notebook, and select the installed environment. Supply the data described above where required. The [verification record](../VERIFICATION.md) distinguishes executed checks from workflows requiring external assets.

## Questions to answer in the next experiment

- What baseline is the method compared against?
- Does the evaluation split represent the intended use?
- Which errors remain, and what evidence explains them?
- Can another person reproduce the result from the documented data and settings?
