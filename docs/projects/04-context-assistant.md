# Document assistant

**Assignment task 4 · Manahil Iftikhar**

Retrieval, source inspection, and conversation state.

[Maintained notebook](../../notebooks/04-context-assistant.ipynb) · [Original submission](../../archive/Task%204_%20Context-Aware%20Chatbot%20Using%20LangChain%20or%20RAG) · [Portfolio](../../README.md)

## Current state

Offline retrieval baseline recorded; embedding and generation evaluation pending.

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

## Recorded offline retrieval baseline

On September 29, 2026, a TF-IDF cosine baseline ran on [six authored portfolio documents and 12 labelled questions](../../evaluations/retrieval.json). Labels and the 0.15 similarity cutoff were fixed before this run; no threshold tuning followed. This diagnostic corpus is separate from the app's short sample text.

| Measure | Result |
| --- | --- |
| Answerable questions | 8, including 2 follow-ups with an explicit previous question |
| Source recall at 3 | 1.00 |
| Source reciprocal rank at 3, averaged | 1.00 |
| Unanswerable questions with no retrieved context | 1/4 (25%) |
| Unanswerable questions receiving context | 3/4 (75%) |

[Full report](../../reports/retrieval-baseline.json) includes each question, its reference source labels, the actual retrieval query, passages, scores, corpus hash and environment versions. Retrieval uses the existing 500-character splitter with 80-character overlap, a corpus-fitted unigram/bigram TF-IDF vocabulary, normalized cosine similarity, and at most three chunks. Source-level metrics deduplicate retrieved chunk sources; repeated chunks cannot inflate recall.

The unanswered topics include a telephone number, an API subscription price, a deployed assistant URL, and BERT accuracy. Topic overlap caused the last three to retrieve context despite the corpus lacking their answers. This illustrates why returning a related source is insufficient evidence that a question can be answered.

### Reproduce without model downloads

From the repository root, install the pinned core requirements, then run:

```bash
python -m pip install -r requirements.txt
python -m portfolio.retrieval_eval
python -m unittest discover -s tests -v
```

The CLI writes `reports/retrieval-baseline.json`; use `--output artifacts/retrieval-baseline.json` to keep the committed report unchanged.

### Interpretation and next comparison

This is a small, project-authored diagnostic, with obvious vocabulary overlap and no independent train/validation/test split. Perfect ranking here does not establish general retrieval quality. The two follow-ups concatenate an explicitly supplied previous question; they do not test a full multi-turn conversation.

The run executes **TF-IDF only**. It does not execute MiniLM, FAISS, FLAN-T5, the Streamlit UI, or assess answer correctness and grounding. The lexical cutoff of 0.15 is not interchangeable with the app's embedding cutoff of 0.35. No app model or default threshold was changed. A later comparison should freeze an independent question set, run the embedding retriever on the same corpus, and calibrate abstention on separate validation examples.

## Try the offline explorer

[Launch instructions](../SETUP.md#offline-retrieval-explorer) describe a separate Streamlit interface in `offline_app.py`. It reuses the measured lexical baseline and included six-document corpus. A question and optional prior question produce source passages and similarity scores; no answer-generation model is loaded. This makes the known retrieval and abstention limitations directly inspectable.

The interface has automated Streamlit AppTest checks for ordinary/no-match/blank input, follow-up context and fresh-session state, with outbound socket connections blocked. CI runs these separately from the 25 core unit tests. These are in-process interface checks, not browser screenshots or end-to-end model evaluation. The original model-backed `app.py` remains a separate workflow.
