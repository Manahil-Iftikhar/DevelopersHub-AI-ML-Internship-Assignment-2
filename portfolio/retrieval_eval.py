"""Offline lexical retrieval baseline; does not execute embeddings or generation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from portfolio.rag import chunk_documents


def score_cases(cases, predictions):
    if len(cases) != len(predictions):
        raise ValueError('Each case needs one ranked prediction list.')
    answerable, unknown = [], []
    for case, retrieved in zip(cases, predictions):
        relevant = set(case['relevant_sources'])
        # Multiple chunks from one source do not inflate source-level recall.
        ranked = list(dict.fromkeys(item['source'] for item in retrieved))
        if relevant:
            rank = next((i for i, source in enumerate(ranked, 1) if source in relevant), None)
            answerable.append((len(relevant.intersection(ranked)) / len(relevant),
                               1 / rank if rank else 0.0))
        else:
            unknown.append(not retrieved)
    return {
        'answerable_cases': len(answerable), 'unanswerable_cases': len(unknown),
        'source_recall_at_k': sum(x[0] for x in answerable) / len(answerable) if answerable else None,
        'source_mrr_at_k': sum(x[1] for x in answerable) / len(answerable) if answerable else None,
        'unanswerable_abstention_rate': sum(unknown) / len(unknown) if unknown else None,
        'unanswerable_false_accepts': sum(not value for value in unknown),
    }


def evaluate(dataset, k=3, threshold=0.15):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if k < 1 or not 0 <= threshold <= 1:
        raise ValueError('Require k >= 1 and threshold between 0 and 1.')
    chunks = chunk_documents(dataset['documents'])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, norm='l2')
    matrix = vectorizer.fit_transform([chunk['text'] for chunk in chunks])
    outputs = []
    for case in dataset['cases']:
        if not case['question'].strip():
            raise ValueError('Questions must not be empty.')
        if not set(case['relevant_sources']).issubset(dataset['documents']):
            raise ValueError('Unknown relevance source.')
        # Same previous-question concatenation used by Conversation.ask.
        query = ' '.join(filter(None, [case.get('previous_question'), case['question']]))
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        indices = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))[:k]
        results = [{**chunks[i], 'chunk_index': i, 'similarity': float(scores[i])}
                   for i in indices if scores[i] >= threshold]
        outputs.append({**case, 'retrieval_query': query, 'retrieved': results})
    return {
        'method': 'TF-IDF unigram/bigram cosine; corpus-only vocabulary; stable chunk-order ties',
        'k': k, 'minimum_similarity': threshold,
        'threshold_selection': '0.15 declared before first run; not tuned or calibrated',
        'chunking': {'max_chars': 500, 'overlap': 80, 'chunks': len(chunks)},
        'metrics': score_cases(dataset['cases'], [row['retrieved'] for row in outputs]),
        'outputs': outputs,
        'limits': 'Small authored diagnostic set; no independent test split. Retrieval only, no answer generation. Scores are lexical similarities, not confidence.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('evaluations/retrieval.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/retrieval-baseline.json'))
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    report = evaluate(json.loads(raw))
    report['dataset_sha256'] = hashlib.sha256(raw).hexdigest()
    report['environment'] = {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
