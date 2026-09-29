"""Evaluate existing ticket prompts on a declared synthetic diagnostic set."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

from portfolio.tickets import TAGS, build_prompt, tag_ticket

MODEL_ID = 'google/flan-t5-small'
REVISION = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'


def score(rows):
    if not rows:
        raise ValueError('At least one prediction is required.')
    tp = fp = fn = exact = reviewed = 0
    for row in rows:
        gold, predicted = set(row['expected']), set(row['tags'])
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        # Invalid extra text must not count as an exact correct answer.
        exact += gold == predicted and not row['needs_review']
        reviewed += bool(row['needs_review'])
    return {'cases': len(rows), 'exact_match': exact / len(rows),
            'micro_precision': tp / (tp + fp) if tp + fp else 0.0,
            'micro_recall': tp / (tp + fn) if tp + fn else 0.0,
            'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'review_rate': reviewed / len(rows)}


def load_cases(path):
    dataset = json.loads(Path(path).read_text())
    cases = dataset['cases']
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Cases must be nonempty with unique IDs.')
    for case in cases:
        if not case['text'].strip() or not case['tags'] or not set(case['tags']) <= set(TAGS):
            raise ValueError('Each case needs text and known nonempty labels.')
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('evaluations/tickets.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/ticket-evaluation'))
    parser.add_argument('--download', action='store_true', help='Allow pinned model download')
    args = parser.parse_args()
    dataset = load_cases(args.data)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                             local_files_only=not args.download)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, revision=REVISION,
                                                 local_files_only=not args.download,
                                                 use_safetensors=True).to('cpu')
    load_seconds = time.perf_counter() - started
    # One fixed warm-up, excluded from per-ticket timings and never scored.
    tag_ticket('A sample ticket for warm-up.', tokenizer, model)
    outputs, metrics = {}, {}
    for mode in ('zero_shot', 'few_shot'):
        rows = []
        for case in dataset['cases']:
            prompt = build_prompt(case['text'], few_shot=mode == 'few_shot')
            tokens = len(tokenizer(prompt)['input_ids'])
            if tokens > 512:
                raise ValueError('Evaluation prompt would be truncated.')
            started = time.perf_counter()
            result = tag_ticket(case['text'], tokenizer, model, few_shot=mode == 'few_shot')
            rows.append({'id': case['id'], 'text': case['text'], 'expected': case['tags'],
                         'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                         'input_tokens': tokens, 'seconds': time.perf_counter() - started,
                         **result})
        outputs[mode] = rows
        metrics[mode] = score(rows)
        metrics[mode]['median_seconds_per_ticket'] = statistics.median(r['seconds'] for r in rows)
    baseline = [{'expected': c['tags'], 'tags': ['Technical Issue'], 'needs_review': False}
                for c in dataset['cases']]
    metrics['constant_technical_issue_baseline'] = score(baseline)
    report = {
        'model': {'id': MODEL_ID, 'revision': REVISION, 'license': 'Apache-2.0',
                  'parameter_count': sum(p.numel() for p in model.parameters()),
                  'device': 'cpu', 'dtype': str(next(model.parameters()).dtype),
                  'threads': torch.get_num_threads(), 'load_seconds_including_download_if_needed': load_seconds},
        'dataset': {'path': str(args.data), 'sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    'description': dataset['description'], 'annotation_policy': dataset['annotation_policy']},
        'generation': {'max_new_tokens': 48, 'do_sample': False, 'seed': 42,
                       'prompt_modes': ['zero_shot', 'few_shot'], 'max_input_tokens': 512},
        'environment': {name: importlib.metadata.version(name) for name in
                        ['torch', 'transformers', 'tokenizers', 'sentencepiece', 'safetensors', 'huggingface-hub', 'numpy']},
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                    'available_cpu_count': os.cpu_count()},
        'metrics': metrics,
        'limitations': 'Synthetic English diagnostic set with project-authored labels; no independent annotation, real tickets, training or prompt tuning. Existing few-shot examples use overlapping labels, which differ from the specific-category evaluation policy. This protocol mismatch is preserved and must be considered when interpreting mode differences. Timing is one sequential CPU run after warm-up, not a deployment benchmark.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'outputs.json').write_text(json.dumps(outputs, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
