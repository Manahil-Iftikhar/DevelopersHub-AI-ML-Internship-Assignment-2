"""Rebuild the README figure from the recorded synthetic-churn report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'artifacts').mkdir(exist_ok=True)
report = json.loads((ROOT / 'reports/synthetic-churn.json').read_text())
keys = ['accuracy', 'f1', 'roc_auc']
labels = ['Accuracy', 'F1 · churn class', 'ROC-AUC']
x = np.arange(len(keys))
fig, ax = plt.subplots(figsize=(9, 4.8), facecolor='white')
for offset, model, label, color in [
    (-0.18, 'baseline', 'Majority baseline', '#a5b2bd'),
    (0.18, 'selected_model', 'Selected random forest', '#6454c0'),
]:
    bars = ax.bar(x + offset, [report['test'][model][key] for key in keys],
                  width=0.34, label=label, color=color)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=10)
ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel='Score')
ax.set_title('Synthetic churn: held-out test results', loc='left', pad=18)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.15)
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=False)
fig.text(0.1, 0.02, '1,409 test rows · seed 42 · synthetic data; not a real-customer benchmark',
         fontsize=9, color='#4a5d69')
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(ROOT / 'assets/churn-results.svg', bbox_inches='tight')
fig.savefig(ROOT / 'artifacts/churn-preview.png', dpi=150, bbox_inches='tight')
plt.close(fig)
