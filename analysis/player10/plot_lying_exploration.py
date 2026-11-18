"""Summarize paired markets with equal game weights and seed-cluster intervals."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from late_round_inset import add_late_round_inset
from lying_exploration_ablation import OUT, POLICIES, base

PDF = base.ROOT / 'output/pdf'
records = sorted((json.loads(row) for row in (OUT / 'games.jsonl').read_text().splitlines()),
                 key=lambda row: (row['seed'], row['scenario']['scenario_id']))
grid = np.arange(101)
seeds = np.array([row['seed'] for row in records])
draws = np.random.default_rng(27101).choice(np.unique(seeds), size=(10000, len(np.unique(seeds))))
metrics, curves = {}, {}
for policy in POLICIES:
    rows, positive_paths, marriage_paths = [], [], []
    for r in records:
        p = r['worlds'][policy]
        scores = np.array([s['happiness'] for s in p])
        when = np.array([s['married_on'] or np.inf for s in p]) / r['scenario']['rounds']
        rows.append([np.mean([s['proposals_received'] > 0 for s in p]), np.isfinite(when).mean(),
                     (scores > 0).mean(), ((scores <= 0) & np.isfinite(when)).mean(), scores.mean()])
        married = when[:, None] <= grid / 100
        marriage_paths.append(married.mean(axis=0))
        positive_paths.append(((scores[:, None] > 0) & married).mean(axis=0))
    metrics[policy] = np.array(rows)
    curves[policy] = {'positive': 100 * np.mean(positive_paths, axis=0),
                      'married': 100 * np.mean(marriage_paths, axis=0)}


def diff(p, ref):
    delta = metrics[p] - metrics[ref]
    seed_means = np.array([delta[seeds == seed].mean(axis=0) for seed in np.unique(seeds)])
    bootstrap = seed_means[np.searchsorted(np.unique(seeds), draws)].mean(axis=1)
    return np.vstack([delta.mean(axis=0), np.quantile(bootstrap, [.025, .975], axis=0)])


names = ['received_proposal', 'married', 'positive', 'nonpositive_marriage', 'happiness']
summary = {}
for p in POLICIES:
    ref = 'honest' if p in ['final', 'plus5', 'honest'] else 'final'
    summary[p] = {'mean': dict(zip(names, metrics[p].mean(axis=0))), 'reference': ref,
                  'paired_mean_low_high': dict(zip(names, diff(p, ref).T.tolist()))}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
gray = '#697586'


def panel(ax, title):
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='x', color='#e0e4e8', lw=.7)
    ax.set_axisbelow(True)


fig, ax = plt.subplots(figsize=(7, 3.2), layout='constrained')
labels = ['Received a proposal', 'Married', 'Positive marriage', 'Non-positive marriage']
y = np.arange(4)[::-1]
for p, offset, color, label in [('final', .18, '#1D4ED8', '+2 (final)'), ('plus5', -.18, '#60A5FA', '+5')]:
    values = 100 * (metrics[p] - metrics['honest']).mean(axis=0)[:4]
    bars = ax.barh(y + offset, values, height=.31, color=color, label=label)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
ax.set_xlim(0, 6)
ax.set_yticks(y, labels)
ax.set_ylim(-.5, 3.5)
ax.axvline(0, color=gray, lw=1)
panel(ax, 'Effect of exaggerating monotonic traits')
ax.set_xlabel('Difference from truthful reporting (percentage points)')
ax.legend(loc='lower right', frameon=False)
fig.savefig(PDF / 'player10_lying_ablation.pdf')
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.5), layout='constrained', sharey=True,
                         sharex='col')
sweep = [('final', '0%', '#1D4ED8', '-'), ('open10', '10%', '#334155', '--'),
         ('open25', '25%', '#3983BD', '-.'), ('open50', '50%', '#79ACD9', ':')]
outcomes = [('positive', 'Positive outcome rate (%)', 'A   Positive outcomes over time', 'B   Final positive outcome rate'),
            ('married', 'Marriage rate (%)', 'C   Marriages over time', 'D   Final marriage rate')]
for (line_ax, bar_ax), (metric, ylabel, line_title, bar_title) in zip(axes, outcomes):
    for policy, label, color, style in sweep:
        curve = curves[policy][metric]
        rate = 100 * summary[policy]['mean'][metric]
        assert np.isclose(rate, curve[-1])
        line_ax.plot(grid, curve, label=label, color=color, linestyle=style, lw=2)
        bar = bar_ax.bar(label, rate, width=.68, color=color)
        bar_ax.bar_label(bar, fmt='%.2f', padding=3, fontsize=9)
    add_late_round_inset(line_ax)
    line_ax.set(ylabel=ylabel, xlim=(0, 100), ylim=(0, 82), xticks=[0, 25, 50, 75, 100])
    for ax, title in [(line_ax, line_title), (bar_ax, bar_title)]:
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E2E8F0', lw=.65)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False,
           title='Exploration period', fontsize=9, title_fontsize=9)
axes[1, 0].set_xlabel('Scheduled rounds elapsed (%)')
axes[1, 1].set_xlabel('Exploration period (% of rounds)')
fig.savefig(PDF / 'player10_exploration_ablation.pdf')
plt.close(fig)
