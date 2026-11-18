"""Compare P10's successful and weaker distributions using ranked bar charts."""

import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/p10-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'output/analysis/player10_distribution_finding'
OUT = ROOT / 'output/pdf'
INK = '#183249'
BLUE = '#1D4ED8'
LIGHT = '#60A5FA'
GRAY = '#BFD0E4'
DISTRIBUTIONS = {
    'u_shape': 'U-shaped', 'right_skew': 'Right-skewed', 'uniform': 'Uniform',
    'normal': 'Normal', 'mixed': 'Mixed', 'p10_90': 'p10_90', 'p90_10': 'p90_10',
}


def main():
    outcomes = pd.read_csv(DATA / 'competitor_outcome_components.csv', dtype={'group': str})
    focal = outcomes[outcomes.group.eq('10')].set_index('distribution').loc[list(DISTRIBUTIONS)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                         'ytick.color': INK, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(11, 7), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=.13)
    overview = fig.add_subplot(grid[0, :])
    order = ['uniform', 'normal', 'mixed', 'right_skew', 'u_shape', 'p10_90', 'p90_10']
    for i, dist in enumerate(order):
        value = 100 * focal.loc[dist, 'positive_rate']
        rank = int(focal.loc[dist, 'positive_outcome_rank'])
        color = {'u_shape': BLUE, 'right_skew': LIGHT}.get(dist, GRAY)
        overview.bar(i, value, width=.64, color=color)
        overview.text(i, value + 2, f'{value:.2f}%\nrank {rank}/10', ha='center', va='bottom')
    overview.set_xticks(range(len(order)), [DISTRIBUTIONS[d] for d in order])
    overview.set(ylim=(0, 115), yticks=range(0, 101, 20), ylabel='P10 players with positive happiness (%)')
    overview.set_title('A   Positive outcomes and rank across distributions', loc='left', weight='bold', pad=10)
    overview.grid(axis='y', color='#E2E8F0', linewidth=.7)
    overview.set_axisbelow(True)
    overview.tick_params(length=0)
    for spine in overview.spines.values():
        spine.set_visible(False)
    left = fig.add_subplot(grid[1, 0])
    right = fig.add_subplot(grid[1, 1], sharey=left)
    axes = [left, right]
    panels = [
        ('married_rate', 'B   Marriage frequency', 120, [0, 25, 50, 75, 100]),
        ('nonpositive_married_rate', 'C   Nonpositive marriages', 7.4, [0, 2, 4, 6]),
    ]
    y = np.arange(len(focal))
    for ax, (metric, title, limit, ticks) in zip(axes, panels):
        values = 100 * focal[metric]
        ax.barh(y, values, height=.55, color=[BLUE, LIGHT] + [GRAY] * 5)
        for i, (value, rank) in enumerate(zip(values, focal[metric + '_rank'])):
            ax.text(value + limit * .013, i, f'{value:.2f}%  #{rank:d}', va='center', fontsize=9)
        ax.set(xlim=(0, limit), xticks=ticks, yticks=y, yticklabels=list(DISTRIBUTIONS.values()),
               xlabel='Share of all P10 players (%)')
        ax.set_title(title, loc='left', weight='bold', pad=10)
        ax.grid(axis='x', color='#E2E8F0', linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].invert_yaxis()
    fig.savefig(OUT / 'player10_distribution_results.pdf')
    fig.savefig(OUT / 'player10_figures/player10_distribution_results.png', dpi=200)
    plt.close(fig)
    method = {'A': 'Positive happiness / all seats, ranked highest first with mean happiness breaking ties.',
              'B': 'Married / all seats, ranked highest first.',
              'C': '(Negative + zero-happiness marriages) / all seats, ranked lowest first.',
              'scope': 'All archived rosters. Ranks among ten strategies within each distribution.'}
    (DATA / 'figure_method.json').write_text(json.dumps(method, indent=2) + chr(10))


if __name__ == '__main__':
    main()
