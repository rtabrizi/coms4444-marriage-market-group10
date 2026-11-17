"""Five-panel distribution figure from saved, checked analysis results."""

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
GRAY = '#CBD5E1'


def main():
    f = json.loads((DATA / 'findings.json').read_text())
    all_results = pd.read_csv(OUT / 'player10_tables/full_tournament_distribution.csv', dtype={'group': str})
    ranked = all_results.sort_values(['distribution', 'positive_rate', 'mean_happiness'], ascending=[True, False, False]).copy()
    ranked['rank'] = ranked.groupby('distribution').cumcount() + 1
    p10 = ranked[ranked.group.eq('10')].set_index('distribution')
    order = ['uniform', 'normal', 'mixed', 'right_skew', 'u_shape', 'p10_90', 'p90_10']
    names = ['Uniform', 'Normal', 'Mixed', 'Right-skewed', 'U-shaped', 'p10_90', 'p90_10']
    assert p10.loc[order, 'rank'].tolist() == [7, 8, 7, 2, 1, 2, 4]
    focus = ['u_shape', 'right_skew', 'uniform', 'normal']
    focus_names = ['U-shaped', 'Right-skewed', 'Uniform', 'Normal']
    theory = pd.DataFrame(f['exact_random_pair_baseline'])
    theory = theory[theory.dims.eq(5) & theory.bonus.eq(100)].set_index('distribution').reindex(focus)
    screening = pd.DataFrame(f['initial_screening']).T.reindex(focus)
    dims = pd.DataFrame(f['p10_by_dims']).set_index(['distribution', 'dims'])
    shared = pd.DataFrame(f['common_rosters_all_players'])
    shared = shared[shared.group.eq('10')].set_index('distribution')
    assert shared.loc[['u_shape', 'right_skew'], 'rank_with_mean_tiebreak'].tolist() == [3, 3]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.spines.bottom': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(12.6, 7.5))
    gs = fig.add_gridspec(2, 3, left=.07, right=.985, top=.93, bottom=.105, wspace=.40, hspace=.57)
    axes = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

    def style(ax, title, ylabel, ymax=115):
        ax.set_title(title, loc='left', fontsize=11.5, weight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(length=0)
        ax.grid(axis='y', color='#E2E8F0', linewidth=.65)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ymax)

    ax = axes[0]
    style(ax, 'A  Positive outcomes and rank across distributions', 'P10 players with positive happiness (%)')
    rates = 100 * p10.loc[order, 'positive_rate']
    ax.bar(np.arange(7), rates, width=.66, color=[BLUE if d == 'u_shape' else LIGHT if d == 'right_skew' else '#BFD7F4' for d in order])
    ax.set_xticks(np.arange(7), names, fontsize=10)
    for i, (dist, value) in enumerate(zip(order, rates)):
        ax.text(i, value + 2.2, f'{value:.2f}%\nrank {p10.loc[dist, "rank"]}/10',
                ha='center', va='bottom', fontsize=10, linespacing=1.3)

    ax = axes[1]
    style(ax, 'B  Compatibility before selection', 'Positive payoff in a random pair (%)')
    values = 100 * theory.random_pair_positive.to_numpy()
    ax.bar(np.arange(4), values, width=.65, color=[BLUE, LIGHT, GRAY, GRAY])
    ax.set_xticks(np.arange(4), ['U-shaped', 'Right-\nskewed', 'Uniform', 'Normal'], fontsize=9)
    for i, value in enumerate(values):
        ax.text(i, value + 2, f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    ax.text(.5, 1.015, '5 traits; 2 monotonic; bonus 100', transform=ax.transAxes,
            ha='center', fontsize=8.5, color='#526475')

    ax = axes[2]
    style(ax, 'C  Ad screening improves the shortlist', 'True positive-payoff candidates (%)', 118)
    x = np.arange(4)
    ax.bar(x - .18, 100 * screening.pool_positive, width=.34, color=GRAY, label='Initial pool')
    ax.bar(x + .18, 100 * screening.selected_positive, width=.34, color=BLUE, label='P10 shortlist')
    for i, name in enumerate(focus):
        for shift, metric in [(-.18, 'pool_positive'), (.18, 'selected_positive')]:
            value = 100 * screening.loc[name, metric]
            ax.text(i + shift, value + 2, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x, ['U-shaped', 'Right-\nskewed', 'Uniform', 'Normal'], fontsize=9)
    ax.legend(frameon=False, loc='upper left', fontsize=8.5)
    ax.text(.5, 1.015, 'Initial-state diagnostic; no dating rounds', transform=ax.transAxes,
            ha='center', fontsize=8.2, color='#526475')

    ax = axes[3]
    style(ax, 'D  Sensitivity to trait count', 'P10 positive-outcome rate (%)', 112)
    trait_counts = [2, 5, 10]
    for dist, label, color, marker in [('u_shape', 'U-shaped', BLUE, 'o'), ('right_skew', 'Right-skewed', LIGHT, 's')]:
        values = [100 * dims.loc[(dist, d), 'positive_rate'] for d in trait_counts]
        ax.plot(trait_counts, values, color=color, marker=marker, linewidth=2, label=label)
        for d, value in zip(trait_counts, values):
            offset = -13 if dist == 'u_shape' and d == 2 else 6
            ax.annotate(f'{value:.1f}', (d, value), xytext=(0, offset), textcoords='offset points',
                        ha='center', fontsize=9, color=INK)
    ax.set_xticks(trait_counts)
    ax.set_xlabel('Number of traits', fontsize=10)
    ax.set_xlim(1, 11)
    ax.legend(frameon=False, loc='lower left', fontsize=8.5)

    ax = axes[4]
    ax.set_title('E  Rank and roster coverage', loc='left', fontsize=11.5, weight='bold', pad=15)
    for dist, label, color, marker in [('right_skew', 'Right-skewed', LIGHT, 's'), ('u_shape', 'U-shaped', BLUE, 'o')]:
        original = int(p10.loc[dist, 'rank'])
        common = int(shared.loc[dist, 'rank_with_mean_tiebreak'])
        ax.plot([0, 1], [original, common], color=color, marker=marker,
                linewidth=2, markersize=7 if dist == 'right_skew' else 5, label=label)
        ax.annotate(f'#{original}', (0, original), xytext=(-8, 0), textcoords='offset points', ha='right', va='center', fontsize=10)
    ax.text(1.08, 3, '#3 both', ha='left', va='center', fontsize=10)
    ax.set_xticks([0, 1], ['All recorded\nrosters', 'Shared roster\nfamilies'], fontsize=9)
    ax.set_ylabel('Rank among 10 strategies (1 is best)', fontsize=10)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(4.3, .45)
    ax.set_xlim(-.2, 1.5)
    ax.tick_params(length=0)
    ax.grid(axis='y', color='#E2E8F0', linewidth=.65)
    ax.legend(frameon=False, loc='lower left', fontsize=8.5)
    ax.text(.5, 1.015, 'Ties broken by mean happiness', transform=ax.transAxes,
            ha='center', fontsize=8.5, color='#526475')

    pdf = OUT / 'player10_distribution_results.pdf'
    png = OUT / 'player10_figures/player10_distribution_results.png'
    fig.savefig(pdf, facecolor='white', metadata={'Title': 'Player 10 distribution performance and mechanisms'})
    fig.savefig(png, dpi=220, facecolor='white')
    plt.close(fig)
    method = {'A': 'All recorded tournament seats; rank by positive rate then mean happiness.',
              'B': 'Exact independent unselected-pair calculation, d=5, m=2, B=100.',
              'C': 'Current initial advertisements and P10 shortlist in equal-group markets: 36 settings x three seeds per distribution; equal setting weights.',
              'D': 'Recorded P10 outcomes in C_exotic, pooled across all available rosters; half the traits are monotonic.',
              'E': 'Shared roster families are equal groups and solo+random; sampled half-half excluded; ranks use the mean-happiness tie-break.'}
    (DATA / 'figure_method.json').write_text(json.dumps(method, indent=2) + '\n')
    print(pdf)
    print(png)


if __name__ == '__main__':
    main()
