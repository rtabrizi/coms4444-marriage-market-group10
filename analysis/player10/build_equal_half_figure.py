"""Build a paper-ready six-panel P10 equal-group / half-half comparison.

Panels A-E use raw tournament outcomes; panel F standardizes common core
settings for every strategy. E's half-half ranking is limited to P10-containing
markets because only P10 and equal-group individual records are cached.
No simulations are run.
"""

import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/p10-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'output/analysis/player10_market_mechanisms'
OUT = ROOT / 'output/pdf'
STEM = 'player10_equal_vs_half_half'
ROSTERS = ['equal', 'half-half']
LABELS = ['Equal groups', 'Half-and-half']
COLORS = ['#1D4ED8', '#93C5FD']
INK = '#183249'


def same_strategy_ranks(r):
    equal = pd.read_pickle('/private/tmp/p10_equal_seats.pkl')
    married = equal[equal.married.eq(1)].merge(
        equal[['cell_id', 'seed', 'seat', 'group']].rename(
            columns={'seat': 'partner', 'group': 'spouse_group'}),
        on=['cell_id', 'seed', 'partner'], validate='one_to_one')
    married['same'] = married.group.eq(married.spouse_group)
    eq = married.groupby('group').agg(married=('seat', 'size'), same=('same', 'sum'))
    eq['roster'] = 'equal'

    # In a two-strategy market each cross-strategy couple contributes one
    # married seat to each side. Therefore opponent self-married seats equal
    # opponent married seats minus P10 cross-married seats, exactly.
    p10 = pd.read_pickle('/private/tmp/p10_seats.pkl')
    half = p10[p10.roster.eq('half-half')].copy()
    own = half[['cell_id', 'seed', 'seat']].rename(columns={'seat': 'partner'})
    own['same'] = True
    half = half.merge(own, on=['cell_id', 'seed', 'partner'], how='left', validate='many_to_one')
    half['cross'] = half.married.eq(1) & half.same.isna()
    cross = half.groupby(['cell_id', 'seed'])['cross'].sum().rename('cross')
    peers = r[r.roster.eq('half-half') & r.group.ne('10')].merge(
        cross, on=['cell_id', 'seed'], validate='many_to_one')
    assert not peers.duplicated(['cell_id', 'seed']).any()
    peers['same'] = peers.married - peers['cross']
    assert peers.same.ge(0).all() and (peers.same % 2 == 0).all()
    hh = peers.groupby('group')[['married', 'same']].sum()
    hh.loc['10', ['married', 'same']] = [half.married.sum(), half.same.eq(True).sum()]
    hh['roster'] = 'half-half'
    result = pd.concat([eq.reset_index(), hh.reset_index()], ignore_index=True)
    result['share'] = result.same / result.married
    result['rank'] = result.groupby('roster')['share'].rank(ascending=False, method='min').astype(int)
    assert result.groupby('roster').group.nunique().eq(10).all()
    return result.set_index(['roster', 'group'])


def common_core_ranks(r):
    core = r[r['slice'].eq('A_core') & r.roster.isin(ROSTERS) & r.group.ne('r')]
    grid = ['people', 'bonus', 'rounds', 'dims', 'monotonic', 'distribution', 'seed']
    cells = core.groupby(['roster', 'group'] + grid)[['seats', 'positive']].sum().reset_index()
    assert cells.groupby(['roster', 'group']).size().eq(972).all()
    cells['positive_rate'] = cells.positive / cells.seats
    cells['mode'] = np.select([cells.monotonic.eq(0), cells.monotonic.eq(cells.dims)], ['none', 'all'], default='half')
    result = cells.groupby(['roster', 'mode', 'group']).positive_rate.mean().to_frame()
    result['rank'] = result.groupby(level=['roster', 'mode']).positive_rate.rank(
        ascending=False, method='min').astype(int)
    assert result.groupby(level=['roster', 'mode']).size().eq(10).all()
    return result


def main():
    r = pd.read_pickle('/private/tmp/p10_runs.pkl')
    assert r.seats.sum() == 5510904
    a = r[r.roster.isin(ROSTERS) & r.group.ne('r')].groupby(['roster', 'group'])[
        ['seats', 'positive', 'married', 'negative', 'happiness']].sum()
    for count in ['positive', 'married', 'negative']:
        a[count + '_rate'] = a[count] / a.seats
    a['mean_happiness'] = a.happiness / a.seats
    for metric in ['positive_rate', 'married_rate', 'negative_rate', 'mean_happiness']:
        a[metric + '_rank'] = a.groupby(level='roster')[metric].rank(
            ascending=metric == 'negative_rate', method='min').astype(int)
    focal = a.xs('10', level='group').reindex(ROSTERS)
    assert focal.seats.tolist() == [14448, 284310]
    assert focal.positive_rate_rank.tolist() == [2, 7]
    assert focal.mean_happiness_rank.tolist() == [5, 9]
    cohorts = pd.read_csv(DATA / 'all_settings.csv').set_index('roster').reindex(ROSTERS)
    np.testing.assert_allclose(cohorts.positive_rate, focal.positive_rate)
    self_ranks = same_strategy_ranks(r)
    np.testing.assert_allclose(self_ranks.xs('10', level='group').reindex(ROSTERS)['share'], cohorts.self_share_of_marriages)
    all_modes = common_core_ranks(r)
    modes = all_modes.xs('10', level='group')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    fig.subplots_adjust(left=.075, right=.985, bottom=.09, top=.88, hspace=.52, wspace=.35)
    fig.legend(handles=[Patch(facecolor=c, label=label) for c, label in zip(COLORS, LABELS)],
               loc='upper center', bbox_to_anchor=(.5, .985), ncol=2,
               frameon=False, fontsize=12, handlelength=1.4, columnspacing=2.5)

    def format_axis(ax, title, ylabel, ymax):
        ax.set_title(title, loc='left', fontsize=12, weight='bold', pad=13)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 1], LABELS, fontsize=10)
        ax.tick_params(axis='both', length=0)
        ax.grid(axis='y', color='#E2E8F0', linewidth=.65)
        ax.set_axisbelow(True)
        ax.margins(x=.2)

    def metric_bars(ax, metric, title, ylabel, ymax, pct=True):
        values = focal[metric].to_numpy() * (100 if pct else 1)
        format_axis(ax, title, ylabel, ymax)
        bars = ax.bar([0, 1], values, width=.58, color=COLORS)
        for i, (bar, value) in enumerate(zip(bars, values)):
            rank = int(focal.iloc[i][metric + '_rank'])
            label = f'{value:.2f}' + ('%' if pct else '')
            ax.text(bar.get_x() + bar.get_width()/2, value + .025 * ymax,
                    f'{label}\nrank {rank}/10', ha='center', va='bottom',
                    fontsize=10, linespacing=1.35, color=INK)

    metric_bars(axes[0, 0], 'positive_rate', 'A  Positive outcomes', 'P10 players (%)', 112)
    metric_bars(axes[0, 1], 'married_rate', 'B  Marriage rate', 'P10 players (%)', 112)
    metric_bars(axes[0, 2], 'negative_rate', 'C  Negative outcomes', 'P10 players (%)', 4.8)
    axes[0, 2].text(.02, .94, 'Lower is better', transform=axes[0, 2].transAxes,
                    color='#526475', fontsize=9, va='top')
    metric_bars(axes[1, 0], 'mean_happiness', 'D  Mean happiness', 'Happiness per P10 player', 100, pct=False)

    ax = axes[1, 1]
    format_axis(ax, 'E  Same-strategy marriages', 'Share of P10 marriages (%)', 112)
    values = 100 * cohorts.self_share_of_marriages.to_numpy()
    ax.bar([0, 1], values, width=.58, color=COLORS)
    for i, value in enumerate(values):
        rank = int(self_ranks.loc[(ROSTERS[i], '10'), 'rank'])
        ax.text(i, value + 3, f'{value:.2f}%\nrank {rank}/10',
                ha='center', va='bottom', fontsize=10, linespacing=1.35)

    ax = axes[1, 2]
    ax.set_title('F  Common core settings', loc='left', fontsize=12, weight='bold', pad=13)
    x = np.arange(3)
    order = ['none', 'half', 'all']
    for j, roster in enumerate(ROSTERS):
        values = [100 * modes.loc[(roster, mode), 'positive_rate'] for mode in order]
        centers = x + (j - .5) * .37
        ax.bar(centers, values, width=.34, color=COLORS[j])
        for center, value, mode in zip(centers, values, order):
            rank = int(modes.loc[(roster, mode), 'rank'])
            ax.text(center, value + 2, f'{value:.1f}', ha='center', va='bottom', fontsize=8.5, color=INK)
            ax.text(center, value - 6, f'#{rank}', ha='center', va='center',
                    fontsize=9, color='white' if j == 0 else INK, weight='bold')
    ax.set_xticks(x, ['None', 'Half', 'All'], fontsize=10)
    ax.set_xlabel('Monotonic traits', fontsize=10)
    ax.set_ylabel('Positive P10 outcomes (%)', fontsize=10)
    ax.set_ylim(0, 112)
    ax.tick_params(axis='both', length=0)
    ax.grid(axis='y', color='#E2E8F0', linewidth=.65)
    ax.set_axisbelow(True)
    ax.set_xlim(-.6, 2.6)

    pdf = OUT / (STEM + '.pdf')
    png = OUT / 'player10_figures' / (STEM + '.png')
    fig.savefig(pdf, metadata={'Title': 'Player 10: equal groups versus half-and-half'}, facecolor='white')
    fig.savefig(png, dpi=220, facecolor='white')
    plt.close(fig)
    source = {
        'raw_p10_metrics_and_ranks': json.loads(focal.to_json(orient='index')),
        'raw_p10_spouse_statistics': json.loads(cohorts[['self_share_of_marriages', 'self_share_of_negatives']].to_json(orient='index')),
        'same_strategy_shares_and_ranks': json.loads(self_ranks.reset_index().to_json(orient='records')),
        'common_core_monotonicity_all_strategies': json.loads(all_modes.reset_index().to_json(orient='records')),
        'raw_sample_sizes': {'equal': 14448, 'half-half': 284310},
        'rank_convention': '1 is highest positive rate, marriage rate, mean happiness or same-strategy marriage share; 1 is lowest negative rate. Ties use minimum rank.',
        'sample_convention': 'A-D use all available raw observations. E half-half benchmarks the ten strategies in P10-containing games only. F uses all seats per strategy on the same core setting grid with equal weights, not matched focal individuals.',
    }
    (DATA / 'equal_vs_half_half_figure_data.json').write_text(json.dumps(source, indent=2) + '\n')
    print(pdf)
    print(png)


if __name__ == '__main__':
    main()
