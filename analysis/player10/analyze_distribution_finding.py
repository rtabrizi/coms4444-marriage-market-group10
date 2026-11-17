"""Investigate finding 6 using archived outcomes and initial-state diagnostics.

No dating rounds or counterfactual games are run. Current player code is used
only to reconstruct initial advertisements and P10's initial shortlist.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/p10-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.engine import Engine
from core.registry import discover
from models.profile import build_distribution

OUT = ROOT / 'output/analysis/player10_distribution_finding'
SHAPES = ['u_shape', 'right_skew', 'uniform', 'normal']
GRID = ['bonus', 'rounds', 'dims', 'monotonic', 'distribution', 'people', 'seed']


def summarize(frame, keys):
    a = frame.groupby(keys)[['seats', 'positive', 'married', 'negative', 'zero_marriage', 'happiness']].sum()
    for name in ['positive', 'married', 'negative', 'zero_marriage']:
        a[name + '_rate'] = a[name] / a.seats
    a['mean_happiness'] = a.happiness / a.seats
    if 'group' in keys:
        strata = [key for key in keys if key != 'group']
        a['rank'] = a.groupby(level=strata).positive_rate.rank(
            ascending=False, method='min').astype(int)
        ordered = a.reset_index().sort_values(strata + ['positive_rate', 'mean_happiness'],
                                              ascending=[True] * len(strata) + [False, False])
        ordered['rank_with_mean_tiebreak'] = ordered.groupby(strata).cumcount() + 1
        a = ordered.set_index(keys).sort_index()
    return a


def random_pair_probability(row, dims, monotonic, bonus):
    """Exact P(B - D_own - D_partner/2 > 0) for an unselected iid pair."""
    p = np.array(row, dtype=float)
    p /= p.sum()
    values = np.arange(11)
    pmf = np.array([1.])
    for dim in range(dims):
        gap = values[:, None] - values[None, :]
        if dim < monotonic:
            gap = np.maximum(0, gap)
        loss = np.bincount((gap * gap).ravel(), weights=np.outer(p, p).ravel(), minlength=101)
        doubled = np.zeros(201)
        doubled[::2] = loss
        pmf = np.convolve(pmf, np.convolve(doubled, loss))
    assert np.isclose(pmf.sum(), 1)
    mean = p @ values
    variance = p @ ((values - mean) ** 2)
    assert np.isclose(np.arange(len(pmf)) @ pmf / 2, (3 * dims - 1.5 * monotonic) * variance)
    return float(pmf[:int(2 * bonus)].sum())


def initial_screening(r, f, rows):
    available, errors = discover()
    assert not errors
    # Exactly the same half-monotonic grid as the exotic tests, for all four
    # distributions. Every record is an equal-group run with the original seat
    # order; the normal/uniform controls come from A_core.
    source = r[r.roster.eq('equal') & r.group.eq('10') & r.distribution.isin(SHAPES)
        & r['slice'].isin(['A_core', 'C_exotic']) & r.monotonic.eq(r.dims // 2)]
    assert source.groupby('distribution').size().eq(108).all()
    records = []
    reconstructed = 0
    for c in source.itertuples():
        random.seed(int(c.seed))
        np.random.seed(int(c.seed))
        custom = [rows[c.distribution]] * c.dims
        codes = [str(g) for g in range(1, 11) for _ in range(c.people // 10)]
        engine = Engine([available[code] for code in codes], rounds=c.rounds,
            bonus=c.bonus, dims=c.dims, monotonic=c.monotonic, seed=c.seed,
            distribution=custom, timeout=0, keep_records=False)
        assert engine.round == 0 and not engine.marriages
        assert not engine.results()['faults']
        profiles, ideals = np.array(engine.profiles), np.array(engine.ideals)
        gap = ideals[:, None, :] - profiles[None, :, :]
        gap[:, :, :c.monotonic] = np.maximum(0, gap[:, :, :c.monotonic])
        loss = (gap * gap).sum(axis=2)
        scores = c.bonus - loss - .5 * loss.T
        observed = f[f.cell_id.eq(c.cell_id) & f.seed.eq(c.seed) & f.married.eq(1)]
        assert np.array_equal(scores[observed.seat.to_numpy(int), observed.partner.to_numpy(int)],
                              observed.happiness.to_numpy())
        reconstructed += len(observed)
        for i, code in enumerate(codes):
            if code != '10':
                continue
            agent = engine.players[i]
            candidates = [j for j in range(c.people) if j != i]
            ranked = sorted(candidates, key=lambda j: (agent._ad_cost(engine.ads[j]), j))
            count = max(1, int(len(ranked) * agent.fresh_candidate_fraction))
            selected = ranked[:count][:agent.list_limit]
            good = scores[i] > 0
            mutual = good & (scores[:, i] > 0)
            records.append(dict(distribution=c.distribution, cell_id=c.cell_id, seed=c.seed,
                bonus=c.bonus, rounds=c.rounds, dims=c.dims, people=c.people, seat=i,
                candidate_count=len(candidates), selected_count=len(selected),
                pool_positive=float(good[candidates].mean()),
                selected_positive=float(good[selected].mean()),
                pool_mutual_positive=float(mutual[candidates].mean()),
                selected_mutual_positive=float(mutual[selected].mean()),
                has_positive_partner=bool(good[candidates].any()),
                has_mutual_positive_partner=bool(mutual[candidates].any())))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'initial_screening_by_person.csv', index=False)
    metrics = ['pool_positive', 'selected_positive', 'pool_mutual_positive',
               'selected_mutual_positive', 'has_positive_partner', 'has_mutual_positive_partner']
    per_run = frame.groupby(['distribution', 'cell_id', 'seed', 'dims', 'bonus', 'rounds', 'people'])[metrics].mean()
    aggregate = per_run.groupby('distribution')[metrics].mean()
    aggregate['selection_lift_pp'] = 100 * (aggregate.selected_positive - aggregate.pool_positive)
    aggregate['mutual_selection_lift_pp'] = 100 * (aggregate.selected_mutual_positive - aggregate.pool_mutual_positive)
    aggregate.to_csv(OUT / 'initial_screening_summary.csv')
    per_run.groupby(['distribution', 'dims'])[metrics].mean().to_csv(OUT / 'initial_screening_by_dims.csv')
    return aggregate, reconstructed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = pd.read_pickle('/private/tmp/p10_runs.pkl')
    f = pd.read_pickle('/private/tmp/p10_seats.pkl')
    assert len(f) == 405540 and r.seats.sum() == 5510904
    e = r[r['slice'].eq('C_exotic') & r.group.ne('r')].copy()
    raw = summarize(e, ['distribution', 'group'])
    raw.to_csv(OUT / 'original_exotic_rankings.csv')
    by_roster = summarize(e, ['distribution', 'roster', 'group'])
    by_roster.to_csv(OUT / 'exotic_by_roster.csv')
    shared = summarize(e[e.roster.isin(['equal', 'solo+random'])], ['distribution', 'group'])
    assert shared.groupby(level='distribution').seats.nunique().eq(1).all()
    shared.to_csv(OUT / 'common_rosters_rankings.csv')
    dimensions = summarize(e[e.group.eq('10')], ['distribution', 'dims'])
    dimensions.to_csv(OUT / 'p10_by_dims.csv')
    pair_codes = e[e.roster.eq('half-half_sample')].groupby(['distribution', 'cell_id', 'seed']).group.agg(
        lambda values: '/'.join(sorted(values, key=int)))
    coverage = pair_codes.groupby(level='distribution').value_counts().rename('runs').reset_index(name='runs')
    coverage.to_csv(OUT / 'sampled_opponent_coverage.csv', index=False)
    sources = {'u_shape': ROOT / 'players/player_9/distribution.json',
               'right_skew': ROOT / 'players/player_2/distribution.json'}
    rows = {name: json.loads(path.read_text())['P'] for name, path in sources.items()}
    rows.update({name: list(build_distribution(name, 1, 0)[0]) for name in ['normal', 'uniform']})
    theory = []
    moments = {}
    values = np.arange(11)
    for name, row in rows.items():
        pmf = np.array(row, dtype=float)
        pmf /= pmf.sum()
        mean = float(pmf @ values)
        variance = float(pmf @ ((values - mean) ** 2))
        moments[name] = {'mean': mean, 'variance': variance}
        for dims in [2, 5, 10]:
            for bonus in [50, 100, 150]:
                monotonic = dims // 2
                theory.append(dict(distribution=name, dims=dims, monotonic=monotonic, bonus=bonus,
                    random_pair_positive=random_pair_probability(row, dims, monotonic, bonus),
                    random_pair_mean_happiness=bonus - (3 * dims - 1.5 * monotonic) * variance))
    theory = pd.DataFrame(theory)
    theory.to_csv(OUT / 'exact_random_pair_baseline.csv', index=False)
    screening, reconstructed = initial_screening(r, f, rows)

    # Compare timing in the same equal-group grid. Timing remains descriptive:
    # a fast marriage can be a consequence of finding a good match.
    equal = f[f.roster.eq('equal') & f.distribution.isin(SHAPES)
        & f['slice'].isin(['A_core', 'C_exotic']) & f.monotonic.eq(f.dims // 2)].copy()
    equal['early_married'] = equal.married.eq(1) & (equal.married_on / equal['rounds'] <= .1)
    equal['early_positive'] = equal.positive.eq(1) & equal.early_married
    equal['nonpositive_married'] = equal.married.eq(1) & equal.positive.eq(0)
    eq_times = equal.groupby(['distribution', 'cell_id', 'seed']).agg(
        seats=('seat', 'size'), positive=('positive', 'sum'), married=('married', 'sum'),
        early_married=('early_married', 'sum'), early_positive=('early_positive', 'sum'),
        nonpositive_married=('nonpositive_married', 'sum'))
    for name in ['positive', 'married', 'early_married', 'early_positive', 'nonpositive_married']:
        eq_times[name + '_rate'] = eq_times[name] / eq_times.seats
    rate_cols = [name for name in eq_times if name.endswith('_rate')]
    timing = eq_times.groupby('distribution')[rate_cols].mean()
    timing['early_share_of_marriages'] = timing.early_married_rate / timing.married_rate
    timing.to_csv(OUT / 'equal_common_grid_timing.csv')

    # A spouse-group join is possible for every equal-group seat retained in
    # the cache, and is restricted to the same grid as the screening analysis.
    all_equal = pd.read_pickle('/private/tmp/p10_equal_seats.pkl')
    spouses = equal[equal.married.eq(1)].merge(
        all_equal[['cell_id', 'seed', 'seat', 'group']].rename(
            columns={'seat': 'partner', 'group': 'spouse_group'}),
        on=['cell_id', 'seed', 'partner'], validate='one_to_one')
    spouse_summary = spouses.groupby(['distribution', 'spouse_group']).agg(
        married=('seat', 'size'), positive=('positive', 'sum'), negative=('negative', 'sum'))
    spouse_summary['negative_if_married'] = spouse_summary.negative / spouse_summary.married
    spouse_summary.to_csv(OUT / 'equal_grid_spouse_groups.csv')

    findings = {
        'original_p10': json.loads(raw.xs('10', level='group').to_json(orient='index')),
        'p10_by_roster': json.loads(by_roster.xs('10', level='group').reset_index().to_json(orient='records')),
        'common_rosters_all_players': json.loads(shared.reset_index().to_json(orient='records')),
        'distribution_moments': moments,
        'exact_random_pair_baseline': json.loads(theory.to_json(orient='records')),
        'initial_screening': json.loads(screening.to_json(orient='index')),
        'equal_common_grid_timing': json.loads(timing.to_json(orient='index')),
        'p10_by_dims': json.loads(dimensions.reset_index().to_json(orient='records')),
        'sampled_opponent_coverage': json.loads(coverage.to_json(orient='records')),
        'equal_grid_spouse_groups': json.loads(spouse_summary.reset_index().to_json(orient='records')),
        'method': {
            'games_run': 0, 'dating_rounds_run': 0, 'current_initial_markets_initialized': 432,
            'initial_screening_settings_per_distribution': 36, 'seeds': [1, 2, 3],
            'recorded_married_scores_reconstructed_and_verified': reconstructed,
            'initial_screening': 'All ten groups in equal proportions; current initial advertisements; P10 ranks fresh candidates with its submitted ad-cost rule and actual list cap; true scores are used only for offline evaluation.',
            'screening_weighting': 'Equal weights per initial environment and seed; average within-run P10 individuals first.',
            'rank_coverage': 'P10 and P8 have one exotic half-half opponent; other groups have two. Shared-roster rankings omit half-half_sample.',
            'limitations': ['Current initial advertisements are reconstructed, not recorded original advertisements.',
                'Initial shortlist enrichment does not measure the causal effect on final marriage outcomes.',
                'Common-roster pooled results still weight seats, including different relative weights of equal and solo+random populations.',
                'Only three seeds and a coarse environment grid; small rank gaps are descriptive.'],
            'distribution_files': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in sources.items()},
        },
    }
    (OUT / 'findings.json').write_text(json.dumps(findings, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    plot = screening.reindex(SHAPES)
    x = np.arange(4)
    axes[0].bar(x - .18, 100 * plot.pool_positive, width=.35, color='#cbd5e1', label='All initial candidates')
    axes[0].bar(x + .18, 100 * plot.selected_positive, width=.35, color='#2563eb', label='P10 initial shortlist')
    axes[0].set_xticks(x, ['U-shaped', 'Right-skewed', 'Uniform', 'Normal'])
    axes[0].set_ylabel('True positive-payoff candidates (%)')
    axes[0].set_ylim(0, 110)
    axes[0].set_title('Ad screening enriches the initial shortlist')
    axes[0].legend(frameon=False, fontsize=9)
    b = theory[(theory.dims == 5) & (theory.bonus == 100)].set_index('distribution').reindex(SHAPES)
    axes[1].bar(x, 100 * b.random_pair_positive, color='#2563eb')
    axes[1].set_xticks(x, ['U-shaped', 'Right-skewed', 'Uniform', 'Normal'])
    axes[1].set_ylabel('Positive payoff in an unselected pair (%)')
    axes[1].set_ylim(0, 110)
    axes[1].set_title('Five traits, two monotonic, bonus 100')
    for i, value in enumerate(b.random_pair_positive):
        axes[1].text(i, 100 * value + 2, f'{100*value:.1f}%', ha='center')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'distribution_mechanisms.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('INITIAL SCREENING\n', screening.round(5).to_string())
    print('TIMING\n', timing.round(5).to_string())
    print('VERIFIED SCORES', reconstructed)
    print('OUTPUT', OUT)


if __name__ == '__main__':
    main()
