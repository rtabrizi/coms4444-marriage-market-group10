"""Analyze market composition from cached tournament records, without simulations.

Usage: python analysis/player10/analyze_market_mechanisms.py
The caches are the original report's per-run aggregates and all P10 seats.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/p10-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.profile import build_distribution, sample_profile

OUT = ROOT / 'output/analysis/player10_market_mechanisms'
MARKETS = ['equal', 'equal+random', 'half-half', 'solo+random', '90random_10group']
GRID = ['base_people', 'bonus', 'rounds', 'dims', 'monotonic', 'distribution', 'seed']
COUNTS = ['seats', 'positive', 'married', 'negative', 'zero_marriage',
          'self_married', 'outside_married', 'self_positive', 'outside_positive',
          'self_negative', 'outside_negative', 'self_zero', 'outside_zero',
          'early_married', 'early_positive', 'early_negative']


def add_rates(frame):
    frame = frame.copy()
    for count in COUNTS[1:]:
        frame[count + '_rate'] = frame[count] / frame.seats
    frame['self_share_of_marriages'] = frame.self_married / frame.married
    frame['self_negative_if_married'] = frame.self_negative / frame.self_married.replace(0, np.nan)
    frame['outside_negative_if_married'] = frame.outside_negative / frame.outside_married.replace(0, np.nan)
    frame['self_positive_if_married'] = frame.self_positive / frame.self_married.replace(0, np.nan)
    frame['outside_positive_if_married'] = frame.outside_positive / frame.outside_married.replace(0, np.nan)
    frame['positive_if_married'] = frame.positive / frame.married
    frame['self_share_of_negatives'] = frame.self_negative / frame.negative.replace(0, np.nan)
    frame['early_share_of_marriages'] = frame.early_married / frame.married
    return frame


def reconstruct_scores(core):
    """Regenerate initial profiles only, then validate against recorded scores.

    No engine, players, dates, matching, or simulations are executed. Every
    reconstructed married score must match its recorded value exactly.
    """
    core = core.copy()
    core['reconstructed_happiness'] = np.nan
    core['p10_self_estimate'] = np.nan
    structure = ['people', 'dims', 'monotonic', 'distribution', 'seed']
    for (people, dims, monotonic, distribution, seed), part in core.groupby(structure):
        rng = random.Random(int(seed))
        distribution_table = build_distribution(distribution, int(dims), int(monotonic))
        truth = [(sample_profile(rng, distribution_table), sample_profile(rng, distribution_table))
                 for _ in range(int(people))]
        profiles = np.array([pair[0] for pair in truth])
        ideals = np.array([pair[1] for pair in truth])
        own = part.seat.to_numpy(int)
        partner = part.partner.fillna(0).to_numpy(int)
        own_profile, own_ideal = profiles[own], ideals[own]
        partner_profile, partner_ideal = profiles[partner], ideals[partner]

        def costs(wanted, actual):
            gap = wanted - actual
            gap[:, :monotonic] = np.maximum(0, gap[:, :monotonic])
            return (gap * gap).sum(axis=1)

        actual_score = part.bonus.to_numpy() - costs(own_ideal, partner_profile) - .5 * costs(partner_ideal, own_profile)
        married = part.married.to_numpy(bool)
        assert np.array_equal(actual_score[married], part.happiness.to_numpy()[married])
        core.loc[part.index[married], 'reconstructed_happiness'] = actual_score[married]
        projection = partner_profile.copy()
        projection[:, :monotonic] = np.minimum(10, projection[:, :monotonic] + 2)
        estimated = part.bonus.to_numpy() - costs(own_ideal, projection) - .5 * costs(partner_ideal, own_profile)
        self_match = part.self_married.to_numpy(bool)
        core.loc[part.index[self_match], 'p10_self_estimate'] = estimated[self_match]
    assert core.loc[core.married.eq(1), 'reconstructed_happiness'].notna().all()
    self_matches = core[core.self_married.eq(1)].copy()
    self_matches['estimated_threshold'] = .25 * self_matches.bonus * (
        (self_matches['rounds'] - self_matches.married_on + 1) / self_matches['rounds']) ** 2
    self_matches['prediction_error'] = self_matches.p10_self_estimate - self_matches.happiness
    assert (self_matches.prediction_error >= 0).all()
    assert (self_matches.p10_self_estimate > self_matches.estimated_threshold).all()
    self_matches['nonpositive_true_score'] = self_matches.happiness.le(0).astype(int)
    self_matches['truth_below_threshold'] = self_matches.happiness.le(self_matches.estimated_threshold).astype(int)
    fields = dict(seats=('seat', 'size'), negative=('negative', 'sum'),
        nonpositive=('nonpositive_true_score', 'sum'), truth_below_threshold=('truth_below_threshold', 'sum'),
        mean_true_happiness=('happiness', 'mean'), mean_estimate=('p10_self_estimate', 'mean'),
        mean_prediction_error=('prediction_error', 'mean'))
    self_matches.groupby('roster').agg(**fields).to_csv(OUT / 'core_self_match_calibration.csv')
    self_matches[self_matches.negative.eq(1)].groupby('roster').agg(**fields).to_csv(
        OUT / 'core_negative_self_match_calibration.csv')
    return {'married_scores_reconstructed_and_matched': int(core.married.sum()),
        'self_match_seats_with_reconstructed_estimates': len(self_matches),
        'self_match_seats_above_submitted_policy_threshold': int((self_matches.p10_self_estimate > self_matches.estimated_threshold).sum()),
        'negative_self_match_seats_above_threshold': int(self_matches.negative.sum()),
        'nonpositive_self_match_seats_above_threshold': int(self_matches.nonpositive_true_score.sum()),
        'interpretation': 'Estimates reconstructed under checked-out +2 P10 policy, not original decision logs or an honest-policy simulation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=Path('/private/tmp'))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [args.cache_dir / name for name in ['p10_runs.pkl', 'p10_seats.pkl']]
    r, f = [pd.read_pickle(path) for path in paths]
    assert r.seats.sum() == 5510904
    assert len(r[['cell_id', 'seed']].drop_duplicates()) == 80520
    assert len(f) == 405540
    assert not f.duplicated(['cell_id', 'seed', 'seat']).any()
    assert (f.positive == (f.happiness > 0)).all()
    assert (f.negative == (f.happiness < 0)).all()
    assert (r.married == r.positive + r.negative + r.zero_marriage).all()
    raw_check = f.groupby(['cell_id', 'seed']).agg(seats=('seat', 'size'), positive=('positive', 'sum'),
            married=('married', 'sum'), negative=('negative', 'sum'), happiness=('happiness', 'sum'))
    run_check = r[r.group == '10'].set_index(['cell_id', 'seed'])[raw_check.columns].sort_index()
    pd.testing.assert_frame_equal(raw_check.sort_index(), run_check, check_dtype=False)

    # All P10 seats are retained: a partner-ID lookup identifies every P10 spouse
    # exactly, even though other groups' seat records are not retained everywhere.
    lookup = f[['cell_id', 'seed', 'seat', 'partner']].rename(
        columns={'seat': 'partner', 'partner': 'partner_back'})
    lookup['partner_is_p10'] = True
    f = f.merge(lookup, on=['cell_id', 'seed', 'partner'], how='left', validate='many_to_one')
    own = f.partner_is_p10.eq(True)
    married = f.married.eq(1)
    assert (married[own]).all()
    assert (f.loc[own, 'partner_back'] == f.loc[own, 'seat']).all()
    assert own.sum() == 207856
    assert f.loc[own, 'negative'].sum() == 11421
    f['seats'] = 1
    for cohort, mask in [('self', own & married), ('outside', ~own & married)]:
        f[cohort + '_married'] = mask.astype(int)
        f[cohort + '_positive'] = (mask & f.positive.eq(1)).astype(int)
        f[cohort + '_negative'] = (mask & f.negative.eq(1)).astype(int)
        f[cohort + '_zero'] = (mask & f.zero_marriage.eq(1)).astype(int)
    early = married & (f.married_on / f['rounds'] <= .05)
    f['early_married'] = early.astype(int)
    f['early_positive'] = (early & f.positive.eq(1)).astype(int)
    f['early_negative'] = (early & f.negative.eq(1)).astype(int)
    f['mode'] = np.select([f.monotonic == 0, f.monotonic == f.dims], ['none', 'all'], default='half')
    # P10s report true ideals and inflate only monotonic dimensions. With no
    # monotonic traits a P10-P10 score estimate is exact and stays positive.
    assert f.loc[f.monotonic.eq(0), 'self_negative'].sum() == 0
    full = add_rates(f.groupby('roster')[COUNTS].sum())
    full.to_csv(OUT / 'all_settings.csv')

    core = f[f['slice'] == 'A_core'].copy()
    core['base_people'] = core.people - np.where(core.roster.eq('equal+random'), 2, 0)
    reconstruction = reconstruct_scores(core)
    assert set(core.roster) == set(MARKETS)
    cells = add_rates(core.groupby(['roster'] + GRID)[COUNTS].sum()).reset_index()
    grid_sets = {m: set(map(tuple, cells[cells.roster == m][GRID].to_numpy())) for m in MARKETS}
    assert all(grid_sets[m] == grid_sets['equal'] for m in MARKETS)
    assert len(grid_sets['equal']) == 972  # 324 settings x 3 seeds.
    rate_cols = [count + '_rate' for count in COUNTS[1:]]
    standardized = cells.groupby('roster')[rate_cols].mean()
    standardized['self_share_of_marriages'] = standardized.self_married_rate / standardized.married_rate
    standardized['self_negative_if_married'] = standardized.self_negative_rate / standardized.self_married_rate
    standardized['outside_negative_if_married'] = standardized.outside_negative_rate / standardized.outside_married_rate
    standardized['early_share_of_marriages'] = standardized.early_married_rate / standardized.married_rate
    standardized['settings'] = 324
    standardized['seeds'] = 3
    standardized.to_csv(OUT / 'common_core_standardized.csv')
    cells.to_csv(OUT / 'common_core_cells.csv', index=False)
    add_rates(core.groupby('roster')[COUNTS].sum()).to_csv(OUT / 'common_core_seat_weighted.csv')
    by_seed = cells.groupby(['seed', 'roster'])[rate_cols].mean()
    by_seed.to_csv(OUT / 'common_core_by_seed.csv')
    cells['mode'] = np.select([cells.monotonic == 0, cells.monotonic == cells.dims], ['none', 'all'], default='half')
    for axis in ['mode', 'dims', 'bonus', 'distribution', 'rounds', 'base_people']:
        cells.groupby([axis, 'roster'])[rate_cols].mean().to_csv(OUT / f'common_core_by_{axis}.csv')

    # The core roster generator assigns groups to fixed blocks of seats.
    # Equal-group P10s occupy the last numbered seats; solo/random-heavy P10s
    # occupy the first seats. Match the exact focal individuals where possible.
    # Equal+random adds its two extra people after the numbered population, so
    # existing numbered seats keep their profiles and ideals under the seed.
    anchors = core[core.roster.eq('equal')][GRID + ['seat']]
    same_people = core.merge(anchors, on=GRID + ['seat'], validate='many_to_one')
    assert set(same_people.roster) == {'equal', 'equal+random', 'half-half'}
    matched = add_rates(same_people.groupby(['roster'] + GRID)[COUNTS].sum()).reset_index()
    assert (matched.groupby('roster').size() == 972).all()
    matched_rates = matched.groupby('roster')[rate_cols].mean()
    matched_rates['self_share_of_marriages'] = matched_rates.self_married_rate / matched_rates.married_rate
    matched_rates['self_negative_if_married'] = matched_rates.self_negative_rate / matched_rates.self_married_rate
    matched_rates['outside_negative_if_married'] = matched_rates.outside_negative_rate / matched_rates.outside_married_rate
    matched_rates.to_csv(OUT / 'identical_p10_people_standardized.csv')
    matched.to_csv(OUT / 'identical_p10_people_cells.csv', index=False)
    matched.groupby(['seed', 'roster'])[rate_cols].mean().to_csv(OUT / 'identical_p10_people_by_seed.csv')
    matched['mode'] = np.select([matched.monotonic == 0, matched.monotonic == matched.dims], ['none', 'all'], default='half')
    matched.groupby(['mode', 'roster'])[rate_cols].mean().to_csv(OUT / 'identical_p10_people_by_mode.csv')
    matched.groupby(['base_people', 'roster'])[rate_cols].mean().to_csv(OUT / 'identical_p10_people_by_population.csv')
    pd.DataFrame([
        {'reference': ref, 'comparison': 'half-half',
         **{metric + '_gap_pp': 100 * (matched_rates.loc[ref, metric] - matched_rates.loc['half-half', metric])
            for metric in rate_cols}}
        for ref in ['equal', 'equal+random']
    ]).to_csv(OUT / 'identical_p10_people_gap_decomposition.csv', index=False)

    # A second matched-person contrast changes P10 density while keeping the
    # remaining strategy random, at the same total N and initial true profiles.
    random_anchors = core[core.roster.eq('90random_10group')][GRID + ['seat']]
    density_people = core[core.roster.isin(['90random_10group', 'solo+random'])].merge(
        random_anchors, on=GRID + ['seat'], validate='many_to_one')
    density_cells = add_rates(density_people.groupby(['roster'] + GRID)[COUNTS].sum()).reset_index()
    assert (density_cells.groupby('roster').size() == 972).all()
    density_rates = density_cells.groupby('roster')[rate_cols].mean()
    density_rates['self_share_of_marriages'] = density_rates.self_married_rate / density_rates.married_rate
    density_rates.to_csv(OUT / 'identical_p10_people_density.csv')
    density_cells.groupby(['seed', 'roster'])[rate_cols].mean().to_csv(OUT / 'identical_p10_people_density_by_seed.csv')
    density_cells['mode'] = np.select([density_cells.monotonic == 0, density_cells.monotonic == density_cells.dims], ['none', 'all'], default='half')
    density_cells.groupby(['mode', 'roster'])[rate_cols].mean().to_csv(OUT / 'identical_p10_people_density_by_mode.csv')

    contrasts = []
    for reference in ['equal', 'equal+random']:
        for market in [m for m in MARKETS if m != reference]:
            delta = standardized.loc[reference, rate_cols] - standardized.loc[market, rate_cols]
            assert np.isclose(delta.positive_rate, delta.married_rate - delta.negative_rate - delta.zero_marriage_rate)
            contrasts.append({'reference': reference, 'comparison': market,
                **{name + '_gap_pp': 100 * delta[name] for name in rate_cols},
                'self_negative_contribution_pp': -100 * delta.self_negative_rate,
                'outside_negative_contribution_pp': -100 * delta.outside_negative_rate})
    pd.DataFrame(contrasts).to_csv(OUT / 'common_core_gap_decomposition.csv', index=False)

    # Rank and relative advantage on the same core settings. For half-half,
    # P10's comparison is the other numbered group in its own run.
    numbered = r[(r['slice'] == 'A_core') & (r.group != 'r')].copy()
    peer = numbered[numbered.group != '10'].groupby(['cell_id', 'seed'])[
        ['seats', 'positive', 'married', 'negative', 'zero_marriage']].sum()
    focal = numbered[numbered.group == '10'].set_index(['cell_id', 'seed'])
    together = focal.join(peer, rsuffix='_peer', how='inner')
    together['base_people'] = together.people - np.where(together.roster.eq('equal+random'), 2, 0)
    for metric in ['positive', 'married', 'negative', 'zero_marriage']:
        together[metric + '_gap'] = together[metric] / together.seats - together[metric + '_peer'] / together.seats_peer
    peer_cols = [metric + '_gap' for metric in ['positive', 'married', 'negative', 'zero_marriage']]
    together.groupby(['roster'] + GRID)[peer_cols].mean().groupby('roster').mean().to_csv(OUT / 'common_core_peer_gaps.csv')

    # Two-panel descriptive plot; equal setting weights in every market.
    names = ['Equal groups', 'Equal + random', 'Half-and-half', 'Group + random', '90% random']
    plot = standardized.reindex(MARKETS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    x = np.arange(5)
    axes[0].bar(x, 100 * plot.married_rate, color='#d9e7f3', label='All marriages')
    axes[0].bar(x, 100 * plot.positive_rate, color='#2563eb', label='Positive marriages')
    for i, value in enumerate(plot.positive_rate):
        axes[0].text(i, value * 100 - 4, f'{value*100:.2f}%', ha='center', color='white', weight='bold', fontsize=9)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Share of initial P10 seats (%)')
    axes[0].set_title('Marriage frequency and positive outcomes')
    axes[0].legend(frameon=False, loc='lower left', fontsize=9)
    axes[1].bar(x, 100 * plot.self_negative_rate, color='#2563eb', label='Married to P10')
    axes[1].bar(x, 100 * plot.outside_negative_rate, bottom=100 * plot.self_negative_rate,
                color='#b5c8e3', label='Married outside P10')
    axes[1].set_ylabel('Negative outcomes / initial P10 seats (%)')
    axes[1].set_title('Where negative outcomes occur')
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.set_xticks(x, names, rotation=18, ha='right', fontsize=9)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('P10 across 324 common core settings and three seeds', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT / 'common_core_outcomes.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    matched_order = ['equal', 'equal+random', 'half-half']
    p = matched_rates.reindex(matched_order)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    x = np.arange(3)
    axes[0].bar(x, 100 * p.married_rate, color='#d9e7f3', label='All marriages')
    axes[0].bar(x, 100 * p.positive_rate, color='#2563eb', label='Positive marriages')
    for i, value in enumerate(p.positive_rate):
        axes[0].text(i, value * 100 - 5, f'{value*100:.2f}%', ha='center', color='white', weight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Share of initial P10 seats (%)')
    axes[0].set_title('Positive outcomes and marriage rate')
    axes[0].legend(frameon=False, loc='lower left', fontsize=9)
    axes[1].bar(x, 100 * p.self_negative_rate, color='#2563eb', label='Married to P10')
    axes[1].bar(x, 100 * p.outside_negative_rate, bottom=100 * p.self_negative_rate,
                color='#b5c8e3', label='Married outside P10')
    axes[1].set_ylabel('Negative outcomes / initial P10 seats (%)')
    axes[1].set_title('Where negative outcomes occur')
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.set_xticks(x, ['Equal groups', 'Equal + random', 'Half-and-half'], fontsize=10)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Same P10 individuals across 324 core settings and three seeds', fontsize=13)
    fig.text(.5, .015, 'Equal weights per setting and seed. Equal + random adds two random players. Half-and-half pools nine opponents.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .95))
    fig.savefig(OUT / 'identical_p10_people_outcomes.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    provenance = {
        'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        'simulations_run': 0, 'raw_archive_available': (ROOT / 'tuned_p10_tournament_results.zip').exists(),
        'analysis': 'Descriptive comparisons from the cached original tournament outcomes.',
        'common_core_settings': 324, 'seeds': [1, 2, 3],
        'weighting': 'Each environment and seed receives equal weight; half-half opponents are pooled within environment/seed.',
        'equal_plus_random_population': 'Two extra random players are added to the same numbered-player population; total N differs by two.',
        'partner_classification': 'Exact partner-ID membership in the complete P10 seat cache; reciprocal self-partner links checked.',
        'score_reconstruction': reconstruction,
        'identical_focal_people': {'rosters': ['equal', 'equal+random', 'half-half'],
            'equal_seats': int(same_people.roster.eq('equal').sum()),
            'equal_plus_random_seats': int(same_people.roster.eq('equal+random').sum()),
            'half_half_seats': int(same_people.roster.eq('half-half').sum()),
            'interpretation': 'Same profile/ideal at same seat, seed and core parameters; nine half-half opponents per environment.'},
        'limitations': ['Observational roster comparisons do not isolate density, diversity, opponents, and population size.',
            'Only three seeds; no formal significance claims or independent-seed confidence intervals.',
            'No date-level estimates or proposal logs are available in these caches.',
            'P10 seat assignments and sampled traits differ across rosters in the all-five comparison; identical-person analysis addresses this for three rosters only.',
            'All comparisons describe recorded code versions; original archive has no player code hashes.',
            'Sampled half-half is excluded from common core because it has no A_core records.'],
    }
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    findings = {
        'method': provenance,
        'common_settings_all_focal_seats': json.loads(standardized.to_json(orient='index')),
        'identical_p10_people_comparison': json.loads(matched_rates.to_json(orient='index')),
        'identical_p10_people_by_monotonicity': json.loads(matched.groupby(['mode', 'roster'])[rate_cols].mean().reset_index().to_json(orient='records')),
        'identical_p10_people_density_comparison': json.loads(density_rates.to_json(orient='index')),
        'negative_self_match_calibration': json.loads(pd.read_csv(OUT / 'core_negative_self_match_calibration.csv').to_json(orient='records')),
    }
    (OUT / 'findings.json').write_text(json.dumps(findings, indent=2) + '\n')
    print('STANDARDIZED COMMON CORE (percentages)')
    print((100 * standardized.drop(columns=['settings', 'seeds'])).round(3).to_string())
    print('BY SEED POSITIVE RATE')
    print((100 * by_seed.positive_rate.unstack()).round(3).to_string())
    print('ALL SETTINGS SELF MATCHES')
    print(full[['seats', 'self_share_of_marriages', 'self_negative_if_married', 'outside_negative_if_married', 'self_share_of_negatives']].round(5).to_string())
    print('OUTPUT', OUT)


if __name__ == '__main__':
    main()
