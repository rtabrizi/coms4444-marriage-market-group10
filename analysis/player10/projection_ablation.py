"""Factorial ablation of P10's outgoing exaggeration and incoming trust.

Runs new paired games with current code. Policies change every P10 in a market;
other strategies, P10's ads, proposal threshold and date selection stay fixed.
No-monotonic games are exact negative controls. Rerunning resumes safely.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/p10-matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.engine import Engine
from core.registry import discover
from players.player_10.player import Player10

OUT = ROOT / 'output/analysis/player10_projection_ablation'
POLICIES = ['current', 'truthful', 'discount', 'truthful_discount']
ROSTERS = ['equal', 'equal+random', 'half-P2', 'half-P9', 'self-play']
SEEDS = list(range(28001, 28009))
_AVAILABLE = None


class CurrentP10(Player10):
    EXAGGERATE = True
    DISCOUNT = False

    def __init__(self, snapshot, ctx):
        super().__init__(snapshot, ctx)
        self.monotonic_exaggeration = 2 if self.EXAGGERATE else 0

    def calculate_estimate(self, their_profile, their_ideal):
        if not self.DISCOUNT:
            return super().calculate_estimate(their_profile, their_ideal)
        believed = list(their_profile)
        for dim in range(self.monotonic):
            believed[dim] = max(0, believed[dim] - 2)
        return super().calculate_estimate(tuple(believed), their_ideal)


class TruthfulP10(CurrentP10):
    EXAGGERATE = False


class DiscountP10(CurrentP10):
    DISCOUNT = True


class TruthfulDiscountP10(CurrentP10):
    EXAGGERATE = False
    DISCOUNT = True


CLASSES = {'current': CurrentP10, 'truthful': TruthfulP10,
           'discount': DiscountP10, 'truthful_discount': TruthfulDiscountP10}


def scenarios():
    result = []
    for roster, distribution, monotonic, bonus, rounds in product(ROSTERS,
            ['mixed', 'uniform', 'u_shape', 'right_skew'], [0, 2, 5], [50, 100], [50, 150]):
        result.append(dict(scenario_id=len(result), roster=roster, distribution=distribution,
                           dims=5, monotonic=monotonic, bonus=bonus, rounds=rounds))
    return result


def roster_codes(c):
    if c['roster'].startswith('equal'):
        codes = [str(g) for g in range(1, 11) for _ in range(2)]
        if c['roster'] == 'equal+random':
            codes += ['r', 'r']
        return codes
    if c['roster'] == 'self-play':
        return ['10']*20
    return ['2' if c['roster'] == 'half-P2' else '9']*10 + ['10']*10


def distribution_for(c):
    if c['distribution'] not in ['u_shape', 'right_skew']:
        return c['distribution']
    group = '9' if c['distribution'] == 'u_shape' else '2'
    row = json.loads((ROOT / f'players/player_{group}/distribution.json').read_text())['P']
    return [row] * c['dims']


def game_signature(result):
    return [(p['happiness'], p['partner'], p['married_on'], p['dates'], p['proposals_made'])
            for p in result['players']]


def run_pair(job):
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE, errors = discover()
        assert not errors
    c, seed = job
    codes = roster_codes(c)
    initial, control = None, None
    worlds = {}
    for policy in POLICIES:
        random.seed(seed)
        np.random.seed(seed)
        engine = Engine([CLASSES[policy] if code == '10' else _AVAILABLE[code] for code in codes],
            rounds=c['rounds'], bonus=c['bonus'], dims=c['dims'], monotonic=c['monotonic'],
            seed=seed, distribution=distribution_for(c), timeout=0, keep_records=False)
        state = (engine.profiles, engine.ideals, engine.ads, engine.rng.getstate())
        if initial is None:
            initial = state
        else:
            assert state == initial, 'Initial populations, ads or RNG differ across policies'
        result = engine.run()
        assert not result['faults'], (c, seed, policy, result['faults'])
        if c['monotonic'] == 0:
            signature = game_signature(result)
            if control is None:
                control = signature
            else:
                assert signature == control, 'No-monotonic negative control failed'
        focal, others = [], []
        for i, person in enumerate(result['players']):
            if codes[i] != '10':
                others.append(float(person['happiness']))
                continue
            married = bool(person['married'])
            partner = person['partner']
            same = married and codes[partner] == '10'
            estimate = float(engine.players[i].estimates[partner]) if married else None
            if married:
                assert estimate > 0
            if same and policy != 'current':
                # Truthful self-reports, or subtracting the full +2 inflation,
                # cannot overstate P10-P10 compatibility (including the cap at 10).
                assert person['happiness'] > 0, 'Safe self-marriage bound failed'
            focal.append({'seat': i, 'happiness': float(person['happiness']), 'married': married,
                'married_on': person['married_on'], 'partner': partner,
                'same_strategy_spouse': bool(same), 'final_estimate': estimate,
                'partner_happiness': float(result['players'][partner]['happiness']) if married else None})
        worlds[policy] = {'p10': focal, 'other_scores': others}
    return {'scenario': c, 'seed': seed, 'worlds': worlds,
            'initial_profiles_sha256': hashlib.sha256(json.dumps(initial[:2]).encode()).hexdigest(),
            'no_monotonic_control_identical': c['monotonic'] == 0}


def baseline_equivalence_smoke():
    available, errors = discover()
    assert not errors
    c = {'roster': 'equal', 'distribution': 'mixed', 'dims': 5, 'monotonic': 2,
         'bonus': 50, 'rounds': 50}
    signatures = []
    for cls in [Player10, CurrentP10]:
        random.seed(SEEDS[0])
        np.random.seed(SEEDS[0])
        codes = roster_codes(c)
        engine = Engine([cls if code == '10' else available[code] for code in codes],
            rounds=c['rounds'], bonus=c['bonus'], dims=5, monotonic=2, seed=SEEDS[0],
            distribution='mixed', timeout=0, keep_records=False)
        result = engine.run()
        assert not result['faults']
        signatures.append(game_signature(result))
    assert signatures[0] == signatures[1], 'Baseline wrapper changes original P10 behavior'


def build_outputs(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    rows, error_rows = [], []
    for record in records:
        c, seed = record['scenario'], record['seed']
        for policy, world in record['worlds'].items():
            people = world['p10']
            h = np.array([p['happiness'] for p in people])
            married = np.array([p['married'] for p in people])
            same = np.array([p['same_strategy_spouse'] for p in people])
            other = np.array(world['other_scores'])
            row = {**c, 'seed': seed, 'policy': policy, 'p10_seats': len(people),
                'positive_rate': float((h > 0).mean()), 'negative_rate': float((h < 0).mean()),
                'married_rate': float(married.mean()), 'mean_happiness': float(h.mean()),
                'self_marriage_rate': float(same.mean()), 'self_negative_rate': float((same & (h < 0)).mean()),
                'outside_negative_rate': float((~same & (h < 0)).mean()),
                'other_mean_happiness': float(other.mean()) if len(other) else np.nan,
                'other_positive_rate': float((other > 0).mean()) if len(other) else np.nan}
            rows.append(row)
            for p in people:
                if p['married']:
                    error_rows.append({**c, 'seed': seed, 'policy': policy,
                        'same_strategy_spouse': p['same_strategy_spouse'],
                        'prediction_error': p['final_estimate'] - p['happiness'],
                        'happiness': p['happiness'], 'estimate': p['final_estimate']})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'per_game_results.csv', index=False)
    errors = pd.DataFrame(error_rows)
    errors.to_csv(OUT / 'marriage_estimation_errors.csv', index=False)
    active = frame[frame.monotonic > 0]
    metrics = ['positive_rate', 'negative_rate', 'married_rate', 'mean_happiness',
               'self_marriage_rate', 'self_negative_rate', 'outside_negative_rate',
               'other_mean_happiness', 'other_positive_rate']
    summary = active.groupby('policy')[metrics].mean().reindex(POLICIES)
    summary.to_csv(OUT / 'active_trait_summary.csv')
    by_roster = active.groupby(['roster', 'policy'])[metrics].mean()
    by_roster.to_csv(OUT / 'by_roster.csv')
    active.groupby(['monotonic', 'policy'])[metrics].mean().to_csv(OUT / 'by_monotonic.csv')
    active.groupby(['distribution', 'policy'])[metrics].mean().to_csv(OUT / 'by_distribution.csv')
    frame[frame.monotonic == 0].groupby('policy')[metrics].mean().to_csv(OUT / 'negative_control_summary.csv')
    assert np.isclose(summary.loc['current', 'negative_rate'],
                      summary.loc['current', 'self_negative_rate'] + summary.loc['current', 'outside_negative_rate'])
    assert active[active.policy != 'current'].self_negative_rate.eq(0).all()
    seed_stats = active.groupby(['seed', 'policy'])[metrics].mean()
    random_indices = np.random.default_rng(18272).integers(0, len(SEEDS), (4000, len(SEEDS)))
    contrasts = {}
    for policy in POLICIES[1:]:
        contrasts[policy] = {}
        for metric in metrics:
            delta = seed_stats.xs(policy, level='policy')[metric].reindex(SEEDS) - seed_stats.xs('current', level='policy')[metric].reindex(SEEDS)
            boots = delta.to_numpy()[random_indices].mean(axis=1)
            contrasts[policy][metric] = {'difference': float(delta.mean()),
                'seed_bootstrap_95_interval': np.quantile(boots, [.025, .975]).tolist()}
    factorial_effects = {}
    effect_weights = {
        'exaggeration_with_trusting_receivers': {'current': 1, 'truthful': -1},
        'exaggeration_with_discounting_receivers': {'discount': 1, 'truthful_discount': -1},
        'incoming_discount_while_exaggerating': {'discount': 1, 'current': -1},
        'incoming_discount_while_truthful': {'truthful_discount': 1, 'truthful': -1},
        'interaction': {'discount': 1, 'truthful_discount': -1, 'current': -1, 'truthful': 1},
    }
    for name, weights in effect_weights.items():
        factorial_effects[name] = {}
        for metric in metrics:
            delta = sum(weight * seed_stats.xs(policy, level='policy')[metric].reindex(SEEDS)
                        for policy, weight in weights.items())
            boot = delta.to_numpy()[random_indices].mean(axis=1)
            factorial_effects[name][metric] = {'difference': float(delta.mean()),
                'seed_bootstrap_95_interval': np.quantile(boot, [.025, .975]).tolist()}
    roster_seed = active.groupby(['roster', 'seed', 'policy'])[metrics].mean()
    roster_contrasts = []
    for roster in ROSTERS:
        delta = (roster_seed.xs((roster, 'truthful'), level=('roster', 'policy'))['positive_rate'].reindex(SEEDS)
                 - roster_seed.xs((roster, 'current'), level=('roster', 'policy'))['positive_rate'].reindex(SEEDS))
        low, high = np.quantile(delta.to_numpy()[random_indices].mean(axis=1), [.025, .975])
        roster_contrasts.append({'roster': roster, 'truthful_minus_current': float(delta.mean()),
                                 'lower_95': float(low), 'upper_95': float(high)})
    findings = {'games': len(records)*4, 'paired_initial_worlds': len(records),
        'active_trait_settings': len(active.scenario_id.unique()),
        'negative_control_settings': len(frame[frame.monotonic == 0].scenario_id.unique()),
        'negative_control_worlds': sum(record['no_monotonic_control_identical'] for record in records),
        'p10_seats_per_policy_active': int(active[active.policy.eq('current')].p10_seats.sum()),
        'final_metrics_active_traits': json.loads(summary.to_json(orient='index')),
        'differences_from_current': contrasts, 'truthful_effect_by_roster': roster_contrasts,
        'factorial_effects': factorial_effects,
        'weighting': 'Equal weight to scenario and seed after computing P10 rates within each run. No-monotonic controls excluded from effect estimates.',
        'scope': 'Five traits; monotonic counts 0,2,5; N=20/22; bonuses 50/100; horizons 50/150; five rosters; four distributions; eight fresh seeds.',
        'interpretation': 'Every P10 in the market receives the policy. This estimates population policy effects, not the incentive for one mutant to lie.',
        'incoming_discount': 'Subtract two from every reported monotonic date trait, clipped at zero. It also discounts honest partners; ads and stated ideals are unchanged.',
        'uncertainty': 'Intervals resample eight paired seed-level means; they are conditional on the finite test grid.'}
    (OUT / 'findings.json').write_text(json.dumps(findings, indent=2) + '\n')

    labels = ['Current\n+2 / trust', 'Truthful\ntrust', '+2 /\ndiscount', 'Truthful /\ndiscount']
    colors = ['#1D4ED8', '#60A5FA', '#0891B2', '#94A3B8']
    ink = '#183249'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'text.color': ink, 'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.spines.bottom': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))
    fig.subplots_adjust(left=.075, right=.985, top=.92, bottom=.105, hspace=.58, wspace=.40)
    x = np.arange(4)
    for ax, metric, title, ylabel, scale in [
        (axes[0, 0], 'positive_rate', 'A  Positive outcomes', 'Initial P10 players (%)', 100),
        (axes[0, 1], 'negative_rate', 'B  Negative outcomes', 'Initial P10 players (%)', 100),
        (axes[0, 2], 'mean_happiness', 'C  Mean happiness', 'Happiness / initial P10 player', 1)]:
        values = scale * summary[metric].to_numpy()
        ax.bar(x, values, width=.66, color=colors)
        limit = max(values.max()*1.2, 1)
        ax.set_ylim(0, limit)
        for i, value in enumerate(values):
            ax.text(i, value + limit*.025, f'{value:.2f}' + ('%' if scale == 100 else ''), ha='center', fontsize=9.5)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_title(title, loc='left', fontsize=12, weight='bold', pad=14)
        ax.set_ylabel(ylabel)

    ax = axes[1, 0]
    d = pd.DataFrame(roster_contrasts)
    values = 100 * d.truthful_minus_current.to_numpy()
    errors95 = np.vstack([values - 100*d.lower_95.to_numpy(), 100*d.upper_95.to_numpy() - values])
    ax.bar(np.arange(5), values, width=.65, color='#60A5FA', yerr=np.maximum(0, errors95),
           capsize=3, error_kw={'elinewidth': .8, 'capthick': .8})
    ax.axhline(0, color=ink, linewidth=.8)
    ax.set_xticks(np.arange(5), ['Equal', 'Equal +\nrandom', 'Half /\nP2', 'Half /\nP9', 'All\nP10'], fontsize=8.5)
    ax.set_title('D  Effect of truthful projection', loc='left', fontsize=12, weight='bold', pad=14)
    ax.set_ylabel('Positive rate minus current (pp)')

    ax = axes[1, 1]
    self_loss = 100*summary.self_negative_rate.to_numpy()
    other_loss = 100*summary.outside_negative_rate.to_numpy()
    ax.bar(x, self_loss, width=.66, color='#1D4ED8', label='Same-strategy spouse')
    ax.bar(x, other_loss, width=.66, bottom=self_loss, color='#CBD5E1', label='Other spouse')
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_title('E  Source of negative outcomes', loc='left', fontsize=12, weight='bold', pad=14)
    ax.set_ylabel('Negative outcomes / initial P10 (%)')
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_ylim(0, (self_loss + other_loss).max()*1.35 + .05)

    ax = axes[1, 2]
    metric = 'other_mean_happiness'
    values, bounds = [], []
    for policy in POLICIES[1:]:
        delta = contrasts[policy][metric]
        value = delta['difference']
        low, high = delta['seed_bootstrap_95_interval']
        values.append(value)
        bounds.append([max(0, value-low), max(0, high-value)])
    ax.bar(np.arange(3), values, width=.62, color=colors[1:], yerr=np.array(bounds).T,
           capsize=3, error_kw={'elinewidth': .8, 'capthick': .8})
    ax.axhline(0, color=ink, linewidth=.8)
    ax.set_xticks(np.arange(3), labels[1:], fontsize=9)
    ax.set_title("F  Other players' happiness", loc='left', fontsize=12, weight='bold', pad=14)
    ax.set_ylabel('Mean happiness change vs current')
    for ax in axes.flat:
        ax.grid(axis='y', color='#E2E8F0', linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    pdf = ROOT / 'output/pdf/player10_projection_ablation.pdf'
    png = ROOT / 'output/pdf/player10_figures/player10_projection_ablation.png'
    fig.savefig(pdf, facecolor='white', metadata={'Title': 'P10 projection and incoming-trust ablation'})
    fig.savefig(png, dpi=220, facecolor='white')
    plt.close(fig)
    print('ACTIVE-TRAIT FINAL METRICS\n' + summary.round(5).to_string(), flush=True)
    print('TRUTHFUL EFFECT BY MARKET\n' + d.round(5).to_string(index=False), flush=True)
    print('FIGURE ' + str(pdf), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = scenarios()
    sources = sorted(list((ROOT/'core').glob('*.py')) + list((ROOT/'models').glob('*.py'))
        + list((ROOT/'players').glob('**/*.py')) + list((ROOT/'players').glob('**/distribution.json')) + [Path(__file__)])
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(ROOT)).encode())
        text = path.read_text()
        digest.update((text.split('\ndef build_outputs', 1)[0] if path == Path(__file__) else text).encode())
    config = {'source_sha256': digest.hexdigest(), 'scenarios': cases, 'seeds': SEEDS,
        'policies': {'current': {'outgoing_bump': 2, 'incoming_date_discount': 0},
            'truthful': {'outgoing_bump': 0, 'incoming_date_discount': 0},
            'discount': {'outgoing_bump': 2, 'incoming_date_discount': 2},
            'truthful_discount': {'outgoing_bump': 0, 'incoming_date_discount': 2}},
        'invariants': 'Original threshold, date selection, advertisement and ideal reporting unchanged. All P10 copies share the policy.',
        'pairing_checks': 'Profiles, ideals, initial advertisements and engine RNG state identical across four policies.'}
    manifest, logfile = OUT/'experiment.json', OUT/'games.jsonl'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == config, 'Experiment inputs changed; use a new output directory'
    else:
        manifest.write_text(json.dumps(config, indent=2) + '\n')
    baseline_equivalence_smoke()
    records = [json.loads(line) for line in logfile.read_text().splitlines()] if logfile.exists() else []
    done = {(r['scenario']['scenario_id'], r['seed']) for r in records}
    assert len(done) == len(records)
    jobs = [(case, seed) for case in cases for seed in SEEDS if (case['scenario_id'], seed) not in done]
    total = len(cases)*len(SEEDS)
    print(f'EXPERIMENT {len(cases)} settings x {len(SEEDS)} seeds x 4 policies; {len(jobs)} paired jobs remaining', flush=True)
    started = time.monotonic()
    with logfile.open('a') as stream, ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(run_pair, job) for job in jobs]
        for i, task in enumerate(as_completed(tasks), 1):
            record = task.result()
            records.append(record)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            if i % 128 == 0 or i == len(jobs):
                print(f'PROGRESS {len(records)}/{total} paired jobs; elapsed {time.monotonic()-started:.1f}s', flush=True)
    assert len(records) == total
    records.sort(key=lambda record: (record['scenario']['scenario_id'], record['seed']))
    build_outputs(records)


if __name__ == '__main__':
    main()
