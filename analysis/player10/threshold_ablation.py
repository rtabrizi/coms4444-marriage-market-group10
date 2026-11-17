"""Paired threshold-policy experiment and cumulative-outcome figure.

Runs new games with current code, separate from the original tournament.
All other P10 behavior and all opponents remain fixed. Resume by rerunning.
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

OUT = ROOT / 'output/analysis/player10_threshold_ablation'
POLICIES = ['quadratic', 'linear', 'fixed', 'zero']
LABELS = {'quadratic': 'Quadratic decay (current)', 'linear': 'Linear decay',
          'fixed': 'Fixed threshold (0.25B)', 'zero': 'Zero threshold'}
ROSTERS = ['equal', 'equal+random', 'half-P2', 'half-P9']
ROSTER_LABELS = ['Equal groups', 'Equal + random', 'Half P10 / P2', 'Half P10 / P9']
SEEDS = list(range(27001, 27009))
_AVAILABLE = None


class LinearP10(Player10):
    def __init__(self, snapshot, ctx):
        super().__init__(snapshot, ctx)
        self.decay_exponent = 1.0


class FixedP10(Player10):
    def __init__(self, snapshot, ctx):
        super().__init__(snapshot, ctx)
        self.decay_exponent = 0.0


class ZeroP10(Player10):
    def __init__(self, snapshot, ctx):
        super().__init__(snapshot, ctx)
        self.threshold_scale = 0.0


CLASSES = {'quadratic': Player10, 'linear': LinearP10, 'fixed': FixedP10, 'zero': ZeroP10}


def scenarios():
    cases = []
    for roster, distribution, bonus, rounds in product(ROSTERS,
            ['mixed', 'uniform', 'u_shape', 'right_skew'], [50, 100], [50, 150]):
        cases.append(dict(scenario_id=len(cases), roster=roster, distribution=distribution,
                          bonus=bonus, rounds=rounds, dims=5, monotonic=2))
    return cases


def run_pair(job):
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE, errors = discover()
        if errors:
            raise RuntimeError(str(errors))
    c, seed = job
    if c['roster'].startswith('equal'):
        codes = [str(g) for g in range(1, 11) for _ in range(2)]
        if c['roster'] == 'equal+random':
            codes += ['r', 'r']
    else:
        opponent = '2' if c['roster'] == 'half-P2' else '9'
        codes = [opponent] * 10 + ['10'] * 10
    if c['distribution'] in ['u_shape', 'right_skew']:
        group = '9' if c['distribution'] == 'u_shape' else '2'
        row = json.loads((ROOT / f'players/player_{group}/distribution.json').read_text())['P']
        distribution = [row] * c['dims']
    else:
        distribution = c['distribution']
    initial = None
    worlds = {}
    for policy in POLICIES:
        random.seed(seed)
        np.random.seed(seed)
        classes = [CLASSES[policy] if code == '10' else _AVAILABLE[code] for code in codes]
        engine = Engine(classes, rounds=c['rounds'], bonus=c['bonus'], dims=c['dims'],
            monotonic=c['monotonic'], seed=seed, distribution=distribution,
            timeout=0, keep_records=False)
        state = (engine.profiles, engine.ideals, engine.ads, engine.rng.getstate())
        if initial is None:
            initial = state
        else:
            assert state == initial, 'Paired policies have different initial worlds'
        result = engine.run()
        if result['faults']:
            raise RuntimeError(f'Player faults in scenario {c}, seed {seed}, policy {policy}: {result["faults"]}')
        focal = []
        for i, row in enumerate(result['players']):
            if codes[i] == '10':
                focal.append({'seat': i, **{key: row[key] for key in ['happiness', 'married', 'married_on']}})
        assert len(focal) == codes.count('10')
        worlds[policy] = focal
    return {'scenario': c, 'seed': seed, 'people': len(codes), 'worlds': worlds,
            'initial_profiles_sha256': hashlib.sha256(json.dumps(initial[:2]).encode()).hexdigest()}


def build_outputs(records, include_mean_control=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    policies = POLICIES + (['mean_fixed'] if include_mean_control else [])
    labels_by_policy = {**LABELS, 'mean_fixed': 'Fixed time-average threshold'}
    x = np.linspace(0, 1, 101)
    curves = {p: {m: [] for m in ['positive', 'negative', 'married', 'happiness']} for p in policies}
    finals = []
    for record in records:
        c, seed = record['scenario'], record['seed']
        for policy in policies:
            people = record['worlds'][policy]
            scores = np.array([person['happiness'] for person in people], dtype=float)
            married = np.array([person['married'] for person in people], dtype=bool)
            times = np.array([person['married_on'] / c['rounds'] if person['married_on'] is not None else np.inf for person in people])
            happened = times[:, None] <= x[None, :]
            events = {'positive': (happened & (scores[:, None] > 0)).mean(axis=0),
                      'negative': (happened & (scores[:, None] < 0)).mean(axis=0),
                      'married': happened.mean(axis=0),
                      'happiness': (happened * scores[:, None]).mean(axis=0)}
            for metric, value in events.items():
                assert np.isclose(value[-1], {'positive': (scores > 0).mean(),
                    'negative': (scores < 0).mean(), 'married': married.mean(),
                    'happiness': scores.mean()}[metric])
                curves[policy][metric].append(value)
            finals.append({**c, 'seed': seed, 'policy': policy, 'p10_seats': len(people),
                'positive_rate': float((scores > 0).mean()), 'negative_rate': float((scores < 0).mean()),
                'married_rate': float(married.mean()), 'mean_happiness': float(scores.mean())})
    frame = pd.DataFrame(finals)
    frame.to_csv(OUT / 'per_game_results.csv', index=False)
    metrics = ['positive_rate', 'negative_rate', 'married_rate', 'mean_happiness']
    summary = frame.groupby('policy')[metrics].mean().reindex(policies)
    roster_summary = frame.groupby(['roster', 'policy'])[metrics].mean()
    roster_summary.to_csv(OUT / 'results_by_roster.csv')
    frame.groupby(['distribution', 'policy'])[metrics].mean().to_csv(OUT / 'results_by_distribution.csv')
    averaged = {p: {m: np.mean(values, axis=0).tolist() for m, values in measures.items()} for p, measures in curves.items()}
    (OUT / 'curves.json').write_text(json.dumps({'fraction_rounds_elapsed': x.tolist(), 'policies': averaged}, indent=2) + '\n')
    seed_stats = frame.groupby(['seed', 'policy'])[metrics].mean()
    rng = np.random.default_rng(88201)
    resamples = rng.integers(0, len(SEEDS), size=(4000, len(SEEDS)))
    gaps = {}
    for policy in policies[1:]:
        gaps[policy] = {}
        for metric in metrics:
            delta = seed_stats.xs(policy, level='policy')[metric].reindex(SEEDS) - seed_stats.xs('quadratic', level='policy')[metric].reindex(SEEDS)
            boots = delta.to_numpy()[resamples].mean(axis=1)
            gaps[policy][metric] = {'mean_difference': float(delta.mean()),
                'seed_bootstrap_95_interval': np.quantile(boots, [.025, .975]).tolist()}
    (OUT / 'summary.json').write_text(json.dumps({'paired_initial_worlds': len(records),
        'games': len(records) * len(policies), 'p10_seats_per_policy': int(frame[frame.policy.eq('quadratic')].p10_seats.sum()),
        'weighting': 'Each scenario and seed receives equal weight; means within P10 seats are calculated first.',
        'final_metrics': json.loads(summary.to_json(orient='index')), 'differences_from_current': gaps,
        'scope': 'Current code; N=20 or 22, d=5, m=2, B=50/100, T=50/150, four rosters and four distributions, eight fresh seeds.',
        'uncertainty': 'Intervals resample eight seed-level paired mean differences; they do not cover untested scenarios or distributions.'}, indent=2) + '\n')

    ink = '#183249'
    colors = {'quadratic': '#1D4ED8', 'linear': '#60A5FA', 'fixed': '#64748B', 'zero': '#0891B2', 'mean_fixed': '#7C3AED'}
    styles = {'quadratic': '-', 'linear': '--', 'fixed': ':', 'zero': '-.', 'mean_fixed': (0, (5, 2, 1, 2))}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': ink,
        'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.7))
    fig.subplots_adjust(left=.075, right=.985, top=.84 if include_mean_control else .87, bottom=.105, hspace=.48, wspace=.38)
    for policy in policies:
        remaining = np.minimum(1, (101 - np.arange(101)) / 100)
        exponent = {'quadratic': 2, 'linear': 1, 'fixed': 0, 'zero': 2, 'mean_fixed': 0}[policy]
        scale = 0 if policy == 'zero' else .25 * 101 * 201 / (6 * 100 ** 2) if policy == 'mean_fixed' else .25
        threshold = scale * remaining ** exponent
        axes[0, 0].plot(100*x, threshold, color=colors[policy], linestyle=styles[policy], linewidth=2, label=labels_by_policy[policy])
    axes[0, 0].set_title('A  Threshold schedules', loc='left', weight='bold', fontsize=11.5, pad=12)
    axes[0, 0].set_ylabel('Threshold / marriage bonus')
    axes[0, 0].set_ylim(-.01, .29)
    specs = [(axes[0, 1], 'positive', 'B  Cumulative positive outcomes', 'Initial P10 players (%)'),
             (axes[0, 2], 'negative', 'C  Cumulative negative outcomes', 'Initial P10 players (%)'),
             (axes[1, 0], 'married', 'D  Cumulative marriages', 'Initial P10 players (%)'),
             (axes[1, 1], 'happiness', 'E  Cumulative happiness', 'Happiness / initial P10 player')]
    for ax, metric, title, ylabel in specs:
        for policy in policies:
            values = np.array(averaged[policy][metric]) * (1 if metric == 'happiness' else 100)
            ax.step(100*x, values, where='post', color=colors[policy], linestyle=styles[policy], linewidth=1.9)
        ax.set_title(title, loc='left', weight='bold', fontsize=11.5, pad=12)
        ax.set_ylabel(ylabel)
        if metric != 'happiness':
            ax.set_ylim(bottom=0)
    ax = axes[1, 2]
    positions = np.arange(4)
    roster_seed = frame.groupby(['roster', 'seed', 'policy']).positive_rate.mean()
    roster_intervals = []
    alternatives = policies[1:]
    width = .8 / len(alternatives)
    for i, policy in enumerate(alternatives):
        differences = [100 * (roster_summary.loc[(roster, policy), 'positive_rate'] - roster_summary.loc[(roster, 'quadratic'), 'positive_rate']) for roster in ROSTERS]
        errors = []
        for roster, value in zip(ROSTERS, differences):
            delta = (roster_seed.loc[(roster, slice(None), policy)].reindex(SEEDS)
                     - roster_seed.loc[(roster, slice(None), 'quadratic')].reindex(SEEDS)).to_numpy()
            lower, upper = 100 * np.quantile(delta[resamples].mean(axis=1), [.025, .975])
            errors.append([max(0, value - lower), max(0, upper - value)])
            roster_intervals.append({'roster': roster, 'policy': policy, 'difference_pp': value,
                                     'lower_95_pp': lower, 'upper_95_pp': upper})
        ax.bar(positions + (i - (len(alternatives)-1)/2)*width, differences, width=width*.9,
               color=colors[policy], yerr=np.array(errors).T, capsize=2,
               error_kw={'elinewidth': .7, 'capthick': .7}, label=labels_by_policy[policy])
    (OUT / 'roster_gap_intervals.json').write_text(json.dumps(roster_intervals, indent=2) + '\n')
    ax.axhline(0, color=ink, linewidth=.8)
    ax.set_xticks(positions, ['Equal', 'Equal +\nrandom', 'Half /\nP2', 'Half /\nP9'], fontsize=9)
    ax.set_ylabel('Positive rate minus current (pp)')
    ax.set_title('F  Final positive-rate change', loc='left', weight='bold', fontsize=11.5, pad=12)
    for ax in axes.flat:
        ax.grid(axis='y', color='#E2E8F0', linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    for ax in [axes[0, 0]] + [spec[0] for spec in specs]:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Rounds elapsed (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .985), ncol=3 if include_mean_control else 4,
               frameon=False, fontsize=10, columnspacing=1.5)
    pdf = ROOT / 'output/pdf/player10_threshold_ablation.pdf'
    png = ROOT / 'output/pdf/player10_figures/player10_threshold_ablation.png'
    fig.savefig(pdf, facecolor='white', metadata={'Title': 'Player 10 proposal-threshold ablation'})
    fig.savefig(png, dpi=220, facecolor='white')
    plt.close(fig)
    print('FINAL METRICS\n' + summary.round(5).to_string(), flush=True)
    print('FIGURE ' + str(pdf), flush=True)


class MeanThresholdP10(Player10):
    def __init__(self, snapshot, ctx):
        super().__init__(snapshot, ctx)
        self.threshold_scale = .25 * (self.rounds + 1) * (2*self.rounds + 1) / (6*self.rounds**2)
        self.decay_exponent = 0.0


def run_mean_control(record):
    available, errors = discover()
    assert not errors
    c, seed = record['scenario'], record['seed']
    if c['roster'].startswith('equal'):
        codes = [str(g) for g in range(1, 11) for _ in range(2)]
        if c['roster'] == 'equal+random':
            codes += ['r', 'r']
    else:
        codes = ['2' if c['roster'] == 'half-P2' else '9']*10 + ['10']*10
    if c['distribution'] in ['u_shape', 'right_skew']:
        group = '9' if c['distribution'] == 'u_shape' else '2'
        row = json.loads((ROOT / f'players/player_{group}/distribution.json').read_text())['P']
        distribution = [row] * c['dims']
    else:
        distribution = c['distribution']
    initial = None
    for cls in [Player10, MeanThresholdP10]:
        random.seed(seed)
        np.random.seed(seed)
        engine = Engine([cls if code == '10' else available[code] for code in codes],
            rounds=c['rounds'], bonus=c['bonus'], dims=c['dims'], monotonic=c['monotonic'],
            seed=seed, distribution=distribution, timeout=0, keep_records=False)
        state = (engine.profiles, engine.ideals, engine.ads, engine.rng.getstate())
        if initial is None:
            initial = state
        else:
            assert state == initial
    assert hashlib.sha256(json.dumps(initial[:2]).encode()).hexdigest() == record['initial_profiles_sha256']
    result = engine.run()
    assert not result['faults']
    focal = [{'seat': i, **{key: row[key] for key in ['happiness', 'married', 'married_on']}}
             for i, row in enumerate(result['players']) if codes[i] == '10']
    return {'scenario_id': c['scenario_id'], 'seed': seed, 'world': focal,
            'initial_profiles_sha256': record['initial_profiles_sha256']}


def add_mean_control(records):
    path = OUT / 'mean_control_games.jsonl'
    metadata_path = OUT / 'mean_control_experiment.json'
    text = Path(__file__).read_text()
    source = text.split('\nclass MeanThresholdP10', 1)[1].split('\ndef main', 1)[0]
    metadata = {'base_experiment': json.loads((OUT / 'experiment.json').read_text()),
        'additional_policy': 'Fixed threshold equal to the exact discrete time-average of the quadratic schedule.',
        'scale': '0.25 * (T+1) * (2T+1) / (6T^2)',
        'control_source_sha256': hashlib.sha256(source.encode()).hexdigest()}
    if metadata_path.exists():
        assert json.loads(metadata_path.read_text()) == metadata
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    controls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(row['scenario_id'], row['seed']) for row in controls}
    jobs = [r for r in records if (r['scenario']['scenario_id'], r['seed']) not in done]
    print(f'MEAN CONTROL {len(jobs)} additional games; all previous simulation results reused', flush=True)
    with path.open('a') as stream, ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(run_mean_control, record) for record in jobs]
        for i, task in enumerate(as_completed(tasks), 1):
            record = task.result()
            controls.append(record)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            if i % 128 == 0 or i == len(jobs):
                print(f'MEAN CONTROL PROGRESS {len(controls)}/{len(records)}', flush=True)
    lookup = {(row['scenario_id'], row['seed']): row for row in controls}
    assert len(lookup) == len(records)
    for record in records:
        control = lookup[record['scenario']['scenario_id'], record['seed']]
        assert control['initial_profiles_sha256'] == record['initial_profiles_sha256']
        record['worlds']['mean_fixed'] = control['world']


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean-control', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = scenarios()
    sources = sorted(list((ROOT / 'core').glob('*.py')) + list((ROOT / 'models').glob('*.py'))
        + list((ROOT / 'players').glob('**/*.py')) + list((ROOT / 'players').glob('**/distribution.json')) + [Path(__file__)])
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(ROOT)).encode())
        if path == Path(__file__):
            # Plot-only edits do not invalidate completed simulation records.
            digest.update(path.read_text().split('\ndef build_outputs', 1)[0].encode())
        else:
            digest.update(path.read_bytes())
    experiment = {'source_sha256': digest.hexdigest(), 'scenarios': cases, 'seeds': SEEDS,
        'policies': {'quadratic': {'threshold_scale': .25, 'decay_exponent': 2},
                     'linear': {'threshold_scale': .25, 'decay_exponent': 1},
                     'fixed': {'threshold_scale': .25, 'decay_exponent': 0},
                     'zero': {'threshold_scale': 0, 'decay_exponent': 2}},
        'paired_initial_state_assertion': 'Profiles, ideals, advertisements and engine RNG state match exactly across all four policies.'}
    manifest = OUT / 'experiment.json'
    logfile = OUT / 'games.jsonl'
    if manifest.exists():
        if json.loads(manifest.read_text()) != experiment:
            if logfile.exists() and logfile.stat().st_size:
                raise RuntimeError('Experiment inputs changed; preserve existing output and choose a new experiment directory.')
            manifest.write_text(json.dumps(experiment, indent=2) + '\n')
    else:
        manifest.write_text(json.dumps(experiment, indent=2) + '\n')
    records = [json.loads(line) for line in logfile.read_text().splitlines()] if logfile.exists() else []
    done = {(record['scenario']['scenario_id'], record['seed']) for record in records}
    assert len(done) == len(records)
    jobs = [(case, seed) for case in cases for seed in SEEDS if (case['scenario_id'], seed) not in done]
    print(f'EXPERIMENT {len(cases)} scenarios x {len(SEEDS)} seeds x 4 policies; {len(jobs)} paired jobs remaining', flush=True)
    started = time.monotonic()
    with logfile.open('a') as output, ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(run_pair, job) for job in jobs]
        for i, task in enumerate(as_completed(tasks), 1):
            record = task.result()
            records.append(record)
            output.write(json.dumps(record) + '\n')
            output.flush()
            if i % 32 == 0 or i == len(jobs):
                print(f'PROGRESS {len(records)}/{len(cases)*len(SEEDS)} paired jobs; elapsed {time.monotonic()-started:.1f}s', flush=True)
    assert len(records) == len(cases) * len(SEEDS)
    records.sort(key=lambda record: (record['scenario']['scenario_id'], record['seed']))
    if args.mean_control:
        add_mean_control(records)
    build_outputs(records, include_mean_control=args.mean_control)


if __name__ == '__main__':
    main()
