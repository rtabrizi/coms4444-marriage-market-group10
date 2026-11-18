"""Compare the two quadratic threshold clocks on the threshold ablation's paired markets."""

from concurrent.futures import ProcessPoolExecutor
import json

import threshold_ablation as experiment


class SinglesP10(experiment.Player10):
    def __init__(self, snapshot, ctx):
        super().__init__(snapshot, ctx)
        self.decay_by_singles = True


def run(record):
    # run_pair resets all RNGs and checks the starting state between policies.
    experiment.POLICIES = ['quadratic', 'singles']
    experiment.CLASSES['singles'] = SinglesP10
    result = experiment.run_pair((record['scenario'], record['seed']))
    assert result['initial_profiles_sha256'] == record['initial_profiles_sha256']
    assert result['worlds']['quadratic'] == record['worlds']['quadratic']
    return result


def main():
    import numpy as np

    records = [json.loads(line) for line in (experiment.OUT / 'games.jsonl').read_text().splitlines()]
    keys = {(row['scenario']['scenario_id'], row['seed']) for row in records}
    assert len(keys) == len(records) and keys == {
        (case['scenario_id'], seed) for case in experiment.scenarios() for seed in experiment.SEEDS}
    records.sort(key=lambda r: (r['seed'], r['scenario']['scenario_id']))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, records, chunksize=2))
    (experiment.OUT / 'singles_games.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in results))
    rates = []
    for result in results:
        pair = []
        for policy in ['quadratic', 'singles']:
            players = result['worlds'][policy]
            scores = np.array([p['happiness'] for p in players])
            pair.append([(scores > 0).mean(), np.mean([p['married'] for p in players]),
                         (scores < 0).mean(), scores.mean()])
        rates.append(pair)
    rates = np.array(rates)
    averages = rates.mean(axis=0)
    seed_gaps = (rates[:, 1] - rates[:, 0]).reshape(len(experiment.SEEDS), -1, 4).mean(axis=1)
    draws = np.random.default_rng(27001).integers(len(seed_gaps), size=(10000, len(seed_gaps)))
    bounds = np.quantile(seed_gaps[draws].mean(axis=1), [.025, .975], axis=0)
    summary = {metric: dict(rounds=averages[0, i], singles=averages[1, i],
                           paired_difference=seed_gaps[:, i].mean(), seed_bootstrap_95=bounds[:, i].tolist())
               for i, metric in enumerate(['positive', 'married', 'negative', 'mean_happiness'])}
    report = json.dumps(summary, indent=2)
    (experiment.OUT / 'singles_summary.json').write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
