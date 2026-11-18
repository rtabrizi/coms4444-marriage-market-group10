"""Paired ablations of outgoing profile projection and opening date selection."""
from concurrent.futures import ProcessPoolExecutor
import json
import random

import threshold_ablation as base
from projection_ablation import roster_codes, distribution_for
from models.player import Decision

import numpy as np

OUT = base.ROOT / 'output/analysis/player10_lying_exploration'
# (outgoing monotonic bump, fraction using opening lists, defer commitment)
POLICIES = {'final': (2, 0, False), 'honest': (0, 0, False), 'plus5': (5, 0, False),
            'open10': (2, .1, False), 'open25': (2, .25, False), 'open50': (2, .5, False),
            'defer25': (2, .25, True)}
AVAILABLE = None


class MeasuredPlayer(base.Player10):
    def decide(self, date, observed, stated_ideal):
        decision = super().decide(date, observed, stated_ideal)
        if self.defer and self.is_collect_data(date):
            return Decision(incompatible=decision.incompatible)
        return decision


def run(record):
    global AVAILABLE
    if AVAILABLE is None:
        AVAILABLE, errors = base.discover()
        assert not errors, errors
    c, seed = record['scenario'], record['seed']
    codes = roster_codes(c)
    seats = [i for i, code in enumerate(codes) if code == '10']
    result = {}
    initial = None
    for name, (bump, duration, defer) in POLICIES.items():
        random.seed(seed)
        np.random.seed(seed)
        engine = base.Engine([MeasuredPlayer if code == '10' else AVAILABLE[code] for code in codes],
                             rounds=c['rounds'], bonus=c['bonus'], dims=c['dims'], monotonic=c['monotonic'],
                             seed=seed, distribution=distribution_for(c), timeout=0, keep_records=False)
        state = (engine.profiles, engine.ideals, engine.ads, engine.rng.getstate())
        if initial is None:
            initial = state
        assert state == initial
        for seat in seats:
            player = engine.players[seat]
            player.monotonic_exaggeration = bump
            player.exploration_duration = duration
            player.defer = defer
        game = engine.run()
        assert not game['faults'], game['faults']
        players = [game['players'][i] for i in seats]
        if name == 'final':
            for player, archived in zip(players, record['worlds']['quadratic'], strict=True):
                assert all(player[k] == archived[k] for k in ['happiness', 'married', 'married_on'])
        result[name] = [{k: p[k] for k in ('happiness', 'married_on', 'proposals_received')} for p in players]
    return {'scenario': c, 'seed': seed, 'worlds': result}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (base.OUT / 'games.jsonl').read_text().splitlines()]
    keys = {(r['scenario']['scenario_id'], r['seed']) for r in records}
    assert len(keys) == len(records) and keys == {(c['scenario_id'], s) for c in base.scenarios() for s in base.SEEDS}
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, records, chunksize=2))
    (OUT / 'games.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in results))
