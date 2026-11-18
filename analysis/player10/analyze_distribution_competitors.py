"""Compare distribution outcomes and ranks using saved tournament totals.

No new games are simulated. All metrics use the same seats as the overview.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/analysis/player10_distribution_finding'
AGGREGATES = ROOT / 'output/pdf/player10_tables/full_tournament_distribution.csv'
COMPARE = ['u_shape', 'right_skew', 'uniform', 'normal', 'mixed']


def main():
    outcomes = pd.read_csv(AGGREGATES, dtype={'group': str})
    assert outcomes.groupby('distribution').group.nunique().eq(10).all()
    outcomes['nonpositive_married'] = outcomes.married - outcomes.positive
    assert outcomes.nonpositive_married.eq(outcomes.negative + outcomes.zero_marriage).all()
    outcomes['nonpositive_married_rate'] = outcomes.nonpositive_married / outcomes.seats
    for metric, ascending in [('married_rate', False), ('nonpositive_married_rate', True)]:
        outcomes[metric + '_rank'] = outcomes.groupby('distribution')[metric].rank(
            ascending=ascending, method='min').astype(int)
    outcomes = outcomes.sort_values(['distribution', 'positive_rate', 'mean_happiness'],
                                    ascending=[True, False, False])
    outcomes['positive_outcome_rank'] = outcomes.groupby('distribution').cumcount() + 1
    outcomes['rank_positions_lost'] = outcomes.positive_outcome_rank - outcomes.married_rate_rank
    assert ((outcomes.married_rate - outcomes.nonpositive_married_rate
             - outcomes.positive_rate).abs() < 1e-12).all()
    outcomes.to_csv(OUT / 'competitor_outcome_components.csv', index=False)
    focal = outcomes[outcomes.group.eq('10')].set_index('distribution').loc[COMPARE]
    assert focal.positive_outcome_rank.tolist() == [1, 2, 7, 8, 7]
    assert focal.nonpositive_married_rate_rank.tolist() == [9, 9, 10, 10, 10]
    assert focal.rank_positions_lost.tolist() == [-1, 1, 6, 7, 5]
    provenance = {
        'games_simulated': 0,
        'aggregate_sha256': hashlib.sha256(AGGREGATES.read_bytes()).hexdigest(),
        'comparison_distributions': COMPARE,
        'denominator': 'All recorded players of each strategy and distribution, including singles.',
        'positive_rank': 'Rank by positive-outcome rate; mean happiness breaks ties.',
        'other_ranks': 'Within each distribution; ties share minimum rank; lower nonpositive-marriage rate is better.',
        'rank_positions_lost': 'Positive-outcome rank minus marriage-frequency rank; negative values indicate an improvement.',
        'metric_selection': {
            'retained': 'Nonpositive-marriage frequency and the associated change between marriage-frequency and positive-outcome rankings.',
            'rejected': 'Marriage frequency alone and early happy-marriage frequency: high ranks also occur in weaker-ranking distributions.',
        },
        'limitations': [
            'Recorded samples have uneven opponent coverage and roster weights.',
            'Rank change compares outcome definitions, not an intervention that removes bad marriages.',
            'Nonpositive-marriage rank is ninth in the stronger distributions and tenth in the weaker ones; this is a reduced weakness, not a leading safety rank.',
            'Rank change is an accounting explanation derived from the outcome rankings, not independent causal evidence about the policy.',
        ],
    }
    (OUT / 'competitor_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(focal[['positive_outcome_rank', 'nonpositive_married_rate',
                 'nonpositive_married_rate_rank', 'married_rate_rank', 'rank_positions_lost']].to_string())


if __name__ == '__main__':
    main()
