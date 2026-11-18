# Reproducing the figures in sections 3 and 4

Run the commands below from the repository root with Python 3.12 and NumPy, pandas, Matplotlib, and ReportLab installed. Each figure heading links to the PDF used in the paper. The plotting scripts write to `output/pdf/`; Overleaf uses a copy of each PDF with the same filename at the project root.

## Section 3 · Tournament results

These figures summarize the original tournament. A *seat* is one player in one game. A positive outcome means that player's final happiness was strictly positive. A single player contributes zero happiness and remains in the denominator when computing rates over all seats.

### Figure 6 · [Overall rankings by market](output/pdf/player10_market_rankings_combined.pdf)

[build_market_figure.py](analysis/player10/build_market_figure.py) reads the saved [positive-outcome rankings](output/pdf/player10_tables/market_rankings.csv) and [mean-happiness rankings](output/pdf/player10_tables/market_rankings_mean_happiness.csv). It checks the ranks against the outcome values, then draws the two tables with ReportLab. Each column ranks the ten submitted strategies within a market type. The overall column pools seats, so a market with more recorded players contributes more weight. Random players can appear as opponents but do not receive a rank.

```sh
python analysis/player10/build_market_figure.py
```

### Figure 7 · [Equal groups versus half-and-half](output/pdf/player10_equal_vs_half_half.pdf)

[build_equal_half_figure.py](analysis/player10/build_equal_half_figure.py) pools each strategy's seats separately for equal-group and half-and-half markets. Panels A–D show Player 10's positive-outcome rate, marriage rate, negative-outcome rate, and mean happiness, with its rank among the submitted strategies. Panel E uses **married seats** as its denominator and identifies same-strategy spouses from partner IDs and roster counts. Panel F compares positive-outcome rates on the shared tournament settings, giving each setting and seed equal weight before grouping by the number of monotonic traits. This weighting explains why panel F cannot be reproduced by simply splitting the pooled bars in panel A.

The renderer needs [all_settings.csv](output/analysis/player10_market_mechanisms/all_settings.csv) and three local caches in `/private/tmp/` named `p10_runs.pkl`, `p10_seats.pkl`, and `p10_equal_seats.pkl`. With those available, run:

```sh
python analysis/player10/build_equal_half_figure.py
```

To rebuild the caches on another machine, place the original `tuned_p10_tournament_results.zip` at the repository root and ensure `/private/tmp/` exists and is writable. The ZIP is absent from this checkout. [build_report.py](analysis/player10/build_report.py) streams the raw seat records and writes the caches. [analyze_market_mechanisms.py](analysis/player10/analyze_market_mechanisms.py) produces the cohort table and checks the cached counts. Run these before the renderer when rebuilding from the archive:

```sh
python analysis/player10/build_report.py
python analysis/player10/analyze_market_mechanisms.py
```

### Figure 8 · [Outcomes by trait distribution](output/pdf/player10_distribution_results.pdf)

[analyze_distribution_competitors.py](analysis/player10/analyze_distribution_competitors.py) reads the saved [tournament totals by distribution and strategy](output/pdf/player10_tables/full_tournament_distribution.csv) and writes [the rates and ranks used in the plot](output/analysis/player10_distribution_finding/competitor_outcome_components.csv). [build_distribution_figure.py](analysis/player10/build_distribution_figure.py) then plots Player 10's results across all seven distributions. Panel A ranks positive-outcome rates, with mean happiness breaking ties. Panel B ranks marriage rates. Panel C counts zero- and negative-happiness marriages and divides by **all starting seats**. Lower is better in panel C. With these denominators, panel B minus panel C equals panel A within each distribution.

```sh
python analysis/player10/analyze_distribution_competitors.py
python analysis/player10/build_distribution_figure.py
```

## Section 4 · Paired ablations

The ablations use 512 starting markets from four rosters, four trait distributions, two marriage bonuses, two game lengths, and eight seeds. Each policy is applied to every Player 10 seat in a market. Opponent strategies stay fixed. The scripts compute rates within each game and then average games equally. For cumulative curves, a player enters the curve in their marriage round, expressed as a fraction of scheduled rounds. A positive-outcome curve counts only marriages yielding positive happiness for that player. The [inset helper](analysis/player10/late_round_inset.py) enlarges the final quarter of the same curves.

### Figure 9 · [Threshold schedules and outcomes](output/pdf/player10_threshold_ablation.pdf)

[threshold_ablation.py](analysis/player10/threshold_ablation.py) runs the paired policies, aggregates game outcomes, and draws the figure. Use the flag below to include the fixed threshold equal to the time average of the quadratic schedule. The script reuses completed [base games](output/analysis/player10_threshold_ablation/games.jsonl) and [time-average control games](output/analysis/player10_threshold_ablation/mean_control_games.jsonl) and runs missing cases. Panel A shows the threshold schedules. Panels B–D show cumulative positive outcomes, negative outcomes, and marriages. The experiment also records a zero-threshold policy, which is discussed in the text; the figure shows the positive-threshold policies.

```sh
python analysis/player10/threshold_ablation.py --mean-control
```

The singles-based decay comparison in section 4.1 is a separate, text-only result produced by [singles_ablation.py](analysis/player10/singles_ablation.py).

### Figure 10 · [Lying on monotonic traits](output/pdf/player10_lying_ablation.pdf)

[lying_exploration_ablation.py](analysis/player10/lying_exploration_ablation.py) replays the threshold experiment's starting markets and checks that the final policy reproduces each archived Player 10 outcome. For the lying comparison, it changes the outgoing profile on dates by adding zero, two, or five points to each monotonic trait, capped at ten. [plot_lying_exploration.py](analysis/player10/plot_lying_exploration.py) subtracts each game's truthful-reporting rate from its exaggerated-reporting rate and plots the average difference in percentage points. The outcomes are receiving at least one proposal, marrying, marrying with positive happiness, and marrying with non-positive happiness.

### Figure 11 · [Exploration period](output/pdf/player10_exploration_ablation.pdf)

The same experiment and plotter compare exploration periods of 0%, 10%, 25%, and 50% of scheduled rounds. During exploration the player lists eligible candidates in market order and can still request repeat dates and propose. Panels A and C show the cumulative positive-outcome and marriage rates. Panels B and D show the final percentages, and the plotter checks that each bar matches the endpoint of its curve. The separate policy that also defers commitment is present in the saved games and excluded from this duration sweep.

**To redraw both Figures 10 and 11** from the [saved paired games](output/analysis/player10_lying_exploration/games.jsonl), run:

```sh
python analysis/player10/plot_lying_exploration.py
```

To rerun those simulations, run the following commands after generating the base games for Figure 9. The simulation command rewrites the saved lying and exploration games.

```sh
python analysis/player10/lying_exploration_ablation.py
python analysis/player10/plot_lying_exploration.py
```
