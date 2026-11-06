# ANALYZOL V8.0.0-RESEARCH

This is a reproducible statistical foundation, **not a validated betting model**.

## Included
- Transparent team attack/defence Poisson goal model with home advantage.
- Dixon-Coles low-score correction.
- Fixed-seed Monte Carlo simulation from the fitted score distribution.
- Expanding-window chronological walk-forward evaluation.
- Separate final chronological 20% holdout.
- Accuracy, multiclass Brier (sum and half-scaled conventions), log loss and goals MAE.
- Excel templates for match data, immutable predictions, feature timestamps, results and audit controls.

## Required CSV columns
`date,home_team,away_team,home_goals,away_goals`

One row per completed match, sorted automatically by date. Use a single competition first. Do not mix competitions without modelling league effects.

## Run
Install Python 3.10+ and dependencies:
`pip install pandas numpy scipy`

Run:
`python analyz_ol_v8.py --data matches.csv --out output --simulations 50000 --seed 20261105`

## Important limitations
- No historical match dataset or executable V7.1 engine was provided; this package therefore makes no claim that V8.0 beats V7.1 or any market.
- The initial team-strength model is static, not time-decayed.
- Promoted teams, season changes, league effects and sparse team histories require additional handling.
- The walk-forward and holdout reports are preliminary; add baseline predictions on exactly the same test rows before interpreting performance.
- Player, tactical, fatigue, motivation and referee features are intentionally not included yet. Add them one at a time with timestamped data and ablation tests.
- The Monte Carlo simulation samples from the fitted score distribution. It does not create extra predictive information or improve the underlying model.
- Verify optimizer convergence, probability calibration, data provenance and paired uncertainty intervals before promotion.
- The included sample CSV is a format example only, not training data.

## Release rule
Keep this build labelled `V8.0.0-RESEARCH`. Do not call it statistically validated until it passes data-integrity, leakage, baseline, calibration, uncertainty and prospective blind-test gates. V2–V7.1 historical predictions must remain unchanged.
