# ANALYZOL V8.1 development branch

## Branch protection boundary
- This is development branch `2`.
- The frozen V8.0 baseline on `main` is not modified.
- No changes are made to V8.0 parameters, training data, or feature set in this commit.

## Implemented
1. **Corrected 1X2 probability labels.** The score matrix stores home goals in rows and away goals in columns. Therefore, home wins are the lower triangle, draws are the diagonal, and away wins are the upper triangle. The prior V8.0 implementation assigned the two win probabilities in reverse.
2. **Added regression tests** for asymmetric score matrices so a future change cannot silently reverse home and away probabilities.

## Not yet implemented
- Probability calibration (for example, multiclass calibration): requires fitting on a calibration set that is separate from the final chronological holdout, followed by a fresh untouched evaluation.
- Any changes to Dixon-Coles parameters, recency weighting, team/player/tactical features, injuries, fatigue, motivation, referee effects, market blending, or betting thresholds. These require independently sourced inputs and walk-forward validation before integration.
- Betting strategy changes. The five-season audit results are not sufficient evidence to approve a profitable strategy.

## Required validation before calling this release stable
- Run the regression tests in the target Python environment.
- Re-run the five-season chronological walk-forward and final-season holdout with corrected labels.
- Report accuracy, multiclass Brier score (both conventions), log loss, goal MAE, and calibration diagnostics.
- Compare against simple baselines and closing-market probabilities.
- Verify all odds coverage and betting simulation calculations independently.
- Do not tune on the final holdout.

## Dataset audit scope
The current reduced audit scope is the five most recent completed EPL seasons, 2021/22 through 2025/26 (1,900 fixtures). The independent score-by-score cross-check for 2025/26 and final reconciliation of the audit metrics remain outstanding.

## Versioning
This branch contains an experimental correction based on V8.0. It is not a production release and must not be merged into `main` without a separate review and explicit approval.
