# Test fixtures

`sample_oracleselixir.csv` and `sample_soloqueue.json` are **synthetic, hand-generated fixture data** for automated tests and offline pipeline demos only — they do not represent real match history, real player performance, or real solo-queue statistics, even though they use real champion/team/player names for realism.

`sample_oracleselixir.csv` currently contains ~78 synthetic games spanning 4 distinct patches (14.1-14.4) with dates spread across roughly four months, so that patch-recency filtering (`PipelineConfig.num_recent_patches`) and the walk-forward backtest (`compstrength_pipeline/backtest.py`) both have enough data/date spread to exercise multiple patches and multiple folds. `sample_soloqueue.json` covers the same ~48-champion pool used in the games fixture, for all 4 patches.
