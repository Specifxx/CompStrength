"""Walk-forward (chronological) cross-validation backtest for the whole
CompStrength pipeline.

The product ask is simple: "do a backtest on the accuracy of this whole
thing." A random train/test split would leak future information into the
past (a champion's blended rating on patch N would be computed using pro
games from patch N+1), which isn't how the model is actually used in
production (predictions are always made for *upcoming* games using only
*past* data). So this module instead does walk-forward validation:

1. Sort all patch-restricted games chronologically.
2. Split into K folds by date (a fold is a contiguous date range).
3. For each fold, recompute the *entire* feature pipeline -- champion
   ratings (``features.py``), pairwise synergy/matchup (``pairwise.py``),
   and the logistic regression (``train_model.py``) -- using only games
   strictly before that fold's start date, with ``reference_date`` pinned
   to the fold boundary (so recency decay is computed "as of" that
   boundary, not as of "today"). This reuses all the existing pipeline
   functions unchanged; the only difference from a normal ``build.py`` run
   is the date-filtered input and the pinned reference date.
4. Use that frozen, boundary-dated snapshot to predict each game in the
   fold (never games used to fit it), and compare to the actual outcome.

Fold count auto-reduces if there isn't enough data: we require at least
``MIN_TEST_GAMES_PER_FOLD`` test games per fold, backing off from
``DEFAULT_K_FOLDS`` down to ``MIN_K_FOLDS``; if even ``MIN_K_FOLDS`` folds
would leave a fold under-sized, we still run with ``MIN_K_FOLDS`` folds but
note the caveat rather than crashing.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from compstrength_pipeline import etl, features, pairwise, train_model
from compstrength_pipeline.config import DEFAULT_CONFIG, PipelineConfig
from compstrength_pipeline.sources.soloqueue import SoloQueueSource, StaticSoloQueueSource

DEFAULT_K_FOLDS = 4
MIN_K_FOLDS = 2
MIN_TEST_GAMES_PER_FOLD = 5

# Quintile calibration buckets by predicted blue-win-probability.
CALIBRATION_BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _resolve_patch_for_snapshot(games_df: pd.DataFrame) -> str:
    """Same logic as ``build._most_recent_patch``, duplicated to avoid an
    import cycle (``build.py`` imports this module)."""
    latest_date = games_df["date"].max()
    latest_rows = games_df[games_df["date"] == latest_date]
    return str(latest_rows["patch"].iloc[0])


def _choose_fold_boundaries(
    sorted_dates: pd.Series, k_folds: int
) -> list[pd.Timestamp]:
    """Return ``k_folds - 1`` interior boundary dates splitting the data into
    ``k_folds`` roughly equal-sized contiguous chronological chunks (by row
    count of the sorted per-game date series)."""
    n = len(sorted_dates)
    boundaries = []
    for i in range(1, k_folds):
        idx = min(int(round(n * i / k_folds)), n - 1)
        boundaries.append(sorted_dates.iloc[idx])
    return boundaries


def _predict_proba(
    coefficients: dict[str, float],
    score_diff: float,
    synergy_diff: float,
    matchup_diff: float,
) -> float:
    """Mirror the frontend's combination rule (see train_model.py docstring):

        logit = intercept + scoreDiffWeight * scoreDiff
                + synergyWeight * synergyDiff + matchupWeight * matchupDiff
                + blueSideBias
    """
    logit_val = (
        coefficients.get("intercept", 0.0)
        + coefficients.get("scoreDiffWeight", 0.0) * score_diff
        + coefficients.get("synergyWeight", 0.0) * synergy_diff
        + coefficients.get("matchupWeight", 0.0) * matchup_diff
        + coefficients.get("blueSideBias", 0.0)
    )
    return 1.0 / (1.0 + math.exp(-logit_val))


def _fit_snapshot_and_predict(
    train_games_df: pd.DataFrame,
    train_bans_df: pd.DataFrame,
    test_games_df: pd.DataFrame,
    solo_source: SoloQueueSource,
    config: PipelineConfig,
    reference_date: pd.Timestamp,
) -> list[tuple[float, int]]:
    """Fit the full pipeline on ``train_games_df`` (as of ``reference_date``)
    and predict blue-win-probability for every game in ``test_games_df``.

    Returns a list of ``(predicted_proba, actual_blue_win)`` tuples, one
    per gameid in ``test_games_df`` (skipping any gameid that doesn't have
    both a blue and red side, which shouldn't happen post-ETL but is
    handled defensively).
    """
    if train_games_df.empty:
        return []

    resolved_patch = _resolve_patch_for_snapshot(train_games_df)
    solo_winrates = solo_source.get_champion_winrates(resolved_patch)

    champion_features_df = features.compute_champion_features(
        games_df=train_games_df,
        bans_df=train_bans_df,
        solo_winrates=solo_winrates,
        config=config,
        reference_date=reference_date,
    )
    champion_strength = champion_features_df["strengthScore"].to_dict()

    # The restricted set used for pairwise/model fitting must match what
    # compute_champion_features actually used internally.
    restricted_train_games, restricted_train_bans, _patches_used = (
        features.restrict_to_recent_games(
            train_games_df, train_bans_df, config.target_training_games
        )
    )

    synergy_table = pairwise.compute_synergy_table(
        restricted_train_games, champion_strength, config, reference_date=reference_date
    )
    matchup_table = pairwise.compute_matchup_table(
        restricted_train_games, champion_strength, config, reference_date=reference_date
    )
    synergy_residuals = pairwise.synergy_lookup(synergy_table)
    matchup_residuals = pairwise.matchup_lookup(matchup_table)

    model_result = train_model.train_model(
        restricted_train_games, champion_strength, synergy_residuals, matchup_residuals
    )

    predictions: list[tuple[float, int]] = []
    for gameid, group in test_games_df.groupby("gameid"):
        blue_rows = group[group["side"].str.lower() == "blue"]
        red_rows = group[group["side"].str.lower() == "red"]
        if blue_rows.empty or red_rows.empty:
            continue

        blue_champs = blue_rows["champion"].tolist()
        red_champs = red_rows["champion"].tolist()
        blue_by_role = dict(zip(blue_rows["position"], blue_rows["champion"]))
        red_by_role = dict(zip(red_rows["position"], red_rows["champion"]))

        score_diff = train_model.compute_score_diff(champion_strength, blue_champs, red_champs)
        synergy_diff = train_model.compute_synergy_diff(
            synergy_residuals, blue_champs, red_champs
        )
        matchup_diff = train_model.compute_matchup_diff(
            matchup_residuals, blue_by_role, red_by_role
        )

        proba = _predict_proba(model_result.coefficients, score_diff, synergy_diff, matchup_diff)
        actual = int(blue_rows["result"].iloc[0] == 1)
        predictions.append((proba, actual))

    return predictions


def _calibration_table(predictions: list[tuple[float, int]]) -> list[dict]:
    """Bucket predictions into quintiles by predicted probability."""
    table = []
    for lo, hi in zip(CALIBRATION_BUCKET_EDGES[:-1], CALIBRATION_BUCKET_EDGES[1:]):
        is_last_bucket = hi >= 1.0
        bucket_preds = [
            (p, y) for p, y in predictions if lo <= p < hi or (is_last_bucket and p == hi)
        ]
        if not bucket_preds:
            continue
        probs = [p for p, _ in bucket_preds]
        outcomes = [y for _, y in bucket_preds]
        table.append(
            {
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "predictedMean": float(np.mean(probs)),
                "actualWinRate": float(np.mean(outcomes)),
                "count": len(bucket_preds),
            }
        )
    return table


def run_backtest(
    games_df: pd.DataFrame,
    bans_df: pd.DataFrame,
    solo_source: SoloQueueSource,
    config: PipelineConfig = DEFAULT_CONFIG,
    k_folds: int = DEFAULT_K_FOLDS,
) -> dict:
    """Run walk-forward cross-validation over ``games_df``/``bans_df``.

    Args:
        games_df: Cleaned per-player-game table (post ``etl.build_raw_tables``).
        bans_df: Cleaned bans table (post ``etl.build_raw_tables``).
        solo_source: A ``SoloQueueSource`` used to fetch solo-queue win
            rates for each fold's resolved patch.
        config: Hyperparameters. Note ``config.target_training_games`` is
            applied *within* each fold's training snapshot (i.e. each fold
            still only looks at its own most-recent games as of that
            fold's boundary), not globally beforehand.
        k_folds: Requested number of folds (auto-reduced if there isn't
            enough data; see module docstring).

    Returns:
        A dict matching the ``data/backtest_report.json`` schema (see
        ``build.py`` / the project spec), except for ``generatedAt`` which
        callers should stamp themselves if writing to disk (this function
        does still include a ``generatedAt`` timestamp for convenience
        when called standalone).
    """
    if games_df is None or games_df.empty:
        return _empty_report("No games available to backtest.")

    # Restrict to complete 10-row games with a resolvable side/date/patch,
    # sorted chronologically by game (using each gameid's date).
    per_game_dates = games_df.groupby("gameid")["date"].max().sort_values()
    n_games = len(per_game_dates)

    if n_games < MIN_TEST_GAMES_PER_FOLD * MIN_K_FOLDS:
        return _empty_report(
            f"Only {n_games} total games available; not enough data to run a "
            f"meaningful walk-forward backtest (need at least "
            f"{MIN_TEST_GAMES_PER_FOLD * MIN_K_FOLDS} games for the minimum "
            f"{MIN_K_FOLDS} folds). This is expected on the small synthetic "
            "fixture; treat any numbers here as illustrative only."
        )

    # Auto-reduce K until each fold's test slice would have enough games,
    # down to MIN_K_FOLDS.
    chosen_k = None
    for k in range(min(k_folds, DEFAULT_K_FOLDS), MIN_K_FOLDS - 1, -1):
        avg_test_games_per_fold = n_games / k
        if avg_test_games_per_fold >= MIN_TEST_GAMES_PER_FOLD:
            chosen_k = k
            break
    degraded_note = ""
    if chosen_k is None:
        chosen_k = MIN_K_FOLDS
        degraded_note = (
            f"Even the minimum {MIN_K_FOLDS} folds have fewer than "
            f"{MIN_TEST_GAMES_PER_FOLD} test games on average ({n_games} total "
            f"games); running anyway, but per-fold metrics (and the aggregate) "
            "should be treated as very noisy. "
        )

    boundaries = _choose_fold_boundaries(per_game_dates, chosen_k)
    # Fold i covers games with date in [boundaries[i-1] (or -inf), boundaries[i] (or +inf)).
    fold_edges = [None] + boundaries + [None]

    all_predictions: list[tuple[float, int]] = []
    folds_run = 0
    skipped_fold_notes: list[str] = []

    gameid_to_date = games_df.groupby("gameid")["date"].max()

    for i in range(chosen_k):
        lo = fold_edges[i]
        hi = fold_edges[i + 1]

        test_gameids = gameid_to_date[
            (gameid_to_date >= lo if lo is not None else True)
            & (gameid_to_date < hi if hi is not None else True)
        ].index
        train_gameids = gameid_to_date[gameid_to_date < lo].index if lo is not None else pd.Index([])

        test_games_df = games_df[games_df["gameid"].isin(test_gameids)]
        train_games_df = games_df[games_df["gameid"].isin(train_gameids)]
        train_bans_df = (
            bans_df[bans_df["gameid"].isin(train_gameids)]
            if bans_df is not None and not bans_df.empty
            else bans_df
        )

        n_test = test_games_df["gameid"].nunique()
        if n_test < 1 or train_games_df.empty:
            skipped_fold_notes.append(
                f"Fold {i + 1}/{chosen_k} skipped: insufficient train/test data "
                f"(train_games={train_games_df['gameid'].nunique() if not train_games_df.empty else 0}, "
                f"test_games={n_test})."
            )
            continue

        reference_date = lo if lo is not None else train_games_df["date"].max()

        try:
            fold_predictions = _fit_snapshot_and_predict(
                train_games_df,
                train_bans_df,
                test_games_df,
                solo_source,
                config,
                reference_date,
            )
        except Exception as exc:  # noqa: BLE001 - one bad fold shouldn't sink the backtest
            warnings.warn(f"Backtest fold {i + 1}/{chosen_k} failed: {exc!r}")
            skipped_fold_notes.append(f"Fold {i + 1}/{chosen_k} failed: {exc!r}")
            continue

        if fold_predictions:
            all_predictions.extend(fold_predictions)
            folds_run += 1

    if not all_predictions:
        return _empty_report(
            "No fold produced any predictions (likely too little data or too "
            "few training games survived patch/date filtering in every fold). "
            + " ".join(skipped_fold_notes)
        )

    probs = np.array([p for p, _ in all_predictions])
    outcomes = np.array([y for _, y in all_predictions])

    preds_binary = (probs >= 0.5).astype(int)
    accuracy = float(np.mean(preds_binary == outcomes))
    brier_score = float(np.mean((probs - outcomes) ** 2))
    log_loss_val = float(log_loss(outcomes, probs, labels=[0, 1]))

    baseline_accuracy = float(max(outcomes.mean(), 1.0 - outcomes.mean()))
    coin_flip_probs = np.full_like(probs, 0.5)
    coin_flip_log_loss = float(log_loss(outcomes, coin_flip_probs, labels=[0, 1]))

    calibration = _calibration_table(all_predictions)

    note_parts = []
    if degraded_note:
        note_parts.append(degraded_note.strip())
    if skipped_fold_notes:
        note_parts.append(" ".join(skipped_fold_notes))
    note_parts.append(
        "This backtest ran on a small synthetic fixture (see "
        "tests/fixtures/README.md), not real historical Oracle's Elixir "
        "data -- treat all numbers here as illustrative of the *methodology* "
        "(walk-forward validation), not as a real-world accuracy claim."
    )

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "folds": folds_run,
        "testGames": len(all_predictions),
        "metrics": {
            "accuracy": accuracy,
            "logLoss": log_loss_val,
            "brierScore": brier_score,
            "baselineAccuracy": baseline_accuracy,
            "coinFlipLogLoss": coin_flip_log_loss,
        },
        "calibration": calibration,
        "note": " ".join(note_parts),
    }


def _empty_report(note: str) -> dict:
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "folds": 0,
        "testGames": 0,
        "metrics": {
            "accuracy": float("nan"),
            "logLoss": float("nan"),
            "brierScore": float("nan"),
            "baselineAccuracy": float("nan"),
            "coinFlipLogLoss": float("nan"),
        },
        "calibration": [],
        "note": note,
    }


def _print_text_report(report: dict) -> None:
    print("=" * 60)
    print("CompStrength walk-forward backtest report")
    print("=" * 60)
    print(f"generatedAt : {report['generatedAt']}")
    print(f"folds       : {report['folds']}")
    print(f"testGames   : {report['testGames']}")
    print("-" * 60)
    print("Metrics:")
    for key, value in report["metrics"].items():
        print(f"  {key:<18}: {value}")
    print("-" * 60)
    print("Calibration (by predicted-probability quintile):")
    if not report["calibration"]:
        print("  (no calibration data)")
    else:
        print(f"  {'bucket':<12}{'predictedMean':<16}{'actualWinRate':<16}{'count':<8}")
        for row in report["calibration"]:
            print(
                f"  {row['bucket']:<12}{row['predictedMean']:<16.4f}"
                f"{row['actualWinRate']:<16.4f}{row['count']:<8}"
            )
    print("-" * 60)
    print(f"note: {report['note']}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    package_dir = Path(__file__).resolve().parent.parent
    default_fixture_csv = package_dir / "tests" / "fixtures" / "sample_oracleselixir.csv"
    default_soloqueue_fixture = package_dir / "tests" / "fixtures" / "sample_soloqueue.json"
    repo_root = package_dir.parent.parent
    default_output_dir = repo_root / "data"

    parser.add_argument("--oracles-elixir-path", default=str(default_fixture_csv))
    parser.add_argument("--soloqueue-fixture", default=str(default_soloqueue_fixture))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args(argv)

    from compstrength_pipeline.sources.oracles_elixir import extract_bans, fetch_oracles_elixir

    games_df_raw = fetch_oracles_elixir(args.oracles_elixir_path)
    raw = pd.read_csv(args.oracles_elixir_path, low_memory=False)
    bans_df_raw = extract_bans(raw)
    games_df, bans_df = etl.build_raw_tables(games_df_raw, bans_df_raw)

    solo_source = StaticSoloQueueSource(args.soloqueue_fixture)

    report = run_backtest(games_df, bans_df, solo_source, DEFAULT_CONFIG, k_folds=args.folds)

    _print_text_report(report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "backtest_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
