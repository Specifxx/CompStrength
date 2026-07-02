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

from compstrength_pipeline import etl, features, pairwise, teams, train_model
from compstrength_pipeline.config import DEFAULT_CONFIG, PipelineConfig
from compstrength_pipeline.sources import soloqueue as soloqueue_module
from compstrength_pipeline.sources.soloqueue import SoloQueueSource, StaticSoloQueueSource

# 8 folds: with two seasons (~16k games) a 4-fold split leaves each fold's
# frozen model up to ~4,000 games stale by the end of its test window,
# badly understating a product that retrains DAILY. ~2k-game folds keep the
# simulated staleness closer to reality while every fold still has ample
# test data. Auto-reduces on small datasets (see module docstring).
DEFAULT_K_FOLDS = 8
MIN_K_FOLDS = 2
MIN_TEST_GAMES_PER_FOLD = 5

# Quintile calibration buckets by predicted blue-win-probability.
CALIBRATION_BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _resolve_patch_for_snapshot(games_df: pd.DataFrame) -> str:
    """Same logic as ``build._most_recent_patch``: the numerically-newest
    patch present (NOT the patch of the latest-dated row), so a snapshot's
    "current patch" / solo-queue key matches the newest patch in that fold.
    Kept as a thin wrapper here to avoid an import cycle (``build.py`` imports
    this module)."""
    newest = features.newest_patch(games_df)
    if newest is not None:
        return str(newest)
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
    presence_diff: float = 0.0,
    team_elo_diff: float = 0.0,
) -> float:
    """Mirror the frontend's combination rule (see train_model.py docstring):

        logit = intercept + scoreDiffWeight * scoreDiff
                + synergyWeight * synergyDiff + matchupWeight * matchupDiff
                + presenceWeight * presenceDiff
                + teamEloWeight * teamEloDiff + blueSideBias
    """
    logit_val = (
        coefficients.get("intercept", 0.0)
        + coefficients.get("scoreDiffWeight", 0.0) * score_diff
        + coefficients.get("synergyWeight", 0.0) * synergy_diff
        + coefficients.get("matchupWeight", 0.0) * matchup_diff
        + coefficients.get("presenceWeight", 0.0) * presence_diff
        + coefficients.get("teamEloWeight", 0.0) * team_elo_diff
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
    team_elo_diffs: dict[str, float] | None = None,
    solo_history: list | None = None,
) -> list[tuple[float, int, str, str, float]]:
    """Fit the full pipeline on ``train_games_df`` (as of ``reference_date``)
    and predict blue-win-probability for every game in ``test_games_df``.

    Returns a list of
    ``(predicted_proba, actual_blue_win, league, patch, draft_only_proba)``
    tuples, one per gameid in ``test_games_df`` (skipping any gameid that
    doesn't have both a blue and red side, which shouldn't happen post-ETL but
    is handled defensively). ``league``/``patch`` are carried so the aggregate
    report can break held-out accuracy down by league and by patch.

    ``team_elo_diffs`` maps gameid -> pre-game Elo gap (see ``teams.py``);
    the per-game PRE-game values only depend on strictly earlier games, so
    sharing one global Elo pass across folds is leak-free. Two models are fit
    per fold: the full one (with the team feature) drives ``predicted_proba``
    and a draft-only refit (team feature zeroed) drives ``draft_only_proba``,
    so the report can honestly separate "how good is the model when you know
    the teams" from "when you only know the draft". When the team feature is
    disabled the two are identical.
    """
    if train_games_df.empty:
        return []

    resolved_patch = _resolve_patch_for_snapshot(train_games_df)
    solo_winrates = solo_source.get_champion_winrates(resolved_patch)

    # Premier-league (LCK+LPL) target-share weighting, solved on this fold's
    # TRAIN set only (leak-free).
    config = features.apply_premier_league_weighting(
        config, train_games_df, reference_date
    )

    champion_features_df = features.compute_champion_features(
        games_df=train_games_df,
        bans_df=train_bans_df,
        solo_winrates=solo_winrates,
        config=config,
        reference_date=reference_date,
    )
    # Champion strength = shrunk decayed win rate (see
    # features.compute_wr_strength), computed on this fold's TRAIN games
    # only; loo_score_diffs feed the training frame (leave-one-game-out).
    # Leakage-free as-of join: only solo-queue snapshots COMMITTED before
    # this fold's boundary are visible to this fold's prior.
    solo_prior = (
        soloqueue_module.solo_winrates_asof(solo_history, reference_date)
        if solo_history
        else None
    )
    wr_strength, loo_score_diffs = features.compute_wr_strength(
        train_games_df, reference_date, config, solo_prior
    )
    champion_strength = wr_strength
    # Meta-presence feature: pickRate + banRate over this fold's training
    # window (train-only, so no leakage into the fold's test games).
    champion_presence = (
        (champion_features_df["pickRate"] + champion_features_df["banRate"]).to_dict()
        if config.use_presence_feature
        else {}
    )

    # The restricted set used for pairwise/model fitting must match what
    # compute_champion_features actually used internally: same min_patch
    # floor then most-recent-games cap. (Applying it to the fold's TRAIN set
    # only keeps the backtest leak-free -- no future patches leak in.)
    floored_train_games, floored_train_bans = features.restrict_to_min_patch(
        train_games_df, train_bans_df, config.min_patch
    )
    restricted_train_games, restricted_train_bans, _patches_used = (
        features.restrict_to_recent_games(
            floored_train_games, floored_train_bans, config.target_training_games
        )
    )

    strength_for_pairwise = {
        c: v * 4.0 / config.strength_feature_scale for c, v in champion_strength.items()
    }
    synergy_table = pairwise.compute_synergy_table(
        restricted_train_games, strength_for_pairwise, config, reference_date=reference_date
    )
    matchup_table = pairwise.compute_matchup_table(
        restricted_train_games, strength_for_pairwise, config, reference_date=reference_date
    )
    synergy_residuals = pairwise.synergy_lookup(synergy_table)
    matchup_residuals = pairwise.matchup_lookup(matchup_table)

    team_elo_diffs = team_elo_diffs or {}
    model_result = train_model.train_model(
        restricted_train_games,
        champion_strength,
        synergy_residuals,
        matchup_residuals,
        champion_presence,
        team_elo_diffs,
        loo_score_diffs,
    )
    # Draft-only companion fit (team feature zeroed) for the honest
    # "you don't know the teams" metrics. Cheap: only the LR refits; all the
    # expensive feature computation above is shared.
    draft_only_result = (
        train_model.train_model(
            restricted_train_games,
            champion_strength,
            synergy_residuals,
            matchup_residuals,
            champion_presence,
            None,
            loo_score_diffs,
        )
        if team_elo_diffs
        else model_result
    )

    predictions: list[tuple[float, int, str, str, float]] = []
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
        presence_diff = train_model.compute_presence_diff(
            champion_presence, blue_champs, red_champs
        )
        team_elo_diff = team_elo_diffs.get(gameid, 0.0)

        proba = _predict_proba(
            model_result.coefficients,
            score_diff,
            synergy_diff,
            matchup_diff,
            presence_diff,
            team_elo_diff,
        )
        draft_only_proba = (
            _predict_proba(
                draft_only_result.coefficients,
                score_diff,
                synergy_diff,
                matchup_diff,
                presence_diff,
                0.0,
            )
            if team_elo_diffs
            else proba
        )
        actual = int(blue_rows["result"].iloc[0] == 1)
        league = (
            str(blue_rows["league"].iloc[0])
            if "league" in blue_rows.columns and pd.notna(blue_rows["league"].iloc[0])
            else "Unknown"
        )
        patch = (
            str(blue_rows["patch"].iloc[0])
            if "patch" in blue_rows.columns and pd.notna(blue_rows["patch"].iloc[0])
            else "Unknown"
        )
        predictions.append((proba, actual, league, patch, draft_only_proba))

    return predictions


def _calibration_table(predictions: list[tuple]) -> list[dict]:
    """Bucket predictions into quintiles by predicted probability.

    ``predictions`` are ``(proba, actual, ...)`` tuples; only the first two
    fields are used here.
    """
    table = []
    for lo, hi in zip(CALIBRATION_BUCKET_EDGES[:-1], CALIBRATION_BUCKET_EDGES[1:]):
        is_last_bucket = hi >= 1.0
        bucket_preds = [
            (rec[0], rec[1])
            for rec in predictions
            if lo <= rec[0] < hi or (is_last_bucket and rec[0] == hi)
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


# Held-out segments (a league / a patch) below this many test games are folded
# into an aggregated "Other" row so the breakdown tables aren't dominated by
# tiny, statistically-meaningless slices.
MIN_SEGMENT_GAMES = 30


def _segment_breakdown(
    predictions: list[tuple], field_index: int, sort_key: str
) -> list[dict]:
    """Break held-out predictions down by a categorical field (league at index
    2, patch at index 3), reporting per-segment held-out accuracy, baseline,
    and log-loss. Segments with fewer than ``MIN_SEGMENT_GAMES`` test games are
    aggregated into a single ``"Other (<n> segments)"`` row.

    ``sort_key`` is ``"patch"`` (newest patch first, numerically) or
    ``"count"`` (largest segment first).
    """
    groups: dict[str, list[tuple[float, int]]] = {}
    for rec in predictions:
        groups.setdefault(rec[field_index], []).append((rec[0], rec[1]))

    def _row(name: str, preds: list[tuple[float, int]]) -> dict:
        probs = np.array([p for p, _ in preds])
        outs = np.array([y for _, y in preds])
        acc = float(np.mean((probs >= 0.5).astype(int) == outs))
        baseline = float(max(outs.mean(), 1.0 - outs.mean()))
        try:
            ll = float(log_loss(outs, probs, labels=[0, 1]))
        except ValueError:
            ll = float("nan")
        return {
            "name": name,
            "testGames": int(len(outs)),
            "accuracy": acc,
            "baselineAccuracy": baseline,
            "logLoss": ll,
            "blueWinRate": float(outs.mean()),
        }

    big = {k: v for k, v in groups.items() if len(v) >= MIN_SEGMENT_GAMES}
    small = {k: v for k, v in groups.items() if len(v) < MIN_SEGMENT_GAMES}

    rows = [_row(str(name), preds) for name, preds in big.items()]

    if sort_key == "patch":
        rows.sort(
            key=lambda r: features.parse_patch(r["name"]) or (-1, -1), reverse=True
        )
    else:
        rows.sort(key=lambda r: r["testGames"], reverse=True)

    if small:
        pooled = [pred for preds in small.values() for pred in preds]
        other = _row(f"Other ({len(small)} smaller)", pooled)
        rows.append(other)  # always last, regardless of sort

    return rows


def _data_composition(games_df: pd.DataFrame, config: PipelineConfig) -> dict:
    """Summarize the data the model is actually built on: total games, plus a
    per-patch and per-league count of games. Patches also carry their relative
    patch-recency weight (``patch_decay_base ** distance``) so the reader can
    see the newest patch is weighted most. Computed over the same restricted
    set the deployed model trains on (min_patch floor + most-recent-games cap).
    """
    floored, _ = features.restrict_to_min_patch(games_df, None, config.min_patch)
    restricted, _, _ = features.restrict_to_recent_games(
        floored, None, config.target_training_games
    )
    if restricted.empty:
        return {"totalGames": 0, "byPatch": [], "byLeague": []}

    per_game = restricted.drop_duplicates("gameid")
    total = int(per_game["gameid"].nunique())

    distances = features.patch_ordinal_distances(restricted)
    patch_counts = per_game.groupby("patch")["gameid"].nunique()
    by_patch = [
        {
            "name": str(p),
            "games": int(patch_counts[p]),
            "recencyWeight": float(config.patch_decay_base ** distances.get(p, 0)),
        }
        for p in patch_counts.index
    ]
    by_patch.sort(key=lambda r: features.parse_patch(r["name"]) or (-1, -1), reverse=True)

    by_league = []
    if "league" in per_game.columns:
        league_counts = per_game.groupby("league")["gameid"].nunique()
        by_league = [
            {"name": str(lg), "games": int(league_counts[lg])} for lg in league_counts.index
        ]
        by_league.sort(key=lambda r: r["games"], reverse=True)

    return {"totalGames": total, "byPatch": by_patch, "byLeague": by_league}


def _provenance_note(games_df: pd.DataFrame, n_predictions: int) -> str:
    """Describe what data this backtest actually ran on, honestly.

    We detect real Oracle's Elixir pro data vs. the tiny synthetic dev
    fixture from the data itself (no need to thread a source label through
    every caller): real OE seasons use game-client patch majors >= 15
    (2025 = 15.x, 2026 = 16.x, ...) and carry thousands of games, whereas
    the offline fixture is a handful of games on fabricated 14.x patches.
    """
    newest_major = None
    if "patch" in games_df.columns:
        majors = [
            parsed[0]
            for parsed in (features.parse_patch(p) for p in games_df["patch"].dropna())
            if parsed is not None
        ]
        if majors:
            newest_major = max(majors)

    is_real = newest_major is not None and newest_major >= 15 and n_predictions >= 200
    if is_real:
        season = 2010 + newest_major
        return (
            f"Walk-forward validation on real Oracle's Elixir pro-match data "
            f"({n_predictions:,} held-out games; newest patch {newest_major}.x = "
            f"{season} season). Each fold refits the entire pipeline using only "
            "games strictly before that fold, so there is no lookahead leakage. "
            "Draft-only prediction (champions picked, no in-game state) is a "
            "genuinely weak signal at the pro level -- treat accuracy a few points "
            "above the pick-majority baseline as expected, not a bug."
        )
    return (
        "This backtest ran on the small synthetic dev fixture (see "
        "tests/fixtures/README.md), not real historical Oracle's Elixir data "
        "-- treat all numbers here as illustrative of the *methodology* "
        "(walk-forward validation), not as a real-world accuracy claim."
    )


def run_backtest(
    games_df: pd.DataFrame,
    bans_df: pd.DataFrame,
    solo_source: SoloQueueSource,
    config: PipelineConfig = DEFAULT_CONFIG,
    k_folds: int = DEFAULT_K_FOLDS,
    solo_history: list | None = None,
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

    # ONE chronological Elo pass over everything, shared by all folds: each
    # game's recorded feature is its PRE-game Elo gap, which depends only on
    # strictly earlier games, so this is leak-free by construction (see
    # teams.py). Disabled -> empty dict -> the feature is zero everywhere and
    # its fitted weight is exactly 0.
    team_elo_diffs: dict[str, float] = {}
    if config.use_team_feature:
        team_elo_diffs = teams.elo_diff_by_gameid(
            teams.compute_team_elo(games_df, k=config.elo_k, season_carryover=config.elo_season_carryover),
            feature_scale=config.elo_feature_scale,
        )

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
                team_elo_diffs,
                solo_history,
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

    probs = np.array([rec[0] for rec in all_predictions])
    outcomes = np.array([rec[1] for rec in all_predictions])

    preds_binary = (probs >= 0.5).astype(int)
    accuracy = float(np.mean(preds_binary == outcomes))
    brier_score = float(np.mean((probs - outcomes) ** 2))
    log_loss_val = float(log_loss(outcomes, probs, labels=[0, 1]))

    baseline_accuracy = float(max(outcomes.mean(), 1.0 - outcomes.mean()))
    coin_flip_probs = np.full_like(probs, 0.5)
    coin_flip_log_loss = float(log_loss(outcomes, coin_flip_probs, labels=[0, 1]))

    # Draft-only companion metrics (team feature zeroed): "how good is the
    # model when you only know the draft". Identical to the main metrics when
    # the team feature is disabled.
    draft_probs = np.array([rec[4] for rec in all_predictions])
    draft_only_metrics = {
        "accuracy": float(np.mean((draft_probs >= 0.5).astype(int) == outcomes)),
        "logLoss": float(log_loss(outcomes, draft_probs, labels=[0, 1])),
        "brierScore": float(np.mean((draft_probs - outcomes) ** 2)),
    }

    # CURRENT-SEASON slice of the SAME predictions (test games on the newest
    # patch major, e.g. all 16.x = the 2026 season). With multi-season data
    # the all-folds average includes early-history folds whose models trained
    # on very little data -- informative about the method, but the number
    # that matches what a user faces TODAY (predicting current-season games
    # with all history available) is this slice.
    current_season_metrics = None
    patch_majors = [
        (features.parse_patch(rec[3]) or (0,))[0] for rec in all_predictions
    ]
    newest_major = max(patch_majors) if patch_majors else 0
    cs_mask = np.array([mj == newest_major for mj in patch_majors])
    if newest_major > 0 and 0 < cs_mask.sum() < len(outcomes):
        cs_out = outcomes[cs_mask]
        cs_probs = probs[cs_mask]
        cs_draft = draft_probs[cs_mask]
        current_season_metrics = {
            "testGames": int(cs_mask.sum()),
            "patchMajor": int(newest_major),
            "accuracy": float(np.mean((cs_probs >= 0.5).astype(int) == cs_out)),
            "logLoss": float(log_loss(cs_out, cs_probs, labels=[0, 1])),
            "baselineAccuracy": float(max(cs_out.mean(), 1 - cs_out.mean())),
            "draftOnlyAccuracy": float(np.mean((cs_draft >= 0.5).astype(int) == cs_out)),
            "draftOnlyLogLoss": float(log_loss(cs_out, cs_draft, labels=[0, 1])),
        }

    calibration = _calibration_table(all_predictions)
    breakdowns = {
        "byPatch": _segment_breakdown(all_predictions, field_index=3, sort_key="patch"),
        "byLeague": _segment_breakdown(all_predictions, field_index=2, sort_key="count"),
    }
    data_composition = _data_composition(games_df, config)

    note_parts = []
    if degraded_note:
        note_parts.append(degraded_note.strip())
    if skipped_fold_notes:
        note_parts.append(" ".join(skipped_fold_notes))
    note_parts.append(_provenance_note(games_df, len(all_predictions)))

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
        # With config.use_team_feature on, "metrics" above are WITH the team
        # feature (both teams known -- the realistic pre-match scenario, and
        # the one relevant for comparing against bookmaker odds); this block
        # is the same held-out games scored by the draft-only companion fit.
        "draftOnlyMetrics": draft_only_metrics,
        "currentSeasonMetrics": current_season_metrics,
        "teamFeatureUsed": bool(team_elo_diffs),
        "calibration": calibration,
        "dataComposition": data_composition,
        "breakdowns": breakdowns,
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
        "dataComposition": {"totalGames": 0, "byPatch": [], "byLeague": []},
        "breakdowns": {"byPatch": [], "byLeague": []},
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
