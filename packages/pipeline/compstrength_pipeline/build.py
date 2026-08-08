"""CLI entrypoint: run the full ETL -> features -> train_model pipeline and
write ``champion_ratings.json`` and ``model.json`` for the website to serve.

Usage:

    python -m compstrength_pipeline.build
    python -m compstrength_pipeline.build --oracles-elixir-path tests/fixtures/sample_oracleselixir.csv \\
        --soloqueue-fixture tests/fixtures/sample_soloqueue.json
    python -m compstrength_pipeline.build --oracles-elixir-url https://.../2026_LoL_esports_match_data...csv
    python -m compstrength_pipeline.build --source leaguepedia

By default, this points at the bundled small synthetic fixtures under
``tests/fixtures/`` so the pipeline can run fully offline (e.g. in this
sandbox, or for a quick local demo). Pass ``--oracles-elixir-url`` (or set
the ``ORACLES_ELIXIR_URL`` env var) to fetch live Oracle's Elixir data
instead, or ``--source leaguepedia`` to fetch real, current match data
directly from Leaguepedia's Cargo API (``sources/leaguepedia.py``) -- this
is the mode used by the scheduled GitHub Actions refresh, where network
egress is not blocked (both this sandbox's dev network policy and
oracleselixir.com/lol.fandom.com being blocked from it are the reason
fixtures are the default here).
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from compstrength_pipeline import backtest, etl, features, pairwise, players, teams, train_model
from compstrength_pipeline.config import DEFAULT_CONFIG, PipelineConfig
from compstrength_pipeline.sources import leaguepedia as leaguepedia_source
from compstrength_pipeline.sources import soloqueue as soloqueue_module
from compstrength_pipeline.sources import oracles_elixir as oracles_elixir_source
from compstrength_pipeline.sources.oracles_elixir import (
    extract_bans,
    fetch_oracles_elixir,
)
from compstrength_pipeline.sources.soloqueue import NullSoloQueueSource, StaticSoloQueueSource

PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_FIXTURE_CSV = PACKAGE_DIR / "tests" / "fixtures" / "sample_oracleselixir.csv"
DEFAULT_SOLOQUEUE_FIXTURE = PACKAGE_DIR / "tests" / "fixtures" / "sample_soloqueue.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["fixture", "oracles-elixir", "leaguepedia"],
        default=os.environ.get("COMPSTRENGTH_SOURCE", "fixture"),
        help=(
            "Which pro-match data source to use. 'oracles-elixir' (RECOMMENDED "
            "for real data) downloads Oracle's Elixir's bulk season CSV from "
            "Google Drive -- one download, thousands of real pro games, no "
            "per-request rate limiting. 'leaguepedia' fetches live from "
            "Leaguepedia's Cargo API (rate-limited on shared CI IPs). "
            "'fixture' (default) reads the bundled synthetic fixture for "
            "offline dev/tests. The real sources need unblocked network "
            "egress (e.g. GitHub Actions), not this dev sandbox."
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        default=(
            int(os.environ["COMPSTRENGTH_YEAR"])
            if os.environ.get("COMPSTRENGTH_YEAR")
            else None
        ),
        help=(
            "Season year for --source oracles-elixir (default: current year "
            "if known to ORACLES_ELIXIR_DRIVE_IDS, else the newest known year)."
        ),
    )
    parser.add_argument(
        "--oracles-elixir-path",
        default=os.environ.get("ORACLES_ELIXIR_PATH", str(DEFAULT_FIXTURE_CSV)),
        help="Local path to an Oracle's Elixir CSV (default: bundled synthetic fixture).",
    )
    parser.add_argument(
        "--oracles-elixir-url",
        default=os.environ.get("ORACLES_ELIXIR_URL"),
        help="Remote URL to a live Oracle's Elixir CSV. If set, overrides --oracles-elixir-path.",
    )
    parser.add_argument(
        "--soloqueue-fixture",
        default=os.environ.get("SOLOQUEUE_FIXTURE_PATH", str(DEFAULT_SOLOQUEUE_FIXTURE)),
        help="Local JSON fixture for StaticSoloQueueSource (default: bundled synthetic fixture).",
    )
    parser.add_argument(
        "--patch",
        default=os.environ.get("COMPSTRENGTH_PATCH"),
        help="Patch to generate ratings for (default: the most recent patch in the games data).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("COMPSTRENGTH_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
        help="Directory to write champion_ratings.json and model.json into.",
    )
    parser.add_argument(
        "--target-games",
        type=int,
        # Note: os.environ.get(key, "0") only falls back to "0" when the key
        # is ABSENT -- GitHub Actions sets COMPSTRENGTH_TARGET_GAMES to an
        # empty string (not unset) when the workflow_dispatch input is left
        # blank, so int(os.environ.get(...)) would crash on int(''). Guard
        # explicitly instead.
        default=(int(os.environ["COMPSTRENGTH_TARGET_GAMES"]) or None)
        if os.environ.get("COMPSTRENGTH_TARGET_GAMES")
        else None,
        help=(
            "Override PipelineConfig.target_training_games (default 1000). "
            "Useful for a quick, fast --source leaguepedia smoke test with a "
            "small number before committing to a full-size live fetch."
        ),
    )
    return parser.parse_args(argv)


def _games_and_bans_from_csv_path(
    csv_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Read a local Oracle's-Elixir-shaped CSV ONCE and produce the normalized
    games table, the bans table, and the per-game gold/min margins from the
    same in-memory frame (all three need the team-summary rows)."""
    # Read patch as a string (see oracles_elixir._READ_CSV_KWARGS): otherwise
    # pandas coerces "16.10" -> 16.1, silently corrupting patch-recency logic.
    raw = pd.read_csv(csv_path, **oracles_elixir_source._READ_CSV_KWARGS)
    games_df = oracles_elixir_source._normalize_player_games(raw)
    if raw.empty:
        return (
            games_df,
            pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"]),
            {},
        )
    return (
        games_df,
        oracles_elixir_source.extract_bans(raw),
        oracles_elixir_source.extract_game_margins(raw),
    )


def load_games_and_bans(
    oracles_elixir_path: str, oracles_elixir_url: str | None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Load raw Oracle's Elixir data (live URL if given, else local path/fixture)."""
    if oracles_elixir_url:
        games_df = fetch_oracles_elixir(oracles_elixir_url)
        # Best-effort: re-fetch raw for ban + margin extraction. Failures here
        # are non-fatal (bans come back empty and Elo falls back to binary).
        try:
            import io

            import requests

            resp = requests.get(oracles_elixir_url, timeout=30)
            resp.raise_for_status()
            raw = pd.read_csv(
                io.StringIO(resp.text), **oracles_elixir_source._READ_CSV_KWARGS
            )
            if raw.empty:
                bans_df = pd.DataFrame(
                    columns=["gameid", "team", "champion", "ban_number"]
                )
                margins: dict[str, float] = {}
            else:
                bans_df = extract_bans(raw)
                margins = oracles_elixir_source.extract_game_margins(raw)
        except Exception as exc:  # pragma: no cover - network path
            warnings.warn(f"Could not fetch raw data for ban extraction: {exc!r}")
            bans_df = pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])
            margins = {}
        return games_df, bans_df, margins
    return _games_and_bans_from_csv_path(oracles_elixir_path)


def _resolve_oe_year(year: int | None) -> int:
    """Pick the Oracle's Elixir season year: explicit --year, else the newest
    year we have a Drive file ID for."""
    if year is not None:
        return year
    from compstrength_pipeline.config import ORACLES_ELIXIR_DRIVE_IDS

    return max(ORACLES_ELIXIR_DRIVE_IDS)


def load_raw_games_and_bans(
    source: str,
    oracles_elixir_path: str,
    oracles_elixir_url: str | None,
    target_games: int,
    year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Dispatch to the requested data source's raw fetch.

    Returns ``(games_df, bans_df, game_margins)``. ``game_margins`` maps
    gameid -> blue-minus-red gold per minute and powers margin-of-victory
    Elo; sources that don't expose team gold totals (Leaguepedia) return an
    empty map, which makes Elo fall back to the classic binary update.

    - ``oracles-elixir``: download the bulk season CSV from Google Drive
      ONCE (one request, no per-request rate limiting), and derive both the
      games and bans tables from that single file.
    - ``leaguepedia``: fetch live from Leaguepedia's Cargo API.
    - anything else (``fixture``): read the bundled synthetic fixture / a
      local CSV path or URL.
    """
    if source == "oracles-elixir":
        resolved_year = _resolve_oe_year(year)
        try:
            csv_path = oracles_elixir_source.download_oracles_elixir_csv(resolved_year)
            games_df, bans_df, margins = _games_and_bans_from_csv_path(str(csv_path))
            # The min_patch floor (25.1 -> current) deliberately spans TWO
            # seasons, so also pull the PREVIOUS season's file: its patches are
            # exponentially down-weighted (patch_decay_base ** distance) so it
            # barely moves the current ratings, but it (a) gives the
            # walk-forward backtest real training data for its earliest folds
            # (fold 1 currently has zero) and (b) keeps the model usable in the
            # first weeks of a new season, when the current-year file is nearly
            # empty. Best-effort: if the previous year's file can't be fetched
            # (e.g. Drive quota / sandbox egress), proceed with one season.
            prev_year = resolved_year - 1
            try:
                prev_csv = oracles_elixir_source.download_oracles_elixir_csv(prev_year)
                prev_games, prev_bans, prev_margins = _games_and_bans_from_csv_path(
                    str(prev_csv)
                )
                # Defensive: a gameid present in BOTH season files would merge
                # into a 20-row "game" that ETL's exactly-10-rows filter then
                # silently drops -- so keep only prev-season games that don't
                # already exist in the current season's file.
                current_ids = set(games_df["gameid"].dropna())
                prev_games = prev_games[~prev_games["gameid"].isin(current_ids)]
                prev_bans = prev_bans[~prev_bans["gameid"].isin(current_ids)]
                games_df = pd.concat([prev_games, games_df], ignore_index=True)
                bans_df = pd.concat([prev_bans, bans_df], ignore_index=True)
                # Current season wins any gameid collision (same rule as above).
                margins = {
                    **{g: m for g, m in prev_margins.items() if g not in current_ids},
                    **margins,
                }
                print(
                    f"Merged {prev_year} season: total {games_df['gameid'].nunique()} "
                    "games across both seasons (older patches exponentially "
                    "down-weighted; min_patch floor still applies)."
                )
            except oracles_elixir_source.DataSourceUnavailableError as prev_exc:
                warnings.warn(
                    f"Could not fetch the previous season ({prev_year}) CSV "
                    f"({prev_exc}); continuing with {resolved_year} only."
                )
            return games_df, bans_df, margins
        except oracles_elixir_source.DataSourceUnavailableError as exc:
            # The shared 2026 Drive file is heavily used and can hit Google's
            # per-file public-download quota ("too many users..."). That's an
            # independent failure mode from Leaguepedia's API rate limit, so
            # fall back to the live Leaguepedia fetch rather than failing the
            # whole refresh -- both are real sources; we just want whichever
            # is reachable this run.
            # Cap the Leaguepedia fallback: it makes one request per ~50-game
            # chunk and rate-limits on shared CI IPs, so a huge window would
            # get throttled. It fetches the MOST RECENT games first, which are
            # exactly the highest-weighted patches (26.13, 26.12, ...) -- the
            # older 25.x tail is exponentially down-weighted to ~nothing
            # anyway, so grabbing the recent slice loses almost no signal.
            fallback_games = min(target_games, 400)
            warnings.warn(
                f"Oracle's Elixir Drive download failed ({exc}); falling back "
                f"to a live Leaguepedia Cargo fetch of the {fallback_games} "
                "most recent games."
            )
            lp_games, lp_bans = leaguepedia_source.fetch_recent_games(
                target_games=fallback_games
            )
            return lp_games, lp_bans, {}
    if source == "leaguepedia":
        lp_games, lp_bans = leaguepedia_source.fetch_recent_games(
            target_games=target_games
        )
        return lp_games, lp_bans, {}
    return load_games_and_bans(oracles_elixir_path, oracles_elixir_url)


def run_pipeline_on_data(
    raw_games_df: pd.DataFrame,
    raw_bans_df: pd.DataFrame,
    soloqueue_source,
    patch: str | None,
    config: PipelineConfig = DEFAULT_CONFIG,
    game_margins: dict[str, float] | None = None,
) -> tuple[dict, dict, dict, dict, dict]:
    """Run ETL -> features -> pairwise -> train_model on already-fetched data.

    This is the pure/reusable core: it takes raw (not yet ETL-cleaned)
    games/bans DataFrames and an already-constructed ``SoloQueueSource``
    rather than fetching/constructing them itself, so ``main()`` can fetch
    once and reuse the same snapshot for both the live build and the
    backtest (important for the ``leaguepedia`` source, where fetching is
    several rate-limited network round-trips and shouldn't be repeated).

    Returns:
        ``(champion_ratings_payload, model_payload, synergy_payload)``
        matching the exact JSON schemas documented in the project spec /
        README.
    """
    games_df, bans_df = etl.build_raw_tables(raw_games_df, raw_bans_df)

    if games_df.empty:
        raise ValueError(
            "No complete games survived ETL cleaning; cannot compute features. "
            "Check the input data source."
        )

    # Restrict the data window up front so every downstream computation
    # (resolved patch, champion features, pairwise synergy/matchup, and the
    # model training frame) operates on exactly the same games:
    #   1. drop patches older than config.min_patch (e.g. "25.1"), then
    #   2. cap at the most recent config.target_training_games as a safety bound.
    games_df, bans_df = features.restrict_to_min_patch(games_df, bans_df, config.min_patch)
    games_df, bans_df, patches_used = features.restrict_to_recent_games(
        games_df, bans_df, config.target_training_games
    )
    if games_df.empty:
        raise ValueError(
            "No games remain after restricting to the most recent games; "
            "cannot compute features. Check the input data source."
        )

    resolved_patch = patch or _most_recent_patch(games_df)
    reference_date = games_df["date"].max()

    # Real solo-queue win rates from the committed, timestamped history
    # (data/soloqueue_history.json), as of the data's reference date (the same
    # leak-free as-of path the backtest uses). On the real pipeline the
    # ``soloqueue_source`` is a NullSoloQueueSource (returns nothing), so
    # without this every champion's displayed solo win rate would be a flat
    # 50% placeholder. We surface the REAL numbers for the display fields
    # (soloWinRate / soloGames / blendedWinRate); the model's champion
    # strength still comes from ``compute_wr_strength`` below, independent of
    # this. The offline fixture path keeps its StaticSoloQueueSource data.
    solo_history = soloqueue_module.load_solo_history(
        REPO_ROOT / "data" / "soloqueue_history.json"
    )
    solo_prior = (
        soloqueue_module.solo_winrates_asof(solo_history, reference_date)
        if solo_history
        else None
    )
    solo_winrates = soloqueue_source.get_champion_winrates(resolved_patch)
    if not solo_winrates and solo_prior:
        solo_winrates = solo_prior
    if solo_prior:
        print(f"solo-queue data: {len(solo_prior)} champions (as of {reference_date.date()})")

    # Owner directive: premier leagues (LCK+LPL) carry a fixed share (~70%)
    # of the decayed training weight. Solves for the multiplier on this exact
    # window and threads it through the existing league-weight machinery.
    config = features.apply_premier_league_weighting(config, games_df, reference_date)
    champion_features_df = features.compute_champion_features(
        games_df=games_df,
        bans_df=bans_df,
        solo_winrates=solo_winrates,
        config=config,
        reference_date=reference_date,
    )

    # Champion strength = shrunk decayed win rate (the draft signal that won
    # the experiment fleet: 55.4% vs 52.5% held-out draft-only accuracy).
    # Overrides the EB-logit strengthScore in the artifact; the EB stats
    # (proWinRate/blendedWinRate/sampleConfidence) remain as display fields.
    # The solo prior into the strength signal is a measured no-op (shipped
    # weight 0.0), so passing solo_prior here does not change the model.
    wr_strength, loo_score_diffs = features.compute_wr_strength(
        games_df, reference_date, config, solo_prior
    )
    champion_features_df["strengthScore"] = [
        wr_strength.get(c, 0.0) for c in champion_features_df.index
    ]
    champion_strength = {
        c: wr_strength.get(c, 0.0) for c in champion_features_df.index
    }
    # Meta-presence feature (pickRate + banRate): same values shipped per
    # champion in champion_ratings.json, so the frontend can reconstruct
    # presence_diff exactly (see apps/web/lib/predict.ts).
    champion_presence = (
        (champion_features_df["pickRate"] + champion_features_df["banRate"]).to_dict()
        if config.use_presence_feature
        else {}
    )

    # Pairwise de-confounding expects LOGIT-scale strengths; near a 50% base
    # rate logit(p) ~= 4*(p - 0.5), so convert from the scaled feature units.
    strength_for_pairwise = {
        c: v * 4.0 / config.strength_feature_scale for c, v in champion_strength.items()
    }
    synergy_table = pairwise.compute_synergy_table(
        games_df, strength_for_pairwise, config, reference_date=reference_date
    )
    matchup_table = pairwise.compute_matchup_table(
        games_df, strength_for_pairwise, config, reference_date=reference_date
    )
    synergy_residuals = pairwise.synergy_lookup(synergy_table)
    matchup_residuals = pairwise.matchup_lookup(matchup_table)

    # Team-strength feature: one chronological Elo pass over the same
    # restricted window. Each TRAINING game's feature is its PRE-game Elo gap
    # (leak-free; see teams.py); the artifact ships the POST-pass "as of
    # today" ratings for the frontend's optional team inputs.
    team_elo_diffs: dict[str, float] = {}
    elo_result = None
    if config.use_team_feature:
        elo_result = teams.compute_team_elo(
            games_df,
            k=config.elo_k,
            season_carryover=config.elo_season_carryover,
            international_leagues=config.international_leagues,
            international_k_multiplier=config.international_elo_k_multiplier,
            game_margins=game_margins,
            mov_scale=config.elo_mov_scale,
        )
        team_elo_diffs = teams.elo_diff_by_gameid(
            elo_result, feature_scale=config.elo_feature_scale
        )

    # Player-level features (players.py): same chronological pre-game
    # construction, riding on the same optional team selection. The artifact
    # ships each team's current roster (its most recent lineup) with per
    # -player Elo / winrate / champion records so the frontend reproduces
    # both features exactly when teams are picked -- and stays draft-only
    # when they aren't.
    player_stats = None
    if config.use_team_feature and config.use_player_features:
        player_stats = players.compute_player_stats(
            games_df,
            elo_feature_scale=config.elo_feature_scale,
            k=config.player_elo_k,
            season_carryover=config.player_season_carryover,
            prof_shrink=config.prof_shrink_games,
        )

    model_result = train_model.train_model(
        games_df, champion_strength, synergy_residuals, matchup_residuals,
        champion_presence, team_elo_diffs, loo_score_diffs,
        player_stats.player_elo_diff if player_stats else None,
        player_stats.prof_diff if player_stats else None,
    )

    champion_ratings_payload = _build_champion_ratings_payload(
        champion_features_df, resolved_patch, patches_used, config
    )
    model_payload = _build_model_payload(model_result)
    synergy_payload = _build_synergy_payload(
        synergy_table, matchup_table, resolved_patch, patches_used, config
    )
    teams_payload = _build_teams_payload(elo_result, resolved_patch, config, player_stats)
    players_payload = _build_players_payload(player_stats, resolved_patch, config)

    return (
        champion_ratings_payload,
        model_payload,
        synergy_payload,
        teams_payload,
        players_payload,
    )


def soloqueue_source_for(source: str, soloqueue_fixture: str):
    """Pick the solo-queue prior source.

    Only the offline ``fixture`` source uses the bundled StaticSoloQueueSource
    (synthetic solo-queue numbers). On the REAL data paths
    (``oracles-elixir``, ``leaguepedia``) we have no reachable real solo-queue
    provider, so we use ``NullSoloQueueSource`` -- an honest neutral 50% prior
    rather than blending in synthetic numbers (which the owner explicitly
    doesn't want). Champions with sparse pro data therefore shrink toward 50%,
    which is the correct data-free behavior. (Wiring a real solo-queue adapter
    is tracked in the README roadmap.)
    """
    if source == "fixture":
        return StaticSoloQueueSource(soloqueue_fixture)
    return NullSoloQueueSource()


def run_pipeline(
    oracles_elixir_path: str,
    oracles_elixir_url: str | None,
    soloqueue_fixture: str,
    patch: str | None,
    config: PipelineConfig = DEFAULT_CONFIG,
    source: str = "fixture",
    year: int | None = None,
) -> tuple[dict, dict, dict, dict, dict]:
    """Fetch from ``source`` and run the full pipeline. See
    :func:`run_pipeline_on_data` for the reusable, already-fetched-data core
    (used by ``main()`` so the backtest doesn't re-fetch).
    """
    raw_games_df, raw_bans_df, game_margins = load_raw_games_and_bans(
        source, oracles_elixir_path, oracles_elixir_url, config.target_training_games, year
    )
    soloqueue_source = soloqueue_source_for(source, soloqueue_fixture)
    return run_pipeline_on_data(
        raw_games_df, raw_bans_df, soloqueue_source, patch, config, game_margins
    )


def _most_recent_patch(games_df: pd.DataFrame) -> str:
    # The "current patch" is the numerically-newest patch present, NOT the
    # patch of whatever row has the latest timestamp: regions move patches on
    # different days, so an older patch can carry a later-dated game. Using the
    # newest patch number keeps the displayed patch and solo-queue lookup key
    # aligned with the newest patch actually in the data (see
    # features.newest_patch / _patches_by_recency).
    newest = features.newest_patch(games_df)
    if newest is not None:
        return str(newest)
    # Degenerate fallback (no parseable patches at all): latest-dated row.
    latest_date = games_df["date"].max()
    latest_rows = games_df[games_df["date"] == latest_date]
    return str(latest_rows["patch"].iloc[0])


def _build_champion_ratings_payload(
    champion_features_df: pd.DataFrame,
    patch: str,
    patches_used: list[str],
    config: PipelineConfig,
) -> dict:
    champions_payload = {}
    for champion, row in champion_features_df.iterrows():
        champions_payload[champion] = {
            "primaryRole": row["primaryRole"],
            "proGames": int(row["proGames"]),
            "proWinRate": float(row["proWinRate"]),
            "soloGames": int(row["soloGames"]),
            "soloWinRate": float(row["soloWinRate"]),
            "blendedWinRate": float(row["blendedWinRate"]),
            "strengthScore": float(row["strengthScore"]),
            "pickRate": float(row["pickRate"]),
            "banRate": float(row["banRate"]),
            "sampleConfidence": row["sampleConfidence"],
        }

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "patch": patch,
        "patchesUsed": list(patches_used),
        "params": {
            "patchHalfLifeDays": config.patch_half_life_days,
            "patchDecayBase": config.patch_decay_base,
            "soloQueueWeight": config.solo_queue_weight,
            "priorGames": config.prior_games,
            "proWindowDays": config.pro_window_days,
        },
        "globalMean": config.global_mean,
        "champions": champions_payload,
    }


def _build_model_payload(model_result: train_model.ModelResult) -> dict:
    return {
        "version": 1,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "trainingGames": model_result.training_games,
        "coefficients": model_result.coefficients,
        "metrics": model_result.metrics,
    }


def _build_synergy_payload(
    synergy_table: dict[str, pairwise.PairStat],
    matchup_table: dict[str, pairwise.PairStat],
    patch: str,
    patches_used: list[str],
    config: PipelineConfig,
) -> dict:
    synergy_payload = {
        key: {"gamesDecayed": stat.games_decayed, "residual": stat.residual}
        for key, stat in synergy_table.items()
    }
    matchup_payload = {
        key: {"gamesDecayed": stat.games_decayed, "residual": stat.residual}
        for key, stat in matchup_table.items()
    }

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "patch": patch,
        "patchesUsed": list(patches_used),
        "params": {
            "synergyPriorGames": config.synergy_prior_games,
            "matchupPriorGames": config.matchup_prior_games,
        },
        "synergy": synergy_payload,
        "matchup": matchup_payload,
    }


def _build_teams_payload(
    elo_result, patch: str, config: PipelineConfig, player_stats=None
) -> dict:
    """``data/teams.json``: per-team Elo "as of today" for the frontend's
    optional team inputs. Empty ``teams`` when the feature is disabled or no
    team names were present in the data.

    When player features are on, each team also carries its CURRENT roster --
    the five (player, position) seats from its most recent game. The players'
    per-player stats (Elo, record, per-champion record) live in the separate
    ``players.json`` index (see ``_build_players_payload``) so the frontend
    can look up ANY player -- including a substitute the user edits in, not
    just the current starters -- to compute playerEloDiff/profDiff exactly
    like the pipeline (players.py / apps/web/lib/predict.ts)."""
    teams_payload = {}
    if elo_result is not None:
        for team, elo in sorted(elo_result.ratings.items()):
            entry = {
                "elo": float(elo),
                "games": int(elo_result.games_played.get(team, 0)),
                "league": elo_result.last_league.get(team, ""),
                "lastPlayed": elo_result.last_played.get(team, ""),
            }
            if player_stats is not None and team in player_stats.rosters:
                entry["roster"] = [
                    {"player": player, "position": position}
                    for player, position in player_stats.rosters[team]
                ]
            teams_payload[team] = entry
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "patch": patch,
        "params": {
            "eloK": config.elo_k,
            # The FEATURE divisor (regularization knob), which is what the
            # frontend must divide the Elo gap by -- not Elo's own 400 curve.
            "eloScale": config.elo_feature_scale,
            "initialElo": teams.INITIAL_ELO,
            # Pseudo-games shrinking a player-champion winrate toward the
            # player's own overall winrate (players.proficiency); the
            # frontend must use the same value to reproduce profDiff.
            "profShrink": config.prof_shrink_games,
        },
        "teams": teams_payload,
    }


# A player needs at least this many games in the window to appear in the
# players index on their own (roster starters are always included regardless).
# Filters one-off appearances that would only clutter the picker; anyone with
# a non-trivial pro sample -- including subs and academy call-ups -- is kept.
MIN_PLAYER_INDEX_GAMES = 10
# Per-champion records below this many games are dropped from a player's
# proficiency table: with prof_shrink=8 a 1-game sample barely moves the
# shrunk edge, and keeping them all roughly doubles the artifact size.
MIN_PLAYER_CHAMP_GAMES = 2


def _build_players_payload(player_stats, patch: str, config: PipelineConfig) -> dict:
    """``data/players.json``: a global index of every pro player's "as of
    today" Elo, overall record, and per-champion record. This is what powers
    the OPTIONAL, editable player inputs: teams.json ships each team's current
    roster as names, and this index provides the stats for those names AND for
    any substitute the user swaps in. Empty when player features are off."""
    players_payload: dict = {}
    if player_stats is None:
        return {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "patch": patch,
            "params": {"profShrink": config.prof_shrink_games},
            "players": players_payload,
        }

    champs_by_player: dict[str, dict] = {}
    for (player, champ), (w, g) in player_stats.champ.items():
        if g >= MIN_PLAYER_CHAMP_GAMES:
            champs_by_player.setdefault(player, {})[champ] = [int(w), int(g)]

    # Always include current roster starters, even below the games threshold.
    roster_players = {
        player for seats in player_stats.rosters.values() for player, _ in seats
    }
    for player, (wins, games) in player_stats.wr.items():
        if not player:
            continue
        if games < MIN_PLAYER_INDEX_GAMES and player not in roster_players:
            continue
        players_payload[player] = {
            "elo": float(player_stats.elo.get(player, players.INITIAL_ELO)),
            "wins": int(wins),
            "games": int(games),
            "champions": champs_by_player.get(player, {}),
        }

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "patch": patch,
        "params": {"profShrink": config.prof_shrink_games},
        "players": players_payload,
    }


def write_outputs(
    champion_ratings: dict,
    model: dict,
    synergy: dict,
    output_dir: str,
    teams_payload: dict | None = None,
    players_payload: dict | None = None,
) -> tuple[Path, Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings_path = out_dir / "champion_ratings.json"
    model_path = out_dir / "model.json"
    synergy_path = out_dir / "synergy.json"

    with ratings_path.open("w", encoding="utf-8") as f:
        json.dump(champion_ratings, f, indent=2, sort_keys=False)
        f.write("\n")

    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, sort_keys=False)
        f.write("\n")

    with synergy_path.open("w", encoding="utf-8") as f:
        json.dump(synergy, f, indent=2, sort_keys=False)
        f.write("\n")

    if teams_payload is not None:
        teams_path = out_dir / "teams.json"
        with teams_path.open("w", encoding="utf-8") as f:
            json.dump(teams_payload, f, separators=(",", ":"), sort_keys=False)
            f.write("\n")

    if players_payload is not None:
        players_path = out_dir / "players.json"
        with players_path.open("w", encoding="utf-8") as f:
            # Compact separators, no indent: the players index is the largest
            # artifact (every player's per-champion records), and
            # pretty-printing it roughly doubles the bytes shipped.
            json.dump(players_payload, f, separators=(",", ":"), sort_keys=False)
            f.write("\n")

    return ratings_path, model_path, synergy_path


def _degraded_backtest_report(exc: Exception) -> dict:
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
        "note": f"Backtest could not be run: {exc!r}",
    }


def write_backtest_report_on_data(
    raw_games_df: pd.DataFrame,
    raw_bans_df: pd.DataFrame,
    soloqueue_source,
    output_dir: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    game_margins: dict[str, float] | None = None,
) -> Path:
    """Run the walk-forward backtest on already-fetched data and write
    ``data/backtest_report.json``. Degrades gracefully: any exception (e.g.
    too little data) is caught and results in a report with an explanatory
    ``note`` rather than crashing the overall build.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "backtest_report.json"

    try:
        games_df, bans_df = etl.build_raw_tables(raw_games_df, raw_bans_df)
        solo_history = soloqueue_module.load_solo_history(
            REPO_ROOT / "data" / "soloqueue_history.json"
        )
        report = backtest.run_backtest(
            games_df, bans_df, soloqueue_source, config,
            solo_history=solo_history, game_margins=game_margins,
        )
    except Exception as exc:  # noqa: BLE001 - build must never crash on backtest failure
        warnings.warn(f"Backtest failed, writing a degraded report instead: {exc!r}")
        report = _degraded_backtest_report(exc)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")

    return report_path


def write_backtest_report(
    oracles_elixir_path: str,
    oracles_elixir_url: str | None,
    soloqueue_fixture: str,
    output_dir: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    source: str = "fixture",
) -> Path:
    """Fetch from ``source`` and run the backtest. See
    :func:`write_backtest_report_on_data` for the reusable,
    already-fetched-data core (used by ``main()`` so this doesn't re-fetch).
    """
    raw_games_df, raw_bans_df, game_margins = load_raw_games_and_bans(
        source, oracles_elixir_path, oracles_elixir_url, config.target_training_games
    )
    soloqueue_source = soloqueue_source_for(source, soloqueue_fixture)
    return write_backtest_report_on_data(
        raw_games_df, raw_bans_df, soloqueue_source, output_dir, config, game_margins
    )


def main(argv: list[str] | None = None) -> None:
    import dataclasses

    args = parse_args(argv)
    config = (
        dataclasses.replace(DEFAULT_CONFIG, target_training_games=args.target_games)
        if args.target_games
        else DEFAULT_CONFIG
    )

    # Fetch once and reuse for both the live build and the backtest -- for
    # a real source (oracles-elixir / leaguepedia) this avoids repeating the
    # download/network round-trips.
    raw_games_df, raw_bans_df, game_margins = load_raw_games_and_bans(
        args.source,
        args.oracles_elixir_path,
        args.oracles_elixir_url,
        config.target_training_games,
        args.year,
    )
    soloqueue_source = soloqueue_source_for(args.source, args.soloqueue_fixture)

    champion_ratings, model, synergy, teams_payload, players_payload = run_pipeline_on_data(
        raw_games_df, raw_bans_df, soloqueue_source, args.patch, config, game_margins
    )

    ratings_path, model_path, synergy_path = write_outputs(
        champion_ratings, model, synergy, args.output_dir, teams_payload, players_payload
    )
    print(
        f"Wrote {Path(args.output_dir) / 'teams.json'} "
        f"({len(teams_payload['teams'])} teams, eloK={teams_payload['params']['eloK']})"
    )
    print(
        f"Wrote {Path(args.output_dir) / 'players.json'} "
        f"({len(players_payload['players'])} players)"
    )

    n_champions = len(champion_ratings["champions"])
    print(f"Wrote {ratings_path} ({n_champions} champions, patch={champion_ratings['patch']!r})")
    print(f"  patchesUsed={champion_ratings['patchesUsed']!r}")
    print(
        f"Wrote {model_path} (trainingGames={model['trainingGames']}, "
        f"accuracy={model['metrics'].get('accuracy')!r})"
    )
    print(
        f"Wrote {synergy_path} (synergy pairs={len(synergy['synergy'])}, "
        f"matchup pairs={len(synergy['matchup'])})"
    )

    report_path = write_backtest_report_on_data(
        raw_games_df, raw_bans_df, soloqueue_source, args.output_dir, config, game_margins
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
