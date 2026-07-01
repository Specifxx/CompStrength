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

from compstrength_pipeline import backtest, etl, features, pairwise, train_model
from compstrength_pipeline.config import DEFAULT_CONFIG, PipelineConfig
from compstrength_pipeline.sources import leaguepedia as leaguepedia_source
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
        choices=["fixture", "leaguepedia"],
        default=os.environ.get("COMPSTRENGTH_SOURCE", "fixture"),
        help=(
            "Which pro-match data source to use. 'fixture' (default) reads "
            "--oracles-elixir-path/--oracles-elixir-url (a bundled synthetic "
            "fixture unless one of those is overridden). 'leaguepedia' "
            "ignores those and fetches real, current data live from "
            "Leaguepedia's Cargo API instead (requires unblocked network "
            "egress to lol.fandom.com, e.g. GitHub Actions)."
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
        default=int(os.environ.get("COMPSTRENGTH_TARGET_GAMES", "0")) or None,
        help=(
            "Override PipelineConfig.target_training_games (default 1000). "
            "Useful for a quick, fast --source leaguepedia smoke test with a "
            "small number before committing to a full-size live fetch."
        ),
    )
    return parser.parse_args(argv)


def load_games_and_bans(
    oracles_elixir_path: str, oracles_elixir_url: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Oracle's Elixir data (live URL if given, else local path/fixture)."""
    source = oracles_elixir_url or oracles_elixir_path
    games_df = fetch_oracles_elixir(source)

    # extract_bans() needs the raw (un-normalized) frame including team rows,
    # so we re-read it directly rather than reusing the normalized games_df.
    if oracles_elixir_url:
        # Best-effort: re-fetch raw for ban extraction. In a live environment
        # this is a second (cheap, cached-by-CDN) request; failures here are
        # non-fatal to the overall pipeline (bans just come back empty).
        try:
            import io

            import requests

            resp = requests.get(oracles_elixir_url, timeout=30)
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        except Exception as exc:  # pragma: no cover - network path
            warnings.warn(f"Could not fetch raw data for ban extraction: {exc!r}")
            raw = pd.DataFrame()
    else:
        raw = pd.read_csv(oracles_elixir_path, low_memory=False)

    bans_df = extract_bans(raw) if not raw.empty else pd.DataFrame(
        columns=["gameid", "team", "champion", "ban_number"]
    )
    return games_df, bans_df


def load_raw_games_and_bans(
    source: str,
    oracles_elixir_path: str,
    oracles_elixir_url: str | None,
    target_games: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch to the requested data source's raw fetch.

    ``source == "leaguepedia"`` fetches real, current data live from
    Leaguepedia's Cargo API (see ``sources/leaguepedia.py``); anything else
    falls back to the Oracle's-Elixir-shaped fixture/URL path (default:
    the bundled synthetic fixture).
    """
    if source == "leaguepedia":
        return leaguepedia_source.fetch_recent_games(target_games=target_games)
    return load_games_and_bans(oracles_elixir_path, oracles_elixir_url)


def run_pipeline_on_data(
    raw_games_df: pd.DataFrame,
    raw_bans_df: pd.DataFrame,
    soloqueue_source,
    patch: str | None,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[dict, dict, dict]:
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

    # Hard-restrict to the most recent `target_training_games` games up
    # front (regardless of patch) so every downstream computation (resolved
    # patch, champion features, pairwise synergy/matchup, and the training
    # frame for the model) operates on exactly the same window.
    games_df, bans_df, patches_used = features.restrict_to_recent_games(
        games_df, bans_df, config.target_training_games
    )
    if games_df.empty:
        raise ValueError(
            "No games remain after restricting to the most recent games; "
            "cannot compute features. Check the input data source."
        )

    resolved_patch = patch or _most_recent_patch(games_df)
    solo_winrates = soloqueue_source.get_champion_winrates(resolved_patch)

    reference_date = games_df["date"].max()
    champion_features_df = features.compute_champion_features(
        games_df=games_df,
        bans_df=bans_df,
        solo_winrates=solo_winrates,
        config=config,
        reference_date=reference_date,
    )

    champion_strength = champion_features_df["strengthScore"].to_dict()

    synergy_table = pairwise.compute_synergy_table(
        games_df, champion_strength, config, reference_date=reference_date
    )
    matchup_table = pairwise.compute_matchup_table(
        games_df, champion_strength, config, reference_date=reference_date
    )
    synergy_residuals = pairwise.synergy_lookup(synergy_table)
    matchup_residuals = pairwise.matchup_lookup(matchup_table)

    model_result = train_model.train_model(
        games_df, champion_strength, synergy_residuals, matchup_residuals
    )

    champion_ratings_payload = _build_champion_ratings_payload(
        champion_features_df, resolved_patch, patches_used, config
    )
    model_payload = _build_model_payload(model_result)
    synergy_payload = _build_synergy_payload(
        synergy_table, matchup_table, resolved_patch, patches_used, config
    )

    return champion_ratings_payload, model_payload, synergy_payload


def run_pipeline(
    oracles_elixir_path: str,
    oracles_elixir_url: str | None,
    soloqueue_fixture: str,
    patch: str | None,
    config: PipelineConfig = DEFAULT_CONFIG,
    source: str = "fixture",
) -> tuple[dict, dict, dict]:
    """Fetch from ``source`` and run the full pipeline. See
    :func:`run_pipeline_on_data` for the reusable, already-fetched-data core
    (used by ``main()`` so the backtest doesn't re-fetch).
    """
    raw_games_df, raw_bans_df = load_raw_games_and_bans(
        source, oracles_elixir_path, oracles_elixir_url, config.target_training_games
    )
    soloqueue_source = (
        NullSoloQueueSource() if source == "leaguepedia" else StaticSoloQueueSource(soloqueue_fixture)
    )
    return run_pipeline_on_data(raw_games_df, raw_bans_df, soloqueue_source, patch, config)


def _most_recent_patch(games_df: pd.DataFrame) -> str:
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


def write_outputs(
    champion_ratings: dict, model: dict, synergy: dict, output_dir: str
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
        "note": f"Backtest could not be run: {exc!r}",
    }


def write_backtest_report_on_data(
    raw_games_df: pd.DataFrame,
    raw_bans_df: pd.DataFrame,
    soloqueue_source,
    output_dir: str,
    config: PipelineConfig = DEFAULT_CONFIG,
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
        report = backtest.run_backtest(games_df, bans_df, soloqueue_source, config)
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
    raw_games_df, raw_bans_df = load_raw_games_and_bans(
        source, oracles_elixir_path, oracles_elixir_url, config.target_training_games
    )
    soloqueue_source = (
        NullSoloQueueSource() if source == "leaguepedia" else StaticSoloQueueSource(soloqueue_fixture)
    )
    return write_backtest_report_on_data(
        raw_games_df, raw_bans_df, soloqueue_source, output_dir, config
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
    # the `leaguepedia` source this avoids repeating several rate-limited
    # network round-trips.
    raw_games_df, raw_bans_df = load_raw_games_and_bans(
        args.source, args.oracles_elixir_path, args.oracles_elixir_url, config.target_training_games
    )
    soloqueue_source = (
        NullSoloQueueSource()
        if args.source == "leaguepedia"
        else StaticSoloQueueSource(args.soloqueue_fixture)
    )

    champion_ratings, model, synergy = run_pipeline_on_data(
        raw_games_df, raw_bans_df, soloqueue_source, args.patch, config
    )

    ratings_path, model_path, synergy_path = write_outputs(
        champion_ratings, model, synergy, args.output_dir
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
        raw_games_df, raw_bans_df, soloqueue_source, args.output_dir, config
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
