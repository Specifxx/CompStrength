"""CLI entrypoint: run the full ETL -> features -> train_model pipeline and
write ``champion_ratings.json`` and ``model.json`` for the website to serve.

Usage:

    python -m compstrength_pipeline.build
    python -m compstrength_pipeline.build --oracles-elixir-path tests/fixtures/sample_oracleselixir.csv \\
        --soloqueue-fixture tests/fixtures/sample_soloqueue.json
    python -m compstrength_pipeline.build --oracles-elixir-url https://.../2026_LoL_esports_match_data...csv

By default, this points at the bundled small synthetic fixtures under
``tests/fixtures/`` so the pipeline can run fully offline (e.g. in this
sandbox, or for a quick local demo). Pass ``--oracles-elixir-url`` (or set
the ``ORACLES_ELIXIR_URL`` env var) to fetch live data instead -- this is
the mode intended for GitHub Actions, where network egress to
oracleselixir.com is not blocked.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from compstrength_pipeline import etl, features, train_model
from compstrength_pipeline.config import DEFAULT_CONFIG, PipelineConfig
from compstrength_pipeline.sources.oracles_elixir import (
    extract_bans,
    fetch_oracles_elixir,
)
from compstrength_pipeline.sources.soloqueue import StaticSoloQueueSource

PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_FIXTURE_CSV = PACKAGE_DIR / "tests" / "fixtures" / "sample_oracleselixir.csv"
DEFAULT_SOLOQUEUE_FIXTURE = PACKAGE_DIR / "tests" / "fixtures" / "sample_soloqueue.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def run_pipeline(
    oracles_elixir_path: str,
    oracles_elixir_url: str | None,
    soloqueue_fixture: str,
    patch: str | None,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[dict, dict]:
    """Run ETL -> features -> train_model and return the two output dicts.

    Returns:
        ``(champion_ratings_payload, model_payload)`` matching the exact
        JSON schemas documented in the project spec / README.
    """
    raw_games_df, raw_bans_df = load_games_and_bans(oracles_elixir_path, oracles_elixir_url)
    games_df, bans_df = etl.build_raw_tables(raw_games_df, raw_bans_df)

    if games_df.empty:
        raise ValueError(
            "No complete games survived ETL cleaning; cannot compute features. "
            "Check the input data source."
        )

    resolved_patch = patch or _most_recent_patch(games_df)

    soloqueue_source = StaticSoloQueueSource(soloqueue_fixture)
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
    model_result = train_model.train_model(games_df, champion_strength)

    champion_ratings_payload = _build_champion_ratings_payload(
        champion_features_df, resolved_patch, config
    )
    model_payload = _build_model_payload(model_result)

    return champion_ratings_payload, model_payload


def _most_recent_patch(games_df: pd.DataFrame) -> str:
    latest_date = games_df["date"].max()
    latest_rows = games_df[games_df["date"] == latest_date]
    return str(latest_rows["patch"].iloc[0])


def _build_champion_ratings_payload(
    champion_features_df: pd.DataFrame, patch: str, config: PipelineConfig
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


def write_outputs(champion_ratings: dict, model: dict, output_dir: str) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings_path = out_dir / "champion_ratings.json"
    model_path = out_dir / "model.json"

    with ratings_path.open("w", encoding="utf-8") as f:
        json.dump(champion_ratings, f, indent=2, sort_keys=False)
        f.write("\n")

    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, sort_keys=False)
        f.write("\n")

    return ratings_path, model_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    champion_ratings, model = run_pipeline(
        oracles_elixir_path=args.oracles_elixir_path,
        oracles_elixir_url=args.oracles_elixir_url,
        soloqueue_fixture=args.soloqueue_fixture,
        patch=args.patch,
    )

    ratings_path, model_path = write_outputs(champion_ratings, model, args.output_dir)

    n_champions = len(champion_ratings["champions"])
    print(f"Wrote {ratings_path} ({n_champions} champions, patch={champion_ratings['patch']!r})")
    print(
        f"Wrote {model_path} (trainingGames={model['trainingGames']}, "
        f"accuracy={model['metrics'].get('accuracy')!r})"
    )


if __name__ == "__main__":
    main()
