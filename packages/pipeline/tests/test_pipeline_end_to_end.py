"""End-to-end test: runs the full build pipeline against the bundled fixtures
and asserts the two output JSON files are written with the correct schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compstrength_pipeline import build

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ORACLES_ELIXIR_FIXTURE = FIXTURES_DIR / "sample_oracleselixir.csv"
SOLOQUEUE_FIXTURE = FIXTURES_DIR / "sample_soloqueue.json"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "data_out"


def test_pipeline_end_to_end_writes_expected_files(output_dir: Path):
    champion_ratings, model = build.run_pipeline(
        oracles_elixir_path=str(ORACLES_ELIXIR_FIXTURE),
        oracles_elixir_url=None,
        soloqueue_fixture=str(SOLOQUEUE_FIXTURE),
        patch=None,
    )

    ratings_path, model_path = build.write_outputs(champion_ratings, model, str(output_dir))

    assert ratings_path.exists()
    assert model_path.exists()

    with ratings_path.open() as f:
        ratings_data = json.load(f)
    with model_path.open() as f:
        model_data = json.load(f)

    # champion_ratings.json top-level schema
    for key in ("generatedAt", "patch", "params", "globalMean", "champions"):
        assert key in ratings_data
    for key in ("patchHalfLifeDays", "soloQueueWeight", "priorGames", "proWindowDays"):
        assert key in ratings_data["params"]
    assert isinstance(ratings_data["champions"], dict)
    assert len(ratings_data["champions"]) > 0

    sample_champion = next(iter(ratings_data["champions"].values()))
    for key in (
        "primaryRole", "proGames", "proWinRate", "soloGames", "soloWinRate",
        "blendedWinRate", "strengthScore", "pickRate", "banRate", "sampleConfidence",
    ):
        assert key in sample_champion

    # model.json top-level schema
    for key in ("version", "trainedAt", "trainingGames", "coefficients", "metrics"):
        assert key in model_data
    for key in ("scoreDiffWeight", "blueSideBias", "intercept"):
        assert key in model_data["coefficients"]
    for key in ("logLoss", "accuracy", "baselineAccuracy"):
        assert key in model_data["metrics"]

    assert model_data["trainingGames"] > 0


def test_pipeline_main_writes_to_output_dir(output_dir: Path):
    build.main(
        [
            "--oracles-elixir-path",
            str(ORACLES_ELIXIR_FIXTURE),
            "--soloqueue-fixture",
            str(SOLOQUEUE_FIXTURE),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "champion_ratings.json").exists()
    assert (output_dir / "model.json").exists()
