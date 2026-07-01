"""Tests for the walk-forward backtest module (backtest.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from compstrength_pipeline import backtest, etl
from compstrength_pipeline.config import DEFAULT_CONFIG
from compstrength_pipeline.sources.oracles_elixir import extract_bans, fetch_oracles_elixir
from compstrength_pipeline.sources.soloqueue import StaticSoloQueueSource

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ORACLES_ELIXIR_FIXTURE = FIXTURES_DIR / "sample_oracleselixir.csv"
SOLOQUEUE_FIXTURE = FIXTURES_DIR / "sample_soloqueue.json"


@pytest.fixture(scope="module")
def cleaned_games_and_bans():
    raw_games = fetch_oracles_elixir(str(ORACLES_ELIXIR_FIXTURE))
    raw = pd.read_csv(ORACLES_ELIXIR_FIXTURE, low_memory=False)
    raw_bans = extract_bans(raw)
    return etl.build_raw_tables(raw_games, raw_bans)


@pytest.fixture(scope="module")
def solo_source():
    return StaticSoloQueueSource(str(SOLOQUEUE_FIXTURE))


REQUIRED_TOP_LEVEL_KEYS = {
    "generatedAt", "folds", "testGames", "metrics", "calibration", "note"
}
REQUIRED_METRICS_KEYS = {
    "accuracy", "logLoss", "brierScore", "baselineAccuracy", "coinFlipLogLoss"
}
REQUIRED_CALIBRATION_KEYS = {"bucket", "predictedMean", "actualWinRate", "count"}


def test_run_backtest_end_to_end_produces_expected_schema(cleaned_games_and_bans, solo_source):
    games_df, bans_df = cleaned_games_and_bans

    report = backtest.run_backtest(games_df, bans_df, solo_source, DEFAULT_CONFIG)

    assert REQUIRED_TOP_LEVEL_KEYS.issubset(report.keys())
    assert REQUIRED_METRICS_KEYS.issubset(report["metrics"].keys())

    assert report["folds"] >= 2
    assert report["testGames"] > 0
    assert isinstance(report["calibration"], list)
    for row in report["calibration"]:
        assert REQUIRED_CALIBRATION_KEYS.issubset(row.keys())
        assert row["count"] > 0

    # Metrics should be finite numbers, not NaN, given the fixture has
    # enough data to run at least the minimum fold count.
    import math

    for key in REQUIRED_METRICS_KEYS:
        assert math.isfinite(report["metrics"][key]), f"{key} should be finite"

    assert 0.0 <= report["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= report["metrics"]["baselineAccuracy"] <= 1.0
    assert report["metrics"]["logLoss"] >= 0.0
    assert report["metrics"]["brierScore"] >= 0.0
    # coin-flip log-loss should be exactly ln(2) regardless of data.
    assert report["metrics"]["coinFlipLogLoss"] == pytest.approx(0.6931471805599453, abs=1e-6)

    assert isinstance(report["note"], str) and len(report["note"]) > 0

    # Per-segment breakdowns + data composition (drive the Methodology page).
    assert "breakdowns" in report and "dataComposition" in report
    for seg in ("byPatch", "byLeague"):
        assert isinstance(report["breakdowns"][seg], list)
        for row in report["breakdowns"][seg]:
            assert {"name", "testGames", "accuracy", "baselineAccuracy", "logLoss"}.issubset(row.keys())
            assert row["testGames"] > 0
            assert 0.0 <= row["accuracy"] <= 1.0
    comp = report["dataComposition"]
    assert comp["totalGames"] > 0
    assert isinstance(comp["byPatch"], list) and isinstance(comp["byLeague"], list)
    for row in comp["byPatch"]:
        assert {"name", "games", "recencyWeight"}.issubset(row.keys())
        assert row["games"] > 0
        assert 0.0 < row["recencyWeight"] <= 1.0
    # The composition should account for exactly the games used (sum over
    # patches == total).
    assert sum(r["games"] for r in comp["byPatch"]) == comp["totalGames"]


def test_run_backtest_does_not_crash_with_too_little_data(solo_source):
    """A tiny dataset (below the minimum-games threshold) should degrade
    gracefully to an explanatory report, not raise."""
    tiny_rows = []
    for i in range(2):
        for side, champ, result in (("Blue", "A", 1), ("Red", "B", 0)):
            tiny_rows.append(
                {
                    "gameid": f"g{i}",
                    "date": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i),
                    "patch": "14.1",
                    "league": "LCK",
                    "side": side,
                    "team": f"Team{side}",
                    "player": f"Player{side}",
                    "position": "top",
                    "champion": champ,
                    "result": result,
                }
            )
    tiny_games_df = pd.DataFrame(tiny_rows)
    tiny_bans_df = pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])

    report = backtest.run_backtest(tiny_games_df, tiny_bans_df, solo_source, DEFAULT_CONFIG)

    assert REQUIRED_TOP_LEVEL_KEYS.issubset(report.keys())
    assert report["folds"] == 0
    assert report["testGames"] == 0
    assert "note" in report and len(report["note"]) > 0


def test_run_backtest_empty_games_df_does_not_crash(solo_source):
    empty = pd.DataFrame(columns=["gameid", "date", "patch", "side", "position", "champion", "result"])
    empty_bans = pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])
    report = backtest.run_backtest(empty, empty_bans, solo_source, DEFAULT_CONFIG)
    assert report["folds"] == 0
    assert report["testGames"] == 0


def test_backtest_main_writes_report(tmp_path, cleaned_games_and_bans):
    argv = [
        "--oracles-elixir-path", str(ORACLES_ELIXIR_FIXTURE),
        "--soloqueue-fixture", str(SOLOQUEUE_FIXTURE),
        "--output-dir", str(tmp_path),
    ]
    backtest.main(argv)

    report_path = tmp_path / "backtest_report.json"
    assert report_path.exists()

    import json

    with report_path.open() as f:
        data = json.load(f)
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(data.keys())
    assert data["folds"] >= 2
