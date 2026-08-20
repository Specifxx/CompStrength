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
    champion_ratings, model, synergy, teams_payload, players_payload = build.run_pipeline(
        oracles_elixir_path=str(ORACLES_ELIXIR_FIXTURE),
        oracles_elixir_url=None,
        soloqueue_fixture=str(SOLOQUEUE_FIXTURE),
        patch=None,
    )

    ratings_path, model_path, synergy_path = build.write_outputs(
        champion_ratings, model, synergy, str(output_dir), teams_payload, players_payload
    )

    assert ratings_path.exists()
    assert model_path.exists()
    assert synergy_path.exists()

    # teams.json schema
    teams_path = output_dir / "teams.json"
    assert teams_path.exists()
    with teams_path.open() as f:
        teams_data = json.load(f)
    for key in ("generatedAt", "patch", "params", "teams"):
        assert key in teams_data
    for key in ("eloK", "eloScale", "initialElo", "profShrink"):
        assert key in teams_data["params"]
    assert isinstance(teams_data["teams"], dict) and len(teams_data["teams"]) > 0
    sample_team = next(iter(teams_data["teams"].values()))
    for key in ("elo", "games", "league", "lastPlayed"):
        assert key in sample_team
    # Player features on by default: every team that played carries its
    # current roster as five (player, position) seats (stats live in the
    # separate players.json index).
    rostered = [t for t in teams_data["teams"].values() if "roster" in t]
    assert rostered, "expected at least one team with a roster"
    sample_seat = rostered[0]["roster"][0]
    assert set(sample_seat.keys()) == {"player", "position"}
    assert len(rostered[0]["roster"]) == 5

    # players.json: global player index with per-player stats. Every roster
    # player must be resolvable in it.
    players_path = output_dir / "players.json"
    assert players_path.exists()
    with players_path.open() as f:
        players_data = json.load(f)
    for key in ("generatedAt", "patch", "params", "players"):
        assert key in players_data
    assert "profShrink" in players_data["params"]
    assert isinstance(players_data["players"], dict) and players_data["players"]
    sample_player = next(iter(players_data["players"].values()))
    for key in ("elo", "wins", "games", "champions"):
        assert key in sample_player
    roster_names = {
        seat["player"] for t in rostered for seat in t["roster"]
    }
    missing = roster_names - set(players_data["players"])
    assert not missing, f"roster players missing from index: {missing}"
    # Elo is zero-sum around the initial rating, so the mean stays ~1500.
    elos = [t["elo"] for t in teams_data["teams"].values()]
    assert abs(sum(elos) / len(elos) - 1500.0) < 1.0

    with ratings_path.open() as f:
        ratings_data = json.load(f)
    with model_path.open() as f:
        model_data = json.load(f)
    with synergy_path.open() as f:
        synergy_data = json.load(f)

    # champion_ratings.json top-level schema
    for key in ("generatedAt", "patch", "patchesUsed", "params", "globalMean", "champions"):
        assert key in ratings_data
    for key in ("patchHalfLifeDays", "soloQueueWeight", "priorGames", "proWindowDays"):
        assert key in ratings_data["params"]
    assert isinstance(ratings_data["champions"], dict)
    assert len(ratings_data["champions"]) > 0
    assert isinstance(ratings_data["patchesUsed"], list)
    assert len(ratings_data["patchesUsed"]) > 0

    sample_champion = next(iter(ratings_data["champions"].values()))
    for key in (
        "primaryRole", "proGames", "proWinRate", "soloGames", "soloWinRate",
        "blendedWinRate", "strengthScore", "pickRate", "banRate", "sampleConfidence",
    ):
        assert key in sample_champion

    # model.json top-level schema
    for key in ("version", "trainedAt", "trainingGames", "coefficients", "metrics"):
        assert key in model_data
    for key in (
        "scoreDiffWeight", "synergyWeight", "matchupWeight", "presenceWeight",
        "teamEloWeight", "playerEloWeight", "profWeight",
        "blueSideBias", "intercept",
    ):
        assert key in model_data["coefficients"]
        assert key in model_data["draftOnlyCoefficients"]
    # The dedicated draft-only fit deliberately EXCLUDES synergy (measured
    # harmful without team features present) and all team/player terms.
    for frozen in ("synergyWeight", "teamEloWeight", "playerEloWeight", "profWeight"):
        assert model_data["draftOnlyCoefficients"][frozen] == 0.0
    for key in ("logLoss", "accuracy", "baselineAccuracy"):
        assert key in model_data["metrics"]

    assert model_data["trainingGames"] > 0

    # synergy.json top-level schema
    for key in ("generatedAt", "patch", "patchesUsed", "params", "synergy", "matchup"):
        assert key in synergy_data
    for key in ("synergyPriorGames", "matchupPriorGames"):
        assert key in synergy_data["params"]
    assert synergy_data["patchesUsed"] == ratings_data["patchesUsed"]
    assert isinstance(synergy_data["synergy"], dict)
    assert isinstance(synergy_data["matchup"], dict)
    if synergy_data["synergy"]:
        sample_pair = next(iter(synergy_data["synergy"].values()))
        assert "gamesDecayed" in sample_pair
        assert "residual" in sample_pair
    if synergy_data["matchup"]:
        sample_matchup = next(iter(synergy_data["matchup"].values()))
        assert "gamesDecayed" in sample_matchup
        assert "residual" in sample_matchup


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
    assert (output_dir / "synergy.json").exists()
    assert (output_dir / "backtest_report.json").exists()

    with (output_dir / "backtest_report.json").open() as f:
        backtest_data = json.load(f)
    for key in ("generatedAt", "folds", "testGames", "metrics", "calibration", "note"):
        assert key in backtest_data
    for key in ("accuracy", "logLoss", "brierScore", "baselineAccuracy", "coinFlipLogLoss"):
        assert key in backtest_data["metrics"]


def test_oracles_elixir_source_merges_previous_season(monkeypatch, tmp_path):
    """The oracles-elixir source fetches the current AND previous season and
    concatenates them (previous season's duplicate gameids dropped)."""
    import pandas as pd

    from compstrength_pipeline.sources import oracles_elixir as oe

    # Build a doctored "previous season" fixture: same shape, distinct gameids.
    raw = pd.read_csv(ORACLES_ELIXIR_FIXTURE, **oe._READ_CSV_KWARGS)
    prev_raw = raw.copy()
    prev_raw["gameid"] = "prev_" + prev_raw["gameid"].astype(str)
    prev_path = tmp_path / "prev.csv"
    prev_raw.to_csv(prev_path, index=False)

    fetched_years = []

    def fake_download(year, dest=None):
        fetched_years.append(year)
        return prev_path if len(fetched_years) > 1 else ORACLES_ELIXIR_FIXTURE

    monkeypatch.setattr(oe, "download_oracles_elixir_csv", fake_download)

    games_df, bans_df, _margins = build.load_raw_games_and_bans(
        "oracles-elixir", str(ORACLES_ELIXIR_FIXTURE), None, target_games=1000
    )

    # Both seasons fetched: newest year first, then year-1.
    assert len(fetched_years) == 2
    assert fetched_years[0] - 1 == fetched_years[1]
    # Merged: every fixture game appears twice (once per pseudo-season).
    single_games, _, _ = build.load_games_and_bans(str(ORACLES_ELIXIR_FIXTURE), None)
    assert games_df["gameid"].nunique() == 2 * single_games["gameid"].nunique()
    assert bans_df["gameid"].nunique() > 0


def test_oracles_elixir_source_drops_gameids_duplicated_across_seasons(
    monkeypatch, tmp_path
):
    """If the previous season's file overlaps the current one, the overlap is
    dropped (else 20-row 'games' would be silently discarded by ETL)."""
    from compstrength_pipeline.sources import oracles_elixir as oe

    def fake_download(year, dest=None):
        return ORACLES_ELIXIR_FIXTURE  # same file for both years -> full overlap

    monkeypatch.setattr(oe, "download_oracles_elixir_csv", fake_download)

    games_df, _, _margins = build.load_raw_games_and_bans(
        "oracles-elixir", str(ORACLES_ELIXIR_FIXTURE), None, target_games=1000
    )
    single_games, _, _ = build.load_games_and_bans(str(ORACLES_ELIXIR_FIXTURE), None)
    # Full overlap -> merged result identical to a single season, and every
    # game still has exactly 10 player rows.
    assert games_df["gameid"].nunique() == single_games["gameid"].nunique()
    assert (games_df.groupby("gameid").size() == 10).all()


def test_oracles_elixir_source_survives_missing_previous_season(monkeypatch):
    """Previous-season download failure is non-fatal: proceed with one year."""
    from compstrength_pipeline.sources import oracles_elixir as oe

    calls = []

    def fake_download(year, dest=None):
        calls.append(year)
        if len(calls) > 1:
            raise oe.DataSourceUnavailableError("prev year unavailable")
        return ORACLES_ELIXIR_FIXTURE

    monkeypatch.setattr(oe, "download_oracles_elixir_csv", fake_download)

    with pytest.warns(UserWarning, match="previous season"):
        games_df, _, _margins = build.load_raw_games_and_bans(
            "oracles-elixir", str(ORACLES_ELIXIR_FIXTURE), None, target_games=1000
        )
    single_games, _, _ = build.load_games_and_bans(str(ORACLES_ELIXIR_FIXTURE), None)
    assert games_df["gameid"].nunique() == single_games["gameid"].nunique()


def test_parse_args_target_games_defaults_to_none_when_env_var_is_empty_string(monkeypatch):
    """GitHub Actions sets an unfilled workflow_dispatch string input to an
    empty string in the job env, not unset -- os.environ.get(key, "0") only
    falls back when the key is absent, so this must be handled explicitly
    rather than crashing on int('')."""
    monkeypatch.setenv("COMPSTRENGTH_TARGET_GAMES", "")
    args = build.parse_args([])
    assert args.target_games is None


def test_parse_args_target_games_reads_env_var_when_set(monkeypatch):
    monkeypatch.setenv("COMPSTRENGTH_TARGET_GAMES", "60")
    args = build.parse_args([])
    assert args.target_games == 60


def test_parse_args_target_games_cli_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("COMPSTRENGTH_TARGET_GAMES", "60")
    args = build.parse_args(["--target-games", "25"])
    assert args.target_games == 25
