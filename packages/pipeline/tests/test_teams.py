"""Tests for the team Elo rating pass (teams.py): leak-freedom (pre-game
ratings only), chronological ordering, zero-sum updates, and the model
feature plumbing."""

from __future__ import annotations

import pandas as pd
import pytest

from compstrength_pipeline.teams import (
    DEFAULT_ELO_K,
    ELO_SCALE,
    INITIAL_ELO,
    compute_team_elo,
    elo_diff_by_gameid,
    expected_score,
)

POSITIONS = ["top", "jng", "mid", "bot", "sup"]


def _game(gameid, date, blue_team, red_team, blue_win, league="LCK"):
    rows = []
    for side, team, result in (
        ("Blue", blue_team, blue_win),
        ("Red", red_team, 1 - blue_win),
    ):
        for i, pos in enumerate(POSITIONS):
            rows.append(
                {
                    "gameid": gameid,
                    "date": pd.Timestamp(date, tz="UTC"),
                    "league": league,
                    "side": side,
                    "team": team,
                    "position": pos,
                    "champion": f"C{i}",
                    "result": result,
                }
            )
    return rows


def test_expected_score_symmetry_and_scale():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    # +400 Elo = 10:1 odds by definition of the scale.
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)
    assert expected_score(1500, 1900) + expected_score(1900, 1500) == pytest.approx(1.0)


def test_first_game_uses_initial_elo_pre_game():
    """Leak-freedom: the recorded feature for a game must be the rating
    BEFORE that game's result is applied."""
    games = pd.DataFrame(_game("g1", "2026-01-01", "T1", "GEN", blue_win=1))
    result = compute_team_elo(games)
    row = result.per_game.loc["g1"]
    assert row["blue_elo_pre"] == INITIAL_ELO
    assert row["red_elo_pre"] == INITIAL_ELO
    # ...and the POST ratings moved (winner up, loser down, zero-sum).
    assert result.ratings["T1"] > INITIAL_ELO > result.ratings["GEN"]
    assert result.ratings["T1"] + result.ratings["GEN"] == pytest.approx(2 * INITIAL_ELO)


def test_second_game_sees_first_games_result():
    games = pd.DataFrame(
        _game("g1", "2026-01-01", "T1", "GEN", blue_win=1)
        + _game("g2", "2026-01-02", "T1", "GEN", blue_win=1)
    )
    result = compute_team_elo(games)
    g2 = result.per_game.loc["g2"]
    assert g2["blue_elo_pre"] > INITIAL_ELO  # T1 already credited for g1
    assert g2["red_elo_pre"] < INITIAL_ELO
    # A win as the favorite moves the rating LESS than a 50/50 win.
    gain_g1 = DEFAULT_ELO_K * (1 - 0.5)
    gain_g2 = result.ratings["T1"] - g2["blue_elo_pre"]
    assert 0 < gain_g2 < gain_g1


def test_games_processed_in_date_order_not_input_order():
    # Feed the later game FIRST; the pass must still process chronologically,
    # so the 2026-01-01 game sees initial ratings.
    games = pd.DataFrame(
        _game("g_late", "2026-02-01", "T1", "GEN", blue_win=1)
        + _game("g_early", "2026-01-01", "T1", "GEN", blue_win=1)
    )
    result = compute_team_elo(games)
    assert result.per_game.loc["g_early"]["blue_elo_pre"] == INITIAL_ELO
    assert result.per_game.loc["g_late"]["blue_elo_pre"] > INITIAL_ELO


def test_strength_propagates_across_leagues():
    """The whole point of Elo over win rates: beating a strong team is worth
    more. A team that beats a proven winner ends up rated above a team that
    beat a proven loser."""
    games = pd.DataFrame(
        # T1 beats GEN twice -> T1 strong, GEN weak.
        _game("g1", "2026-01-01", "T1", "GEN", blue_win=1)
        + _game("g2", "2026-01-02", "T1", "GEN", blue_win=1)
        # X beats the strong T1; Y beats the weak GEN.
        + _game("g3", "2026-01-03", "X", "T1", blue_win=1)
        + _game("g4", "2026-01-03", "Y", "GEN", blue_win=1)
    )
    result = compute_team_elo(games)
    assert result.ratings["X"] > result.ratings["Y"]


def test_elo_diff_by_gameid_scaling_and_metadata():
    games = pd.DataFrame(
        _game("g1", "2026-01-01", "T1", "GEN", blue_win=1, league="LCK")
        + _game("g2", "2026-01-05", "T1", "GEN", blue_win=1, league="MSI")
    )
    result = compute_team_elo(games)
    diffs = elo_diff_by_gameid(result)
    assert diffs["g1"] == pytest.approx(0.0)  # both at initial Elo pre-game
    expected_g2 = (
        result.per_game.loc["g2"]["blue_elo_pre"] - result.per_game.loc["g2"]["red_elo_pre"]
    ) / ELO_SCALE
    assert diffs["g2"] == pytest.approx(expected_g2)
    assert diffs["g2"] > 0
    # Metadata reflects each team's most recent game.
    assert result.games_played["T1"] == 2
    assert result.last_league["T1"] == "MSI"
    assert result.last_played["T1"] == "2026-01-05"


def test_empty_and_teamless_games_handled():
    empty = pd.DataFrame(columns=["gameid", "date", "side", "team", "position", "champion", "result"])
    result = compute_team_elo(empty)
    assert result.ratings == {}
    assert elo_diff_by_gameid(result) == {}

    # Games with missing team names are skipped, not crashed on.
    rows = _game("g1", "2026-01-01", "T1", "GEN", blue_win=1)
    for r in rows:
        r["team"] = None
    result = compute_team_elo(pd.DataFrame(rows))
    assert result.ratings == {}
