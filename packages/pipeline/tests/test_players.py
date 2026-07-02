"""Tests for players.py: per-player Elo, player-champion proficiency, and
roster inference -- the OPTIONAL player-level add-on to the team feature."""

from __future__ import annotations

import pandas as pd
import pytest

from compstrength_pipeline.players import (
    compute_player_stats,
    proficiency,
)
from compstrength_pipeline.teams import INITIAL_ELO


def _game(gameid, date, blue_team, red_team, blue_players, red_players, blue_win):
    """10 rows for one game. ``*_players`` are [(player, position, champion)]."""
    rows = []
    for player, position, champion in blue_players:
        rows.append(
            {"gameid": gameid, "date": pd.Timestamp(date, tz="UTC"), "side": "Blue",
             "team": blue_team, "player": player, "position": position,
             "champion": champion, "result": blue_win}
        )
    for player, position, champion in red_players:
        rows.append(
            {"gameid": gameid, "date": pd.Timestamp(date, tz="UTC"), "side": "Red",
             "team": red_team, "player": player, "position": position,
             "champion": champion, "result": 1 - blue_win}
        )
    return rows


POSITIONS = ["top", "jng", "mid", "bot", "sup"]


def _seats(prefix, champs=None):
    champs = champs or ["A", "B", "C", "D", "E"]
    return [(f"{prefix}{i}", POSITIONS[i], champs[i]) for i in range(5)]


def _two_game_frame(dates=("2026-01-10", "2026-01-11")):
    rows = _game("g1", dates[0], "T1", "T2", _seats("p"), _seats("q"), 1)
    rows += _game("g2", dates[1], "T1", "T2", _seats("p"), _seats("q"), 1)
    return pd.DataFrame(rows)


def test_pre_game_features_exclude_own_game():
    """Game 1's features must be all-neutral (nobody has history), and game
    2's must reflect ONLY game 1 -- the pre-game construction that makes the
    same series safe for walk-forward backtesting."""
    result = compute_player_stats(_two_game_frame(), elo_feature_scale=15.0)

    # Game 1: every player at INITIAL_ELO, no champion history.
    assert result.player_elo_diff["g1"] == 0.0
    assert result.prof_diff["g1"] == 0.0

    # Game 2: blue's five players each won game 1, red's each lost. But
    # proficiency stays 0: with a single-champion history a player's champion
    # record IS their overall record, and proficiency measures the champion
    # -specific edge OVER the player's own base (player Elo carries the
    # general-skill part).
    assert result.player_elo_diff["g2"] > 0.0
    assert result.prof_diff["g2"] == 0.0


def test_proficiency_feature_reflects_champion_specific_history():
    """A player who wins on champion A but loses on champion B has a positive
    champion-specific edge when back on A -- and that shows up in prof_diff."""
    rows = _game("g1", "2026-01-10", "T1", "T2",
                 _seats("p", ["A", "B", "C", "D", "E"]), _seats("q"), 1)
    # Same players, but blue's are all on DIFFERENT champions; they lose.
    rows += _game("g2", "2026-01-11", "T1", "T2",
                  _seats("p", ["V", "W", "X", "Y", "Z"]), _seats("q"), 0)
    # Game 3: blue back on their winning champions.
    rows += _game("g3", "2026-01-12", "T1", "T2",
                  _seats("p", ["A", "B", "C", "D", "E"]), _seats("q"), 1)
    result = compute_player_stats(pd.DataFrame(rows), elo_feature_scale=15.0)

    # Each blue player: overall 1/2, on their game-3 champion 1/1 -> positive
    # edge of (1 + s*0.5)/(1 + s) - 0.5 each. Red players' champion record
    # equals their overall record -> 0 contribution.
    shrink = 8.0
    per_player = (1 + shrink * 0.5) / (1 + shrink) - 0.5
    assert result.prof_diff["g3"] == pytest.approx(5 * per_player)


def test_player_elo_diff_scaling_and_value():
    """Winning one game as 5 average players moves each player by exactly
    k * (1 - 0.5); the game-2 feature is (mean_blue - mean_red)/scale."""
    k, scale = 32.0, 15.0
    result = compute_player_stats(_two_game_frame(), elo_feature_scale=scale, k=k)
    # After g1: blue players all INITIAL+16, red all INITIAL-16 -> gap 32.
    assert result.player_elo_diff["g2"] == pytest.approx(2 * (k * 0.5) / scale)


def test_elo_season_carryover_regresses_toward_initial():
    frame = _two_game_frame(dates=("2025-12-30", "2026-01-05"))
    full = compute_player_stats(frame, elo_feature_scale=15.0, season_carryover=1.0)
    regressed = compute_player_stats(frame, elo_feature_scale=15.0, season_carryover=0.5)
    # The year boundary between the games halves every deviation.
    assert regressed.player_elo_diff["g2"] == pytest.approx(
        0.5 * full.player_elo_diff["g2"]
    )


def test_proficiency_math():
    """proficiency = shrunk player-champion WR minus the player's own overall
    WR: champion-specific skill only, not general skill."""
    wr = {"p": [6.0, 10.0]}          # 60% overall
    champ = {("p", "Ahri"): [4.0, 4.0]}  # 4-0 on Ahri
    shrink = 8.0
    # (4 + 8*0.6) / (4 + 8) - 0.6 = 8.8/12 - 0.6
    assert proficiency(wr, champ, "p", "Ahri", shrink) == pytest.approx(
        8.8 / 12.0 - 0.6
    )
    # Unseen champion: shrinks fully to the player's own base -> exactly 0.
    assert proficiency(wr, champ, "p", "Zed", shrink) == 0.0
    # Unseen player: no information at all -> exactly 0.
    assert proficiency(wr, champ, "nobody", "Ahri", shrink) == 0.0


def test_rosters_are_most_recent_lineup():
    """A substitution shows up in the roster: the team's CURRENT roster is
    the lineup from its most recent game."""
    rows = _game("g1", "2026-01-10", "T1", "T2", _seats("p"), _seats("q"), 1)
    subbed = _seats("p")
    subbed[2] = ("pSub", "mid", "C")  # mid laner replaced in game 2
    rows += _game("g2", "2026-01-11", "T1", "T2", subbed, _seats("q"), 0)
    result = compute_player_stats(pd.DataFrame(rows), elo_feature_scale=15.0)

    assert result.rosters["T1"] == [(p, pos) for p, pos, _ in subbed]
    assert result.rosters["T2"] == [(q, pos) for q, pos, _ in _seats("q")]
    # The substitute debuts at the initial rating; the benched starter's
    # record is retained.
    assert result.elo["pSub"] != INITIAL_ELO  # updated by game 2's result
    assert result.wr["p2"] == [1, 1]  # played (and won) only game 1


def test_final_tables_cover_all_games():
    result = compute_player_stats(_two_game_frame(), elo_feature_scale=15.0)
    # Each of the 10 players has 2 games; blue side won both.
    assert result.wr["p0"] == [2, 2]
    assert result.wr["q0"] == [0, 2]
    assert result.champ[("p0", "A")] == [2, 2]
    # Elo stays zero-sum around the initial rating.
    total = sum(result.elo.values())
    assert total == pytest.approx(len(result.elo) * INITIAL_ELO)


def test_incomplete_games_are_skipped():
    rows = _game("g1", "2026-01-10", "T1", "T2", _seats("p"), _seats("q"), 1)
    rows += _game("g2", "2026-01-11", "T1", "T2", _seats("p")[:4], _seats("q"), 1)
    result = compute_player_stats(pd.DataFrame(rows), elo_feature_scale=15.0)
    assert "g1" in result.player_elo_diff
    assert "g2" not in result.player_elo_diff
