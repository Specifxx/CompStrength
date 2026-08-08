"""Tests for the team-strength refinements added alongside the base Elo pass:
margin-of-victory-scaled K, the opponent-adjusted early-game (gold-at-15)
rating, and the Riot GPR join. The shared theme is leak-freedom -- every one
of these records a PRE-game value -- plus degrading to a no-op rather than
crashing when the source doesn't publish the underlying column."""

from __future__ import annotations

import pandas as pd
import pytest

from compstrength_pipeline.teams import (
    INITIAL_ELO,
    compute_econ_ratings,
    compute_gpr_diffs,
    compute_team_elo,
)

POSITIONS = ["top", "jng", "mid", "bot", "sup"]


def _game(
    gameid,
    date,
    blue_team,
    red_team,
    blue_win,
    league="LCK",
    gamelength=1900,
    blue_gd15=None,
):
    """Ten player rows for one game. ``blue_gd15`` is blue's TEAM gold lead at
    15, split evenly across its five players (Oracle's Elixir stores the
    per-player lane diffs, which sum to the team's)."""
    rows = []
    for side, team, result in (("Blue", blue_team, blue_win), ("Red", red_team, 1 - blue_win)):
        sign = 1 if side == "Blue" else -1
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
                    "gamelength": gamelength,
                    "golddiffat15": None if blue_gd15 is None else sign * blue_gd15 / 5,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Margin-of-victory K scaling
# ---------------------------------------------------------------------------


def test_mov_moves_ratings_more_after_a_decisive_win():
    """Same result, different game length: the short (dominant) win must move
    the rating further than the long grind."""
    games = pd.DataFrame(
        _game("g1", "2026-01-01", "A", "B", 1, gamelength=1500)   # short: dominant
        + _game("g2", "2026-01-01", "C", "D", 1, gamelength=2400)  # long: grindy
        + _game("g3", "2026-01-01", "E", "F", 1, gamelength=1950)  # ~median
    )
    off = compute_team_elo(games, mov_alpha=0.0)
    on = compute_team_elo(games, mov_alpha=1.0)
    # Without MOV every winner gains exactly the same amount.
    assert off.ratings["A"] == pytest.approx(off.ratings["C"])
    # With MOV the stomp is worth more than the grind.
    assert on.ratings["A"] > on.ratings["E"] > on.ratings["C"]
    # It only redistributes weight -- it never flips who gained.
    assert on.ratings["C"] > INITIAL_ELO


def test_mov_is_a_no_op_without_game_lengths():
    rows = _game("g1", "2026-01-01", "A", "B", 1)
    for r in rows:
        r.pop("gamelength")
    games = pd.DataFrame(rows)
    assert compute_team_elo(games, mov_alpha=1.0).ratings == pytest.approx(
        compute_team_elo(games, mov_alpha=0.0).ratings
    )


def test_mov_zero_reproduces_plain_elo_exactly():
    games = pd.DataFrame(
        _game("g1", "2026-01-01", "A", "B", 1, gamelength=1200)
        + _game("g2", "2026-01-02", "A", "B", 0, gamelength=3000)
    )
    assert compute_team_elo(games, mov_alpha=0.0).ratings == pytest.approx(
        compute_team_elo(games).ratings
    )


# ---------------------------------------------------------------------------
# Early-game (gold-at-15) rating
# ---------------------------------------------------------------------------


def test_econ_records_pre_game_ratings_only():
    games = pd.DataFrame(
        _game("g1", "2026-01-01", "A", "B", 1, blue_gd15=3000)
        + _game("g2", "2026-01-02", "A", "B", 1, blue_gd15=3000)
    )
    result = compute_econ_ratings(games, k=0.2, feature_scale=1000.0)
    # First game: both teams start at 0, so the feature is exactly 0.
    assert result.per_game["g1"] == pytest.approx(0.0)
    # Second game sees the first game's lead, and the ratings are zero-sum.
    assert result.per_game["g2"] > 0
    assert result.ratings["A"] == pytest.approx(-result.ratings["B"])
    assert result.ratings["A"] > 0 > result.ratings["B"]


def test_econ_is_opponent_adjusted():
    """+2k gold against a team already known to be weak is worth LESS than the
    same lead against an unknown (average) team -- that's the whole point of
    subtracting the predicted margin."""
    weakened = pd.DataFrame(
        _game("s1", "2026-01-01", "B", "X", 1, blue_gd15=-5000)
        + _game("s2", "2026-01-02", "B", "Y", 1, blue_gd15=-5000)
        + _game("g1", "2026-01-03", "A", "B", 1, blue_gd15=2000)
    )
    fresh = pd.DataFrame(_game("g1", "2026-01-03", "A", "B", 1, blue_gd15=2000))
    against_weak = compute_econ_ratings(weakened, k=0.2).ratings["A"]
    against_average = compute_econ_ratings(fresh, k=0.2).ratings["A"]
    assert against_weak < against_average


def test_econ_skips_games_with_missing_gold_but_still_scores_them():
    games = pd.DataFrame(
        _game("g1", "2026-01-01", "A", "B", 1, blue_gd15=3000)
        + _game("g2", "2026-01-02", "A", "B", 1, blue_gd15=None)  # no gold data
        + _game("g3", "2026-01-03", "A", "B", 1, blue_gd15=None)
    )
    result = compute_econ_ratings(games, k=0.2, feature_scale=1000.0)
    # g2 and g3 still get a (pre-game) feature value...
    assert result.per_game["g2"] == pytest.approx(result.per_game["g3"])
    # ...but contributed no rating movement, so only g1's lead is baked in.
    only_g1 = compute_econ_ratings(
        pd.DataFrame(_game("g1", "2026-01-01", "A", "B", 1, blue_gd15=3000)), k=0.2
    )
    assert result.ratings["A"] == pytest.approx(only_g1.ratings["A"])


def test_econ_partial_lane_data_is_treated_as_missing():
    """A side with only some lanes reported would sum to an understated team
    lead; that must not silently become a real observation."""
    rows = _game("g1", "2026-01-01", "A", "B", 1, blue_gd15=5000)
    for r in rows:
        if r["side"] == "Blue" and r["position"] == "sup":
            r["golddiffat15"] = None
    assert compute_econ_ratings(pd.DataFrame(rows), k=0.2).ratings == {}


def test_econ_without_the_column_is_a_no_op():
    rows = _game("g1", "2026-01-01", "A", "B", 1)
    for r in rows:
        r.pop("golddiffat15")
    result = compute_econ_ratings(pd.DataFrame(rows), k=0.2)
    assert result.ratings == {}
    assert result.per_game["g1"] == pytest.approx(0.0)


def test_econ_season_carryover_regresses_toward_average():
    games = pd.DataFrame(
        _game("g1", "2025-06-01", "A", "B", 1, blue_gd15=4000)
        + _game("g2", "2026-01-15", "A", "B", 1, blue_gd15=4000)
    )
    full = compute_econ_ratings(games, k=0.2, season_carryover=1.0, feature_scale=1.0)
    regressed = compute_econ_ratings(games, k=0.2, season_carryover=0.5, feature_scale=1.0)
    assert regressed.per_game["g2"] == pytest.approx(0.5 * full.per_game["g2"])


# ---------------------------------------------------------------------------
# Riot GPR join
# ---------------------------------------------------------------------------


def _history(pairs):
    return [(pd.Timestamp(ts), teams) for ts, teams in pairs]


def test_gpr_uses_only_snapshots_published_before_each_game():
    games = pd.DataFrame(
        _game("early", "2026-01-05", "T1", "Gen.G", 1)
        + _game("late", "2026-02-20", "T1", "Gen.G", 1)
    )
    history = _history(
        [
            ("2026-01-10T22:00:00Z", {"T1": [1500, 1480, 1], "Gen.G Esports": [1400, 1390, 2]}),
            ("2026-02-10T22:00:00Z", {"T1": [1600, 1580, 1], "Gen.G Esports": [1300, 1290, 4]}),
        ]
    )
    result = compute_gpr_diffs(games, history, feature_scale=100.0)
    # No snapshot predates the early game -> no data -> 0, not a guess.
    assert result.per_game["early"] == pytest.approx(0.0)
    # The later game sees the Feb snapshot: (1600 - 1300) / 100.
    assert result.per_game["late"] == pytest.approx(3.0)
    assert result.covered_games == 1
    assert result.ratings["T1"] == 1600  # "as of today" for the frontend


def test_gpr_is_zero_unless_both_teams_are_rated():
    games = pd.DataFrame(_game("g1", "2026-03-01", "T1", "Some Academy Team", 1))
    history = _history([("2026-01-10T22:00:00Z", {"T1": [1500, 1480, 1]})])
    result = compute_gpr_diffs(games, history, feature_scale=100.0)
    assert result.per_game["g1"] == pytest.approx(0.0)
    assert result.covered_games == 0


def test_gpr_field_selects_score_or_underlying_elo():
    games = pd.DataFrame(_game("g1", "2026-03-01", "T1", "Gen.G", 1))
    history = _history(
        [("2026-01-10T22:00:00Z", {"T1": [1500, 1480, 1], "Gen.G Esports": [1400, 1300, 2]})]
    )
    assert compute_gpr_diffs(games, history, 100.0, "gpr").per_game["g1"] == pytest.approx(1.0)
    assert compute_gpr_diffs(games, history, 100.0, "elo").per_game["g1"] == pytest.approx(1.8)


def test_gpr_without_history_is_a_no_op():
    games = pd.DataFrame(_game("g1", "2026-03-01", "T1", "Gen.G", 1))
    result = compute_gpr_diffs(games, [], feature_scale=100.0)
    assert result.per_game == {} and result.ratings == {} and result.covered_games == 0
