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


def test_international_k_boost_moves_ratings_more_on_cross_region_games():
    """An inter-region game (international league OR the two teams' most-recent
    leagues differ) moves ratings by k*multiplier; a same-region game uses
    plain k. Verifies both the league-name path and the differing-league path."""
    import math

    from compstrength_pipeline.teams import ELO_SCALE

    # 1) International-league game between two fresh teams: pure k*mult delta.
    intl = pd.DataFrame(_game("m1", "2026-05-01", "T1", "GEN", blue_win=1, league="MSI"))
    base = compute_team_elo(intl, k=32.0, international_leagues=frozenset({"MSI"}),
                            international_k_multiplier=1.0)
    boosted = compute_team_elo(intl, k=32.0, international_leagues=frozenset({"MSI"}),
                               international_k_multiplier=3.0)
    exp = 32.0 * (1 - 0.5)  # both at INITIAL_ELO -> expected 0.5
    assert base.ratings["T1"] == pytest.approx(INITIAL_ELO + exp)
    assert boosted.ratings["T1"] == pytest.approx(INITIAL_ELO + 3.0 * exp)

    # 2) Cross-region detected via differing prior leagues (not a named intl
    # league): each team first plays at home, then they meet in a neutral
    # "PlayIn" league -> that meeting is inter-region and gets the boost.
    games = pd.DataFrame(
        _game("a1", "2026-01-01", "LCKteam", "LCKfoe", blue_win=1, league="LCK")
        + _game("b1", "2026-01-01", "LPLteam", "LPLfoe", blue_win=1, league="LPL")
        + _game("x1", "2026-02-01", "LCKteam", "LPLteam", blue_win=1, league="PlayIn")
    )
    nb = compute_team_elo(games, k=32.0, international_leagues=frozenset({"MSI"}),
                          international_k_multiplier=1.0)
    bb = compute_team_elo(games, k=32.0, international_leagues=frozenset({"MSI"}),
                          international_k_multiplier=3.0)
    # The cross-region game x1 should move LCKteam more under the boost.
    assert bb.ratings["LCKteam"] > nb.ratings["LCKteam"]
    # A same-region game (a1) is unaffected by the multiplier.
    assert bb.ratings["LCKfoe"] == pytest.approx(nb.ratings["LCKfoe"])


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


def test_season_carryover_regresses_ratings_at_year_boundary():
    """At a calendar-year boundary, every rating keeps only `carryover` of
    its deviation from 1500 (roster turnover)."""
    games = pd.DataFrame(
        _game("g1", "2025-06-01", "T1", "GEN", blue_win=1)
        + _game("g2", "2025-06-02", "T1", "GEN", blue_win=1)
        + _game("g3", "2026-01-15", "T1", "GEN", blue_win=1)
    )
    full = compute_team_elo(games, season_carryover=1.0)
    reg = compute_team_elo(games, season_carryover=0.5)
    # Pre-game rating for the first 2026 game reflects the regression.
    full_pre = full.per_game.loc["g3"]["blue_elo_pre"]
    reg_pre = reg.per_game.loc["g3"]["blue_elo_pre"]
    assert full_pre > INITIAL_ELO
    assert reg_pre == pytest.approx(INITIAL_ELO + 0.5 * (full_pre - INITIAL_ELO))
    # Within a season nothing regresses: g2's pre-game ratings identical.
    assert reg.per_game.loc["g2"]["blue_elo_pre"] == pytest.approx(
        full.per_game.loc["g2"]["blue_elo_pre"]
    )


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


# --------------------------------------------------------------------------
# Margin-of-victory weighting (mov_scale / game_margins)
# --------------------------------------------------------------------------


def test_mov_scale_none_is_exactly_the_plain_update():
    """The default path must be bit-identical to classic Elo, so enabling the
    feature is the only thing that can change ratings."""
    games = pd.DataFrame(_game("g1", "2026-01-01", "A", "B", blue_win=1))
    margins = {"g1": 0.5}  # huge margin, but mov_scale=None must ignore it

    plain = compute_team_elo(games)
    disabled = compute_team_elo(games, game_margins=margins, mov_scale=None)
    assert disabled.ratings == plain.ratings


def test_mov_dominant_win_moves_ratings_more_than_a_narrow_one():
    """A blowout should teach the ratings more than a coinflip win."""
    games = pd.DataFrame(_game("g1", "2026-01-01", "A", "B", blue_win=1))

    narrow = compute_team_elo(games, game_margins={"g1": 0.01}, mov_scale=10.0)
    blowout = compute_team_elo(games, game_margins={"g1": 0.40}, mov_scale=10.0)

    assert blowout.ratings["A"] > narrow.ratings["A"] > INITIAL_ELO
    # Still zero-sum: whatever the winner gains, the loser loses.
    for res in (narrow, blowout):
        assert res.ratings["A"] - INITIAL_ELO == pytest.approx(INITIAL_ELO - res.ratings["B"])


def test_mov_direction_still_follows_who_actually_won():
    """The margin scales the update; it must never flip its SIGN. A dominant
    loss (winning the gold game but losing the game) still costs rating."""
    games = pd.DataFrame(_game("g1", "2026-01-01", "A", "B", blue_win=0))
    res = compute_team_elo(games, game_margins={"g1": 0.35}, mov_scale=10.0)
    assert res.ratings["A"] < INITIAL_ELO < res.ratings["B"]


def test_mov_games_without_a_margin_fall_back_to_plain_elo():
    """Sources lacking gold data (e.g. Leaguepedia) must still rate normally."""
    games = pd.DataFrame(_game("g1", "2026-01-01", "A", "B", blue_win=1))
    fallback = compute_team_elo(games, game_margins={}, mov_scale=10.0)
    assert fallback.ratings == compute_team_elo(games).ratings


def test_mov_uses_pre_game_ratings_only():
    """The margin may only affect LATER games -- game 1's recorded feature is
    still the pre-game 1500/1500, which is what keeps the feature leak-free."""
    games = pd.DataFrame(
        _game("g1", "2026-01-01", "A", "B", blue_win=1)
        + _game("g2", "2026-01-02", "A", "B", blue_win=1)
    )
    res = compute_team_elo(games, game_margins={"g1": 0.3, "g2": 0.3}, mov_scale=10.0)
    assert res.per_game.loc["g1", "blue_elo_pre"] == pytest.approx(INITIAL_ELO)
    assert res.per_game.loc["g1", "red_elo_pre"] == pytest.approx(INITIAL_ELO)
    # Game 2 has absorbed game 1's dominant result.
    assert res.per_game.loc["g2", "blue_elo_pre"] > INITIAL_ELO


def test_mov_autocorrelation_correction_damps_expected_blowouts():
    """The same blowout should move ratings LESS when the winner was already
    rated far above its opponent -- that damping is what stops runaway
    ratings (see teams._MOV_AUTOCORR_C)."""
    from compstrength_pipeline.teams import _mov_multiplier

    underdog_wins_big = _mov_multiplier(0.3, winner_rating_advantage=-300.0, scale=10.0)
    even_match_big = _mov_multiplier(0.3, winner_rating_advantage=0.0, scale=10.0)
    favourite_wins_big = _mov_multiplier(0.3, winner_rating_advantage=300.0, scale=10.0)
    assert underdog_wins_big > even_match_big > favourite_wins_big > 0
