"""Unit tests for the pure (non-network) parsing/mapping logic in
sources/leaguepedia.py -- the live Cargo API calls themselves aren't
exercised here since lol.fandom.com is blocked in this sandbox (see the
module docstring); these test the data-shaping logic against fixed
DataFrames standing in for what the API returns."""

from __future__ import annotations

import pandas as pd
import pytest

from compstrength_pipeline.sources.leaguepedia import (
    ROLE_TO_POSITION,
    SIDE_TO_LABEL,
    _build_canonical_bans,
    _build_canonical_games,
    _cargo_in_clause,
    infer_league,
)


@pytest.mark.parametrize(
    "tournament,expected",
    [
        ("2026 Mid-Season Invitational/Play-In Day 4", "MSI"),
        ("2026 World Championship/Play-In", "WORLDS"),
        ("Worlds 2026 Main Event", "WORLDS"),
        ("LCK 2026 Spring/Playoffs", "LCK"),
        ("LPL 2026 Summer", "LPL"),
        ("Some Unknown Regional Qualifier", "Some Unknown Regional Qualifier"),
        ("", "UNKNOWN"),
        (None, "UNKNOWN"),
    ],
)
def test_infer_league(tournament, expected):
    assert infer_league(tournament) == expected


def test_cargo_in_clause_quotes_and_escapes():
    clause = _cargo_in_clause(['abc"1', "def_2"])
    assert clause == '"abc\\"1","def_2"'


def test_build_canonical_games_maps_side_role_result():
    games_meta = pd.DataFrame(
        [
            {
                "GameId": "G1",
                "DateTimeUTC": "2026-07-01 10:00:00",
                "Patch": "26.13",
                "Tournament": "2026 Mid-Season Invitational/Play-In Day 4",
            }
        ]
    )
    players = pd.DataFrame(
        [
            {
                "GameId": "G1",
                "Player": "CoreJJ",
                "Champion": "Rakan",
                "Side": "1",
                "Role": "Support",
                "PlayerWin": "No",
                "Team": "Team Liquid",
            },
            {
                "GameId": "G1",
                "Player": "Keria",
                "Champion": "Lulu",
                "Side": "2",
                "Role": "Support",
                "PlayerWin": "Yes",
                "Team": "T1",
            },
        ]
    )

    result = _build_canonical_games(games_meta, [players])

    assert set(result.columns) == {
        "gameid", "date", "league", "patch", "side", "team", "player", "position", "champion", "result",
    }
    assert list(result["side"]) == ["Blue", "Red"]
    assert list(result["position"]) == ["sup", "sup"]
    assert list(result["result"]) == [0, 1]
    assert list(result["league"]) == ["MSI", "MSI"]
    assert list(result["patch"]) == ["26.13", "26.13"]


def test_build_canonical_games_empty_when_no_players():
    result = _build_canonical_games(pd.DataFrame(), [])
    assert list(result.columns) == [
        "gameid", "date", "league", "patch", "side", "team", "player", "position", "champion", "result",
    ]
    assert result.empty


def test_build_canonical_bans_resolves_team_names_and_melts():
    games_meta = pd.DataFrame([{"GameId": "G1", "Team1": "Team Liquid", "Team2": "T1"}])
    bans_wide = pd.DataFrame(
        [
            {
                "GameId": "G1",
                "Team1Ban1": "Azir", "Team1Ban2": "Corki", "Team1Ban3": "",
                "Team1Ban4": "", "Team1Ban5": "",
                "Team2Ban1": "Ahri", "Team2Ban2": "", "Team2Ban3": "",
                "Team2Ban4": "", "Team2Ban5": "",
            }
        ]
    )

    result = _build_canonical_bans(games_meta, [bans_wide])

    assert set(result.columns) == {"gameid", "team", "champion", "ban_number"}
    assert len(result) == 3
    assert set(zip(result["team"], result["champion"])) == {
        ("Team Liquid", "Azir"),
        ("Team Liquid", "Corki"),
        ("T1", "Ahri"),
    }


def test_role_and_side_maps_cover_all_observed_values():
    assert set(ROLE_TO_POSITION.keys()) == {"Top", "Jungle", "Mid", "Bot", "Support"}
    assert SIDE_TO_LABEL == {"1": "Blue", "2": "Red"}
