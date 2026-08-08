"""Tests for the Riot Global Power Rankings source (sources/lolesports_gpr.py):
RSC-payload parsing, the timestamped history file, the leakage rule of the
as-of lookup, and reconciling GPR's team names with Oracle's Elixir's."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from compstrength_pipeline.sources.lolesports_gpr import (
    GPR_TEAM_ALIASES,
    GprUnavailableError,
    build_history,
    build_team_name_map,
    gpr_ratings_asof,
    load_gpr_history,
    merge_history,
    normalize_team_name,
    parse_gpr_html,
)


def _payload(records: list[dict]) -> str:
    """A minimal stand-in for the real page: the GPR array embedded in a much
    larger blob of unrelated markup/JSON, exactly as the live page ships it."""
    return (
        '<!doctype html><body><script>self.__next_f.push([1,"'
        '{\\"unrelated\\":1}"])</script>'
        '<script>{"someOtherKey":[{"a":1}],"teamGPR":'
        + json.dumps(records)
        + ',"trailing":true}</script></body>'
    )


def _record(name, code="XX", league="LCK", history=((("2026-01-10"), 1200, 1180, 5),)):
    return {
        "__typename": "TeamGPR",
        "team": {"name": name, "code": code, "homeLeague": {"name": league}},
        "teamGPRHistory": [
            {"dateCalculated": f"{d}T22:00:00Z", "gprScore": g, "elo": e, "rank": r}
            for d, g, e, r in history
        ],
    }


def test_parse_extracts_history_from_embedded_payload():
    html = _payload(
        [
            _record("T1", "T1", history=(("2026-01-10", 1500, 1480, 1), ("2026-02-10", 1520, 1500, 1))),
            _record("Gen.G Esports", "GEN", history=(("2026-01-10", 1400, 1390, 2),)),
        ]
    )
    snaps = parse_gpr_html(html)
    assert len(snaps) == 3
    t1 = [s for s in snaps if s.team == "T1"]
    assert {s.date[:10] for s in t1} == {"2026-01-10", "2026-02-10"}
    assert t1[0].code == "T1" and t1[0].league == "LCK"
    assert {s.gpr_score for s in t1} == {1500.0, 1520.0}


def test_parse_handles_brackets_inside_team_names():
    """The array is brace-matched, so punctuation inside a string value must
    not terminate the scan early."""
    html = _payload([_record("Team [Brackets] {Braces}", "TB")])
    snaps = parse_gpr_html(html)
    assert len(snaps) == 1
    assert snaps[0].team == "Team [Brackets] {Braces}"


def test_parse_raises_when_no_rankings_present():
    with pytest.raises(GprUnavailableError):
        parse_gpr_html("<html><body>no rankings here</body></html>")
    # Key present but every record unusable -> still an explicit failure
    # rather than silently returning an empty history.
    with pytest.raises(GprUnavailableError):
        parse_gpr_html(_payload([{"team": {"name": "T1"}, "teamGPRHistory": []}]))


def test_build_history_groups_by_date_and_round_trips(tmp_path):
    snaps = parse_gpr_html(
        _payload(
            [
                _record("T1", history=(("2026-02-10", 1520, 1500, 1), ("2026-01-10", 1500, 1480, 1))),
                _record("KT", history=(("2026-01-10", 1300, 1290, 8),)),
            ]
        )
    )
    payload = build_history(snaps)
    assert [s["ts"][:10] for s in payload["snapshots"]] == ["2026-01-10", "2026-02-10"]
    assert set(payload["snapshots"][0]["teams"]) == {"T1", "KT"}
    assert payload["teamMeta"]["T1"]["league"] == "LCK"

    path = tmp_path / "gpr_history.json"
    path.write_text(json.dumps(payload))
    history = load_gpr_history(path)
    assert [ts.date().isoformat() for ts, _ in history] == ["2026-01-10", "2026-02-10"]
    assert history[0][1]["T1"][0] == 1500


def test_load_missing_file_is_empty_not_an_error(tmp_path):
    assert load_gpr_history(tmp_path / "nope.json") == []


def test_merge_history_unions_seasons_and_prefers_fresh():
    old = build_history(parse_gpr_html(_payload([_record("T1", history=(("2025-06-01", 1400, 1390, 3),))])))
    new = build_history(
        parse_gpr_html(
            _payload(
                [
                    _record("T1", history=(("2025-06-01", 1401, 1391, 3), ("2026-01-10", 1500, 1480, 1))),
                    _record("KT", history=(("2026-01-10", 1300, 1290, 8),)),
                ]
            )
        )
    )
    merged = merge_history(old, new)
    dates = [s["ts"][:10] for s in merged["snapshots"]]
    assert dates == ["2025-06-01", "2026-01-10"]
    # Fresh wins on a conflicting (date, team).
    assert merged["snapshots"][0]["teams"]["T1"][0] == 1401
    assert set(merged["teamMeta"]) == {"T1", "KT"}


def test_asof_is_strictly_before_the_query_date():
    """The leakage rule that makes this usable as a backtest feature: a
    snapshot computed AT or AFTER the game is never visible to it."""
    history = load_gpr_history_from(
        [("2026-01-10T22:00:00Z", {"T1": [1500, 1480, 1]}),
         ("2026-02-10T22:00:00Z", {"T1": [1600, 1580, 1]})]
    )
    assert gpr_ratings_asof(history, pd.Timestamp("2026-01-05", tz="UTC")) == {}
    assert gpr_ratings_asof(history, pd.Timestamp("2026-02-01", tz="UTC"))["T1"] == 1500
    # Exactly ON a snapshot's timestamp is NOT visible (strictly before).
    assert gpr_ratings_asof(history, pd.Timestamp("2026-02-10T22:00:00Z"))["T1"] == 1500
    assert gpr_ratings_asof(history, pd.Timestamp("2026-03-01", tz="UTC"))["T1"] == 1600
    # field="elo" selects the underlying raw Elo instead of the headline score.
    assert gpr_ratings_asof(
        history, pd.Timestamp("2026-03-01", tz="UTC"), field="elo"
    )["T1"] == 1580


def load_gpr_history_from(pairs):
    return [(pd.Timestamp(ts), teams) for ts, teams in pairs]


@pytest.mark.parametrize(
    "gpr_name, oe_name",
    [
        # Host-city prefixes GPR adds and Oracle's Elixir doesn't.
        ("Suzhou LNG Esports", "LNG Esports"),
        ("Shenzhen NINJAS IN PYJAMAS", "Ninjas in Pyjamas"),
        ("Relove Deep Cross Gaming", "Deep Cross Gaming"),
        # Apostrophe inside the city prefix ("Xi'an" must not split in two).
        ("Xi'an Team WE", "Team WE"),
        # Unspaced org name vs spaced.
        ("WeiboGaming", "Weibo Gaming"),
        # Accents / non-ASCII.
        ("LOS", "LØS"),
        # Filler words and casing.
        ("The Chiefs Esports Club", "Chiefs Esports Club"),
        ("Gen.G Esports", "Gen.G"),
        # Explicit aliases.
        ("Beijing JDG Esports", "JD Gaming"),
        ("Team Liquid Alienware", "Team Liquid"),
        ("RED Kalunga", "RED Canids"),
    ],
)
def test_team_name_map_reconciles_the_two_sources(gpr_name, oe_name):
    assert build_team_name_map([gpr_name], [oe_name]) == {gpr_name: oe_name}


def test_team_name_map_omits_teams_with_no_counterpart():
    """A team Oracle's Elixir has never seen is simply absent -- the feature
    is then 0 for it, which is always better than a wrong join."""
    mapping = build_team_name_map(["T1", "Some Defunct Org"], ["T1", "Gen.G"])
    assert mapping == {"T1": "T1"}


def test_normalize_never_empties_a_name():
    # Every token is filler; falling back to the raw tokens keeps it distinct
    # from any other all-filler name rather than collapsing to "".
    assert normalize_team_name("Team Gaming Club") != ""


def test_aliases_are_not_self_referential():
    """A no-op alias means someone added a mapping normalization already
    handled -- harmless but misleading, so keep the table honest."""
    for gpr_name, oe_name in GPR_TEAM_ALIASES.items():
        assert normalize_team_name(gpr_name) != normalize_team_name(oe_name), gpr_name
