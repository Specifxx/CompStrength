"""Tests for the min_patch floor: drop patches older than the floor,
compared numerically (not lexically), with a graceful skip when the floor
would empty the dataset (e.g. the synthetic fixture's 14.x patches)."""

from __future__ import annotations

import pandas as pd
import pytest

from compstrength_pipeline.config import PipelineConfig
from compstrength_pipeline.features import parse_patch, restrict_to_min_patch


@pytest.mark.parametrize(
    "patch,expected",
    [
        ("26.13", (26, 13)),
        ("26.2", (26, 2)),
        ("25.1", (25, 1)),
        ("14", (14, 0)),
        ("", None),
        (None, None),
        (float("nan"), None),
        ("Preseason", None),
    ],
)
def test_parse_patch(patch, expected):
    assert parse_patch(patch) == expected


def test_parse_patch_orders_numerically_not_lexically():
    # The whole point: "26.13" is newer than "26.2", which string sort gets wrong.
    assert parse_patch("26.13") > parse_patch("26.2")
    assert parse_patch("26.1") > parse_patch("25.24")


def _games(patches):
    return pd.DataFrame(
        [
            {"gameid": f"g{i}", "patch": p, "champion": "Ahri"}
            for i, p in enumerate(patches)
        ]
    )


def test_restrict_to_min_patch_drops_older_patches():
    games = _games(["26.13", "26.12", "25.1", "24.20", "25.0"])
    bans = pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])
    kept, _ = restrict_to_min_patch(games, bans, "25.1")
    # 24.20 and 25.0 are below the 25.1 floor; the rest stay.
    assert set(kept["patch"]) == {"26.13", "26.12", "25.1"}


def test_restrict_to_min_patch_filters_bans_to_surviving_games():
    games = _games(["26.13", "24.1"])
    bans = pd.DataFrame(
        [
            {"gameid": "g0", "team": "T", "champion": "Zed", "ban_number": 1},
            {"gameid": "g1", "team": "T", "champion": "Yone", "ban_number": 1},
        ]
    )
    kept_games, kept_bans = restrict_to_min_patch(games, bans, "25.1")
    assert set(kept_games["gameid"]) == {"g0"}
    assert set(kept_bans["gameid"]) == {"g0"}


def test_restrict_to_min_patch_skips_when_it_would_empty_everything():
    # The synthetic fixture is all 14.x; a 25.1 floor would drop everything,
    # so the filter must be skipped (keep all) rather than break offline runs.
    games = _games(["14.1", "14.2", "14.3"])
    bans = pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])
    with pytest.warns(UserWarning, match="would drop every game"):
        kept, _ = restrict_to_min_patch(games, bans, "25.1")
    assert len(kept) == 3


def test_restrict_to_min_patch_none_is_noop():
    games = _games(["14.1", "26.13"])
    kept, _ = restrict_to_min_patch(games, None, None)
    assert len(kept) == 2


def test_min_patch_config_validation():
    PipelineConfig(min_patch="25.1")
    PipelineConfig(min_patch=None)
    with pytest.raises(ValueError):
        PipelineConfig(min_patch="not-a-patch")
