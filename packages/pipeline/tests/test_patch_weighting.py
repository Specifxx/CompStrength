"""Tests for latest-patch weighting: newest patches must count more than
older ones in the same window, on top of the calendar-day decay."""

from __future__ import annotations

import pandas as pd
import pytest

from compstrength_pipeline.config import PipelineConfig
from compstrength_pipeline.features import (
    compute_decayed_pro_stats,
    patch_ordinal_distances,
    patch_weight_series,
)


def _games(rows):
    return pd.DataFrame(rows)


def test_patch_ordinal_distances_orders_by_date_not_lexically():
    # "14.10" is chronologically newer than "14.2" but sorts BEFORE it as a
    # string -- distances must come from dates, not string/semver order.
    df = _games(
        [
            {"gameid": "g1", "patch": "14.2", "date": pd.Timestamp("2026-01-01", tz="UTC")},
            {"gameid": "g2", "patch": "14.10", "date": pd.Timestamp("2026-06-01", tz="UTC")},
            {"gameid": "g3", "patch": "14.9", "date": pd.Timestamp("2026-05-01", tz="UTC")},
        ]
    )
    distances = patch_ordinal_distances(df)
    assert distances == {"14.10": 0, "14.9": 1, "14.2": 2}


def test_patch_weight_series_geometric_decay():
    patches = pd.Series(["A", "B", "C", "A"])
    distances = {"A": 0, "B": 1, "C": 2}
    weights = patch_weight_series(patches, distances, patch_decay_base=0.5)
    assert list(weights) == [1.0, 0.5, 0.25, 1.0]


def test_patch_weight_series_disabled_when_base_is_one_or_none():
    patches = pd.Series(["A", "B"])
    distances = {"A": 0, "B": 1}
    assert list(patch_weight_series(patches, distances, 1.0)) == [1.0, 1.0]
    assert list(patch_weight_series(patches, None, 0.5)) == [1.0, 1.0]


def test_latest_patch_game_outweighs_older_patch_game_same_date():
    """Two single-champion games on the SAME calendar date but different
    patches: with patch weighting on, the newer-patch game dominates the
    decayed win rate."""
    ref = pd.Timestamp("2026-06-15", tz="UTC")
    # Newest patch game is a WIN, older-patch game is a LOSS, same date.
    games = _games(
        [
            {"date": ref, "result": 1, "patch": "26.13"},
            {"date": ref, "result": 0, "patch": "26.10"},
        ]
    )
    distances = {"26.13": 0, "26.10": 3}  # older patch is 3 back

    # Without patch weighting (base=1.0): equal weight -> win rate 0.5.
    decayed_flat, wr_flat = compute_decayed_pro_stats(
        games, ref, half_life_days=21, patch_distances=distances, patch_decay_base=1.0
    )
    assert wr_flat == pytest.approx(0.5)

    # With patch weighting (base=0.5): newest win weighted 1.0, old loss
    # weighted 0.5**3 = 0.125 -> win rate 1.0/1.125 ~= 0.889, dominated by
    # the current patch.
    decayed_w, wr_w = compute_decayed_pro_stats(
        games, ref, half_life_days=21, patch_distances=distances, patch_decay_base=0.5
    )
    assert wr_w > 0.85
    assert decayed_w < decayed_flat  # older game contributes less mass


def test_patch_decay_base_config_validation():
    with pytest.raises(ValueError):
        PipelineConfig(patch_decay_base=0.0)
    with pytest.raises(ValueError):
        PipelineConfig(patch_decay_base=1.5)
    PipelineConfig(patch_decay_base=1.0)  # 1.0 = disabled, allowed
    PipelineConfig(patch_decay_base=0.5)
