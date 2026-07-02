"""Tests for the solo-queue-informed champion prior: as-of join leakage rule,
prior math, and the flat-prior fallback."""

from __future__ import annotations

import pandas as pd
import pytest

from compstrength_pipeline.config import PipelineConfig
from compstrength_pipeline.features import compute_wr_strength
from compstrength_pipeline.sources.soloqueue import load_solo_history, solo_winrates_asof


def _hist():
    return [
        (pd.Timestamp("2025-06-01", tz="UTC"), {"Ahri": [500, 1000, 5000, 10000]}),
        (pd.Timestamp("2026-01-01", tz="UTC"), {"Ahri": [600, 1000, 5200, 10000]}),
    ]


def test_asof_join_only_sees_past_snapshots():
    h = _hist()
    # Before any snapshot -> empty (fold keeps the flat prior).
    assert solo_winrates_asof(h, pd.Timestamp("2025-01-01", tz="UTC")) == {}
    # Between snapshots -> the earlier one, never the later one.
    mid = solo_winrates_asof(h, pd.Timestamp("2025-08-01", tz="UTC"))
    assert mid["Ahri"] == pytest.approx(((500 + 5000) / 11000, 11000))
    # After both -> the latest.
    late = solo_winrates_asof(h, pd.Timestamp("2026-02-01", tz="UTC"))
    assert late["Ahri"][0] == pytest.approx((600 + 5200) / 11000)


def test_load_solo_history_missing_file_is_empty():
    assert load_solo_history("/nonexistent/soloqueue_history.json") == []


def _one_game(champ_blue, champ_red):
    ref = pd.Timestamp("2026-06-15", tz="UTC")
    rows = []
    for side, champ, res in (("Blue", champ_blue, 1), ("Red", champ_red, 0)):
        rows.append({"gameid": "g1", "date": ref, "patch": "16.12", "league": "LCK",
                     "side": side, "champion": champ, "result": res, "position": "mid"})
    return pd.DataFrame(rows), ref


def test_solo_prior_moves_low_sample_champions():
    games, ref = _one_game("Ahri", "Zed")
    # Strong solo champion "Yone" has NO pro games; solo mean is 0.5.
    solo = {"Yone": (0.55, 50000), "Ahri": (0.45, 50000), "Filler": (0.50, 100000)}
    cfg_off = PipelineConfig(wr_prior_solo_weight=0.0)
    cfg_on = PipelineConfig(wr_prior_solo_weight=1.0)

    s_off, _ = compute_wr_strength(games, ref, cfg_off, solo)
    s_on, _ = compute_wr_strength(games, ref, cfg_on, solo)

    # Off = exact old behavior: prior is flat, Yone absent (no pro games).
    assert "Yone" not in s_off
    # On: Yone appears with a positive prior-driven strength.
    assert s_on["Yone"] > 0
    # Ahri (1 pro win but weak solo prior) is pulled DOWN vs the flat prior.
    assert s_on["Ahri"] < s_off["Ahri"]


def test_solo_prior_clip_bounds_extremes():
    games, ref = _one_game("Ahri", "Zed")
    solo = {"Yone": (0.80, 50000), "Filler": (0.50, 1000000)}  # absurd 80% WR
    cfg = PipelineConfig(wr_prior_solo_weight=1.5, wr_prior_clip=0.06)
    s, _ = compute_wr_strength(games, ref, cfg, solo)
    # Prior mean capped at 0.56 -> strength <= 0.06 * scale.
    assert s["Yone"] <= 0.06 * cfg.strength_feature_scale + 1e-9


def test_solo_prior_min_games_filter():
    games, ref = _one_game("Ahri", "Zed")
    solo = {"Yone": (0.60, 100), "Filler": (0.50, 1000000)}  # tiny sample
    cfg = PipelineConfig(wr_prior_solo_weight=1.0, wr_prior_min_solo_games=2000)
    s, _ = compute_wr_strength(games, ref, cfg, solo)
    assert "Yone" not in s  # under-sampled solo entries are ignored


def test_tier_bias_is_centered_out():
    """If every solo winrate is shifted +2pp (tier bias), priors are
    unchanged because the games-weighted mean is subtracted."""
    games, ref = _one_game("Ahri", "Zed")
    solo_a = {"Yone": (0.55, 50000), "Filler": (0.50, 50000)}
    solo_b = {"Yone": (0.57, 50000), "Filler": (0.52, 50000)}
    cfg = PipelineConfig(wr_prior_solo_weight=1.0)
    sa, _ = compute_wr_strength(games, ref, cfg, solo_a)
    sb, _ = compute_wr_strength(games, ref, cfg, solo_b)
    assert sa["Yone"] == pytest.approx(sb["Yone"])
