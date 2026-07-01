"""Unit tests for the empirical-Bayes shrinkage math in features.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from compstrength_pipeline.config import PipelineConfig
from compstrength_pipeline.features import (
    compute_blended_win_rate,
    compute_champion_shrinkage,
    compute_decayed_pro_stats,
    compute_prior_mean_win_rate,
    compute_strength_score,
    decay_weight,
    logit,
    sample_confidence_label,
)


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(
        patch_half_life_days=21,
        solo_queue_weight=0.35,
        prior_games=15,
        pro_window_days=90,
        global_mean=0.5,
    )


def test_logit_center_is_zero_at_half():
    assert logit(0.5) == pytest.approx(0.0, abs=1e-9)


def test_logit_clips_extremes():
    # p=1.0 clipped to 0.99, p=0.0 clipped to 0.01 -- should not raise / -> finite.
    assert math.isfinite(logit(1.0))
    assert math.isfinite(logit(0.0))
    assert logit(1.0) == pytest.approx(logit(0.99))
    assert logit(0.0) == pytest.approx(logit(0.01))


def test_decay_weight_at_zero_days_is_one():
    assert decay_weight(0.0, half_life_days=21) == pytest.approx(1.0)


def test_decay_weight_at_half_life_is_half():
    assert decay_weight(21.0, half_life_days=21) == pytest.approx(0.5)


def test_decay_weight_at_two_half_lives_is_quarter():
    assert decay_weight(42.0, half_life_days=21) == pytest.approx(0.25)


def test_decay_weight_negative_days_clipped_to_full_weight():
    assert decay_weight(-5.0, half_life_days=21) == pytest.approx(1.0)


def test_zero_pro_games_blends_fully_to_prior(config: PipelineConfig):
    """A champion with zero pro games should blend fully to the solo-queue
    informed prior (the raw pro win rate should have no influence at all)."""
    solo_win_rate = 0.55
    result = compute_champion_shrinkage(
        pro_games_decayed=0.0,
        pro_win_rate_raw=0.0,  # should be irrelevant since decayed games = 0
        solo_win_rate=solo_win_rate,
        solo_games=20000,
        config=config,
    )
    expected_prior = compute_prior_mean_win_rate(
        solo_win_rate, config.global_mean, config.solo_queue_weight
    )
    assert result.blended_win_rate == pytest.approx(expected_prior)
    assert result.prior_mean_win_rate == pytest.approx(expected_prior)

    # And changing the (irrelevant, since weight=0) raw pro win rate must not
    # change the blended result at all.
    result2 = compute_champion_shrinkage(
        pro_games_decayed=0.0,
        pro_win_rate_raw=0.99,
        solo_win_rate=solo_win_rate,
        solo_games=20000,
        config=config,
    )
    assert result2.blended_win_rate == pytest.approx(result.blended_win_rate)


def test_huge_pro_sample_approaches_raw_pro_win_rate(config: PipelineConfig):
    """As proGamesDecayed grows much larger than PRIOR_GAMES, blendedWinRate
    should approach the raw pro win rate regardless of the prior."""
    raw_pro_win_rate = 0.7
    huge_games = 100_000.0  # >> PRIOR_GAMES=15

    result = compute_champion_shrinkage(
        pro_games_decayed=huge_games,
        pro_win_rate_raw=raw_pro_win_rate,
        solo_win_rate=0.1,  # deliberately very different from raw pro rate
        solo_games=5000,
        config=config,
    )
    assert result.blended_win_rate == pytest.approx(raw_pro_win_rate, abs=1e-3)


def test_strength_score_zero_when_blended_equals_global_mean(config: PipelineConfig):
    score = compute_strength_score(config.global_mean, config.global_mean)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_strength_score_positive_when_above_global_mean(config: PipelineConfig):
    score = compute_strength_score(0.6, config.global_mean)
    assert score > 0


def test_strength_score_negative_when_below_global_mean(config: PipelineConfig):
    score = compute_strength_score(0.4, config.global_mean)
    assert score < 0


@pytest.mark.parametrize(
    "games,expected",
    [
        (0.0, "low"),
        (4.99, "low"),
        (5.0, "medium"),
        (19.99, "medium"),
        (20.0, "high"),
        (500.0, "high"),
    ],
)
def test_sample_confidence_thresholds(games, expected):
    assert sample_confidence_label(games) == expected


def test_compute_blended_win_rate_matches_precision_weighted_formula():
    prior_mean = 0.45
    prior_games = 15
    raw_pro = 0.8
    pro_games_decayed = 5.0

    blended = compute_blended_win_rate(prior_mean, prior_games, raw_pro, pro_games_decayed)
    expected = (prior_mean * prior_games + raw_pro * pro_games_decayed) / (
        prior_games + pro_games_decayed
    )
    assert blended == pytest.approx(expected)
    # Sanity: should be strictly between prior_mean and raw_pro.
    assert prior_mean < blended < raw_pro


def test_compute_blended_win_rate_handles_zero_denominator():
    # prior_games=0 and pro_games_decayed=0 -> degenerate, should not raise/divide by zero.
    result = compute_blended_win_rate(0.5, 0, 0.9, 0.0)
    assert result == pytest.approx(0.5)


def test_compute_decayed_pro_stats_weighted_mean():
    ref_date = pd.Timestamp("2026-01-01", tz="UTC")
    games = pd.DataFrame(
        {
            "date": [
                ref_date,  # weight 1.0, win
                ref_date - pd.Timedelta(days=21),  # weight 0.5, loss
            ],
            "result": [1, 0],
        }
    )
    decayed_games, win_rate = compute_decayed_pro_stats(games, ref_date, half_life_days=21)
    assert decayed_games == pytest.approx(1.5)
    # weighted win sum = 1.0*1 + 0.5*0 = 1.0; win_rate = 1.0 / 1.5
    assert win_rate == pytest.approx(1.0 / 1.5)


def test_compute_decayed_pro_stats_empty_returns_zero():
    ref_date = pd.Timestamp("2026-01-01", tz="UTC")
    empty = pd.DataFrame(columns=["date", "result"])
    decayed_games, win_rate = compute_decayed_pro_stats(empty, ref_date, half_life_days=21)
    assert decayed_games == 0.0
    assert win_rate == 0.0


def test_prior_mean_uses_solo_queue_weight_correctly(config: PipelineConfig):
    solo_win_rate = 0.6
    prior = compute_prior_mean_win_rate(solo_win_rate, config.global_mean, config.solo_queue_weight)
    expected = config.global_mean * (1 - config.solo_queue_weight) + solo_win_rate * config.solo_queue_weight
    assert prior == pytest.approx(expected)
