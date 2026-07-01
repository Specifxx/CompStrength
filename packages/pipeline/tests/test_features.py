"""Unit tests for the empirical-Bayes shrinkage math in features.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from compstrength_pipeline.config import PipelineConfig
from compstrength_pipeline.features import (
    compute_blended_win_rate,
    compute_champion_features,
    compute_champion_shrinkage,
    compute_decayed_pro_stats,
    compute_prior_mean_win_rate,
    compute_strength_score,
    decay_weight,
    logit,
    restrict_to_recent_patches,
    sample_confidence_label,
    select_recent_patches,
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


# ---------------------------------------------------------------------------
# Patch-recency restriction (select_recent_patches / restrict_to_recent_patches)
# ---------------------------------------------------------------------------


def _make_game_row(gameid, date, patch, side, position, champion, result):
    return {
        "gameid": gameid,
        "date": pd.Timestamp(date, tz="UTC"),
        "patch": patch,
        "side": side,
        "position": position,
        "champion": champion,
        "result": result,
    }


def _make_games_df_for_patch_test() -> pd.DataFrame:
    """3 patches, 1 game each, each game with a single champion of interest
    (OldPatchChamp only appears on the oldest patch)."""
    rows = [
        # Oldest patch: 14.1, only OldPatchChamp plays here.
        _make_game_row("g1", "2026-01-01", "14.1", "Blue", "top", "OldPatchChamp", 1),
        _make_game_row("g1", "2026-01-01", "14.1", "Red", "top", "Filler1", 0),
        # Middle patch: 14.2
        _make_game_row("g2", "2026-02-01", "14.2", "Blue", "top", "Filler2", 1),
        _make_game_row("g2", "2026-02-01", "14.2", "Red", "top", "Filler3", 0),
        # Most recent patch: 14.3, RecentChamp plays and always wins.
        _make_game_row("g3", "2026-03-01", "14.3", "Blue", "top", "RecentChamp", 1),
        _make_game_row("g3", "2026-03-01", "14.3", "Red", "top", "Filler4", 0),
    ]
    return pd.DataFrame(rows)


def test_select_recent_patches_orders_by_max_date_not_lexicographic():
    # Deliberately use a patch string ("14.10") that would sort *before*
    # "14.2" lexicographically despite being chronologically later, to
    # prove we're ordering by date, not by string/semver sort.
    df = pd.DataFrame(
        [
            _make_game_row("g1", "2026-01-01", "14.2", "Blue", "top", "A", 1),
            _make_game_row("g1", "2026-01-01", "14.2", "Red", "top", "B", 0),
            _make_game_row("g2", "2026-06-01", "14.10", "Blue", "top", "C", 1),
            _make_game_row("g2", "2026-06-01", "14.10", "Red", "top", "D", 0),
        ]
    )
    patches = select_recent_patches(df, num_recent_patches=2)
    # Most-recent-first: 14.10 (June) should come before 14.2 (January),
    # even though "14.10" < "14.2" as a string.
    assert patches == ["14.10", "14.2"]


def test_select_recent_patches_respects_num_recent_patches():
    df = _make_games_df_for_patch_test()
    assert select_recent_patches(df, num_recent_patches=1) == ["14.3"]
    assert select_recent_patches(df, num_recent_patches=2) == ["14.3", "14.2"]
    assert select_recent_patches(df, num_recent_patches=3) == ["14.3", "14.2", "14.1"]
    # Asking for more than exist just returns all of them.
    assert select_recent_patches(df, num_recent_patches=10) == ["14.3", "14.2", "14.1"]


def test_restrict_to_recent_patches_drops_older_patch_games():
    df = _make_games_df_for_patch_test()
    empty_bans = pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])

    restricted_games, restricted_bans, patches_used = restrict_to_recent_patches(
        df, empty_bans, num_recent_patches=2
    )

    assert patches_used == ["14.3", "14.2"]
    assert set(restricted_games["gameid"]) == {"g2", "g3"}
    assert "OldPatchChamp" not in restricted_games["champion"].values


def test_old_excluded_patch_does_not_affect_champion_rating(config: PipelineConfig):
    """A champion whose only games are on an excluded old patch should end
    up with zero decayed pro games and a rating driven purely by its solo
    queue prior -- the old pro games must have literally zero influence."""
    df = _make_games_df_for_patch_test()
    # Give OldPatchChamp a 100% raw pro win rate on the excluded patch; if
    # the patch restriction leaked through, this would push its rating up.
    restricted_config = PipelineConfig(
        patch_half_life_days=config.patch_half_life_days,
        solo_queue_weight=config.solo_queue_weight,
        prior_games=config.prior_games,
        pro_window_days=config.pro_window_days,
        global_mean=config.global_mean,
        num_recent_patches=2,  # excludes 14.1, where OldPatchChamp lives
    )

    solo_winrates = {
        "OldPatchChamp": (0.5, 10000),
        "Filler2": (0.5, 10000),
        "Filler3": (0.5, 10000),
        "RecentChamp": (0.5, 10000),
        "Filler4": (0.5, 10000),
    }

    result_df = compute_champion_features(
        games_df=df,
        bans_df=pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"]),
        solo_winrates=solo_winrates,
        config=restricted_config,
        reference_date=pd.Timestamp("2026-03-01", tz="UTC"),
    )

    # OldPatchChamp should not appear at all in the patch-restricted table.
    assert "OldPatchChamp" not in result_df.index

    # RecentChamp (100% win rate, most recent patch) should still be there
    # with a strength score above the neutral prior (it actually won).
    assert "RecentChamp" in result_df.index
    assert result_df.loc["RecentChamp", "proGames"] >= 1
    assert result_df.loc["RecentChamp", "strengthScore"] > 0


def test_num_recent_patches_validation():
    with pytest.raises(ValueError):
        PipelineConfig(num_recent_patches=0)
    with pytest.raises(ValueError):
        PipelineConfig(num_recent_patches=-1)
    # Valid values should not raise.
    PipelineConfig(num_recent_patches=1)
    PipelineConfig(num_recent_patches=5)
