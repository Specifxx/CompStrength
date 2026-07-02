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
    international_league_multiplier,
    logit,
    restrict_to_recent_games,
    sample_confidence_label,
    select_recent_games,
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
# Recent-games restriction (select_recent_games / restrict_to_recent_games)
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
    (OldPatchChamp only appears on the oldest patch/game)."""
    rows = [
        # Oldest game/patch: 14.1, only OldPatchChamp plays here.
        _make_game_row("g1", "2026-01-01", "14.1", "Blue", "top", "OldPatchChamp", 1),
        _make_game_row("g1", "2026-01-01", "14.1", "Red", "top", "Filler1", 0),
        # Middle game/patch: 14.2
        _make_game_row("g2", "2026-02-01", "14.2", "Blue", "top", "Filler2", 1),
        _make_game_row("g2", "2026-02-01", "14.2", "Red", "top", "Filler3", 0),
        # Most recent game/patch: 14.3, RecentChamp plays and always wins.
        _make_game_row("g3", "2026-03-01", "14.3", "Blue", "top", "RecentChamp", 1),
        _make_game_row("g3", "2026-03-01", "14.3", "Red", "top", "Filler4", 0),
    ]
    return pd.DataFrame(rows)


def test_select_recent_games_orders_by_max_date_not_lexicographic():
    # Deliberately use a patch string ("14.10") that would sort *before*
    # "14.2" lexicographically despite being chronologically later, to
    # prove game ordering is by date, not by the (irrelevant here) patch
    # string.
    df = pd.DataFrame(
        [
            _make_game_row("g1", "2026-01-01", "14.2", "Blue", "top", "A", 1),
            _make_game_row("g1", "2026-01-01", "14.2", "Red", "top", "B", 0),
            _make_game_row("g2", "2026-06-01", "14.10", "Blue", "top", "C", 1),
            _make_game_row("g2", "2026-06-01", "14.10", "Red", "top", "D", 0),
        ]
    )
    games = select_recent_games(df, target_games=2)
    # Most-recent-first: g2 (June, patch "14.10") before g1 (January).
    assert games == ["g2", "g1"]


def test_select_recent_games_respects_target_games():
    df = _make_games_df_for_patch_test()
    assert select_recent_games(df, target_games=1) == ["g3"]
    assert select_recent_games(df, target_games=2) == ["g3", "g2"]
    assert select_recent_games(df, target_games=3) == ["g3", "g2", "g1"]
    # Asking for more than exist just returns all of them.
    assert select_recent_games(df, target_games=1000) == ["g3", "g2", "g1"]


def test_restrict_to_recent_games_drops_older_games_but_keeps_all_patches_within_target():
    df = _make_games_df_for_patch_test()
    empty_bans = pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])

    # target_games=2 drops the oldest game (g1/14.1/OldPatchChamp) but keeps
    # g2 and g3, which happen to span two different patches -- patchesUsed
    # is derived from whatever patches survive, not the other way around.
    restricted_games, restricted_bans, patches_used = restrict_to_recent_games(
        df, empty_bans, target_games=2
    )
    assert patches_used == ["14.3", "14.2"]
    assert set(restricted_games["gameid"]) == {"g2", "g3"}
    assert "OldPatchChamp" not in restricted_games["champion"].values

    # target_games=1000 (>> available games) keeps everything, regardless
    # of patch -- this is the "more games, no matter the patch" behavior.
    restricted_games_all, _, patches_used_all = restrict_to_recent_games(
        df, empty_bans, target_games=1000
    )
    assert set(restricted_games_all["gameid"]) == {"g1", "g2", "g3"}
    assert patches_used_all == ["14.3", "14.2", "14.1"]
    assert "OldPatchChamp" in restricted_games_all["champion"].values


def test_old_excluded_game_does_not_affect_champion_rating(config: PipelineConfig):
    """A champion whose only game falls outside the target-games window
    should end up with zero decayed pro games and a rating driven purely
    by its solo queue prior -- the excluded pro game must have literally
    zero influence."""
    df = _make_games_df_for_patch_test()
    # Give OldPatchChamp a 100% raw pro win rate on the excluded game; if
    # the restriction leaked through, this would push its rating up.
    restricted_config = PipelineConfig(
        patch_half_life_days=config.patch_half_life_days,
        solo_queue_weight=config.solo_queue_weight,
        prior_games=config.prior_games,
        pro_window_days=config.pro_window_days,
        global_mean=config.global_mean,
        target_training_games=2,  # excludes g1, where OldPatchChamp lives
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

    # OldPatchChamp still appears (every champion does, via the full roster
    # union -- see champions.py), but its excluded game must have zero
    # influence: proGames should be 0 and its rating should be driven
    # purely by its solo-queue prior (0.5 here), not the leaked 100% pro WR.
    assert "OldPatchChamp" in result_df.index
    assert result_df.loc["OldPatchChamp", "proGames"] == 0
    assert result_df.loc["OldPatchChamp", "blendedWinRate"] == pytest.approx(0.5, abs=0.01)

    # RecentChamp (100% win rate, most recent game) should still be there
    # with a strength score above the neutral prior (it actually won).
    assert "RecentChamp" in result_df.index
    assert result_df.loc["RecentChamp", "proGames"] >= 1
    assert result_df.loc["RecentChamp", "strengthScore"] > 0


def test_solo_queue_keys_are_canonicalized_to_match_game_names():
    """Regression: a solo-queue source that spells a champion differently
    ("Jarvan Iv") than the canonical game name ("Jarvan IV") must still join
    onto the SAME champion row -- not drop the prior and emit a phantom
    duplicate champion. This is the game<->solo-queue split-key bug that the
    game-side name canonicalization would otherwise have relocated here."""
    ref = pd.Timestamp("2026-06-15", tz="UTC")
    games = pd.DataFrame(
        [
            {"gameid": "g1", "date": ref, "patch": "16.12", "champion": "Jarvan IV",
             "result": 1, "position": "jungle", "side": "Blue", "team": "T1", "league": "LCK"},
            {"gameid": "g2", "date": ref, "patch": "16.12", "champion": "Jarvan IV",
             "result": 0, "position": "jungle", "side": "Blue", "team": "T2", "league": "LCK"},
        ]
    )
    empty_bans = pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])
    # Solo source uses the MANGLED spelling that naive title-casing produced.
    solo_winrates = {"Jarvan Iv": (0.55, 4000)}

    result = compute_champion_features(
        games, empty_bans, solo_winrates, PipelineConfig(global_mean=0.5),
        reference_date=ref,
    )

    # The mangled solo key must NOT create a phantom champion row...
    assert "Jarvan Iv" not in result.index
    # ...and the canonical champion carries BOTH the pro games AND the solo prior.
    assert "Jarvan IV" in result.index
    row = result.loc["Jarvan IV"]
    assert row["proGames"] > 0
    assert row["soloGames"] == 4000
    assert row["soloWinRate"] == pytest.approx(0.55)


def test_target_training_games_validation():
    with pytest.raises(ValueError):
        PipelineConfig(target_training_games=0)
    with pytest.raises(ValueError):
        PipelineConfig(target_training_games=-1)
    # Valid values should not raise.
    PipelineConfig(target_training_games=1)
    PipelineConfig(target_training_games=1000)


# ---------------------------------------------------------------------------
# International-event weighting (MSI/Worlds/EWC weighted up vs regular season)
# ---------------------------------------------------------------------------


def test_compute_wr_strength_basic_and_loo():
    from compstrength_pipeline.features import compute_wr_strength

    ref = pd.Timestamp("2026-06-15", tz="UTC")
    rows = []
    # Ahri: 2 wins, 1 loss (blue side); Zed: mirror (red side) 1 win, 2 losses.
    for i, ahri_win in enumerate([1, 1, 0]):
        for side, champ, res in (("Blue", "Ahri", ahri_win), ("Red", "Zed", 1 - ahri_win)):
            rows.append(
                {"gameid": f"g{i}", "date": ref, "patch": "16.12", "league": "LCK",
                 "side": side, "champion": champ, "result": res, "position": "mid"}
            )
    games = pd.DataFrame(rows)
    cfg = PipelineConfig()

    strength, loo = compute_wr_strength(games, ref, cfg)
    # Winner above 0 (>50% shrunk WR), loser below, symmetric.
    assert strength["Ahri"] > 0 > strength["Zed"]
    assert strength["Ahri"] == pytest.approx(-strength["Zed"])
    # Shrinkage: 2/3 raw winrate pulled well toward 0.5 by 25 pseudo-games.
    raw_edge = (2 / 3 - 0.5) * cfg.strength_feature_scale
    assert 0 < strength["Ahri"] < raw_edge

    # LOO: every training game's diff must EXCLUDE its own result. g2 is
    # Ahri's loss, so leaving it out RAISES Ahri's winrate -> the LOO diff
    # for g2 (blue=Ahri) is HIGHER than for g0 (a win, whose exclusion
    # lowers it).
    assert set(loo) == {"g0", "g1", "g2"}
    assert loo["g2"] > loo["g0"]


def test_compute_wr_strength_empty():
    from compstrength_pipeline.features import compute_wr_strength

    empty = pd.DataFrame(columns=["gameid", "date", "side", "champion", "result", "position"])
    strength, loo = compute_wr_strength(empty, pd.Timestamp("2026-01-01", tz="UTC"), PipelineConfig())
    assert strength == {} and loo == {}


def test_apply_premier_league_weighting_hits_target_share():
    from compstrength_pipeline.features import (
        apply_premier_league_weighting,
        league_weight_multiplier,
    )

    ref = pd.Timestamp("2026-06-15", tz="UTC")
    # Same date/patch for every game -> uniform base weights, so shares are
    # just game counts: 2 premier (LCK/LPL) vs 6 rest.
    rows = []
    for i, lg in enumerate(["LCK", "LPL"] + ["LCKC"] * 6):
        rows.append(
            {"gameid": f"g{i}", "date": ref, "patch": "16.12", "league": lg,
             "champion": "Ahri"}
        )
    games = pd.DataFrame(rows)
    cfg = PipelineConfig(premier_league_target_share=0.7)

    weighted = apply_premier_league_weighting(cfg, games, ref)
    m = weighted.major_league_weight_multiplier
    assert weighted.major_leagues == cfg.premier_leagues
    # Verify the solved multiplier actually yields a 70% premier share.
    w_premier = 2 * m
    w_rest = 6.0
    assert w_premier / (w_premier + w_rest) == pytest.approx(0.7)
    # ...and it flows through the row-level weighting function.
    assert league_weight_multiplier(
        "LCK", cfg.international_leagues, cfg.international_weight_multiplier,
        weighted.major_leagues, m,
    ) == pytest.approx(m)


def test_apply_premier_league_weighting_disabled_or_absent():
    from compstrength_pipeline.features import apply_premier_league_weighting

    ref = pd.Timestamp("2026-06-15", tz="UTC")
    games = pd.DataFrame(
        [{"gameid": "g0", "date": ref, "patch": "16.12", "league": "LCKC",
          "champion": "Ahri"}]
    )
    # Disabled -> unchanged config object.
    off = PipelineConfig(premier_league_target_share=None)
    assert apply_premier_league_weighting(off, games, ref) is off
    # No premier games in window -> warn and leave weights unchanged.
    on = PipelineConfig(premier_league_target_share=0.7)
    with pytest.warns(UserWarning, match="no premier-league games"):
        out = apply_premier_league_weighting(on, games, ref)
    assert out.major_league_weight_multiplier == on.major_league_weight_multiplier


def test_league_weight_multiplier_tiers():
    from compstrength_pipeline.features import league_weight_multiplier

    intl = frozenset({"MSI", "WLDS"})
    majors = frozenset({"LCK", "LPL", "LEC"})
    # International events take the international multiplier (never the major
    # one, even though they're also top-tier -- multipliers don't stack).
    assert league_weight_multiplier("MSI", intl, 1.5, majors, 2.0) == 1.5
    # Tier-1 regional leagues take the major multiplier, case-insensitively.
    assert league_weight_multiplier("lck", intl, 1.5, majors, 2.0) == 2.0
    assert league_weight_multiplier(" LEC ", intl, 1.5, majors, 2.0) == 2.0
    # Everything else (minor/academy leagues, NaN) stays at 1.0.
    assert league_weight_multiplier("LCKC", intl, 1.5, majors, 2.0) == 1.0
    assert league_weight_multiplier(float("nan"), intl, 1.5, majors, 2.0) == 1.0
    # Defaults (no majors configured) reduce to the old international-only rule.
    assert league_weight_multiplier("LCK", intl, 1.5) == 1.0


def test_international_league_multiplier_matches_case_insensitively():
    intl = frozenset({"MSI", "WORLDS"})
    assert international_league_multiplier("msi", intl, 1.5) == 1.5
    assert international_league_multiplier("MSI", intl, 1.5) == 1.5
    assert international_league_multiplier(" Worlds ", intl, 1.5) == 1.5
    assert international_league_multiplier("LCK", intl, 1.5) == 1.0
    # Missing/non-string league values never get boosted.
    assert international_league_multiplier(float("nan"), intl, 1.5) == 1.0
    assert international_league_multiplier(None, intl, 1.5) == 1.0


def test_msi_games_are_weighted_up_relative_to_regular_season(config: PipelineConfig):
    reference_date = pd.Timestamp("2026-04-15", tz="UTC")
    # Two otherwise-identical champions: one's only game is regular season,
    # the other's only game is MSI, both on the same day. With a >1.0
    # international multiplier, the MSI game must count for strictly more
    # decayed "pseudo-games" even though both are single, same-day games.
    regular = pd.DataFrame(
        [{"date": reference_date, "result": 1, "league": "LCK"}]
    )
    international = pd.DataFrame(
        [{"date": reference_date, "result": 1, "league": "MSI"}]
    )

    regular_games, _ = compute_decayed_pro_stats(
        regular,
        reference_date,
        config.patch_half_life_days,
        international_leagues=config.international_leagues,
        international_weight_multiplier=config.international_weight_multiplier,
    )
    msi_games, _ = compute_decayed_pro_stats(
        international,
        reference_date,
        config.patch_half_life_days,
        international_leagues=config.international_leagues,
        international_weight_multiplier=config.international_weight_multiplier,
    )

    assert msi_games == pytest.approx(regular_games * config.international_weight_multiplier)


def test_international_multiplier_of_one_disables_boost(config: PipelineConfig):
    reference_date = pd.Timestamp("2026-04-15", tz="UTC")
    games = pd.DataFrame([{"date": reference_date, "result": 1, "league": "MSI"}])

    boosted, _ = compute_decayed_pro_stats(
        games, reference_date, config.patch_half_life_days,
        international_leagues=config.international_leagues,
        international_weight_multiplier=1.0,
    )
    unboosted, _ = compute_decayed_pro_stats(
        games, reference_date, config.patch_half_life_days,
    )
    assert boosted == pytest.approx(unboosted)
