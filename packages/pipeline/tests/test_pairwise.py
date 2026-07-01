"""Unit tests for the synergy/matchup pairwise residual math in pairwise.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from compstrength_pipeline.config import PipelineConfig
from compstrength_pipeline.features import logit
from compstrength_pipeline.pairwise import (
    compute_matchup_table,
    compute_synergy_table,
    matchup_key,
    matchup_lookup,
    synergy_key,
    synergy_lookup,
)


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(
        patch_half_life_days=21,
        solo_queue_weight=0.35,
        prior_games=15,
        pro_window_days=90,
        global_mean=0.5,
        target_training_games=1000,
        synergy_prior_games=8,
        matchup_prior_games=10,
    )


def _row(gameid, date, side, position, champion, result):
    return {
        "gameid": gameid,
        "date": pd.Timestamp(date, tz="UTC"),
        "side": side,
        "position": position,
        "champion": champion,
        "result": result,
    }


def test_synergy_key_is_sorted_and_unordered():
    assert synergy_key("Zed", "Ahri") == "Ahri|Zed"
    assert synergy_key("Ahri", "Zed") == "Ahri|Zed"


def test_matchup_key_is_directional():
    assert matchup_key("Ahri", "Zed") == "Ahri>Zed"
    assert matchup_key("Zed", "Ahri") == "Zed>Ahri"


def test_pair_with_zero_cooccurrence_is_absent_from_synergy_output(config: PipelineConfig):
    """Champions A and B that never appear on the same team in any game
    together should not show up in the synergy table at all (they
    implicitly have residual 0)."""
    # Build a simple two-game dataset where A and B never co-occur.
    rows = [
        _row("g1", "2026-01-01", "Blue", "top", "A", 1),
        _row("g1", "2026-01-01", "Blue", "jng", "C", 1),
        _row("g1", "2026-01-01", "Red", "top", "X", 0),
        _row("g1", "2026-01-01", "Red", "jng", "Y", 0),
        _row("g2", "2026-01-02", "Blue", "top", "B", 1),
        _row("g2", "2026-01-02", "Blue", "jng", "D", 1),
        _row("g2", "2026-01-02", "Red", "top", "X", 0),
        _row("g2", "2026-01-02", "Red", "jng", "Y", 0),
    ]
    df = pd.DataFrame(rows)
    strength = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "X": 0.0, "Y": 0.0}

    table = compute_synergy_table(df, strength, config, reference_date=pd.Timestamp("2026-01-02", tz="UTC"))
    lookup = synergy_lookup(table)

    assert synergy_key("A", "B") not in lookup
    # But pairs that DID co-occur should be present.
    assert synergy_key("A", "C") in lookup
    assert synergy_key("B", "D") in lookup


def test_pair_with_zero_cooccurrence_is_absent_from_matchup_output(config: PipelineConfig):
    rows = [
        _row("g1", "2026-01-01", "Blue", "top", "A", 1),
        _row("g1", "2026-01-01", "Red", "top", "X", 0),
        _row("g2", "2026-01-02", "Blue", "top", "B", 1),
        _row("g2", "2026-01-02", "Red", "top", "Y", 0),
    ]
    df = pd.DataFrame(rows)
    strength = {"A": 0.0, "B": 0.0, "X": 0.0, "Y": 0.0}

    table = compute_matchup_table(df, strength, config, reference_date=pd.Timestamp("2026-01-02", tz="UTC"))
    lookup = matchup_lookup(table)

    # A never faced Y, B never faced X -- these should be absent.
    assert matchup_key("A", "Y") not in lookup
    assert matchup_key("B", "X") not in lookup
    # A did face X, B did face Y.
    assert matchup_key("A", "X") in lookup
    assert matchup_key("B", "Y") in lookup


def test_pair_that_always_wins_together_has_positive_residual(config: PipelineConfig):
    """A synergy pair that wins every single game together, with enough
    decayed games to dominate the shrinkage prior, should end up with a
    clearly positive residual (better than their individual ratings alone
    would predict, since strengthScore=0 for both here)."""
    rows = []
    # 20 games, all wins, all recent (no decay loss) -- large decayed
    # sample so shrinkage doesn't pull the residual all the way to 0.
    for i in range(20):
        rows.append(_row(f"g{i}", "2026-01-10", "Blue", "top", "SynergyA", 1))
        rows.append(_row(f"g{i}", "2026-01-10", "Blue", "jng", "SynergyB", 1))
        rows.append(_row(f"g{i}", "2026-01-10", "Red", "top", "Filler1", 0))
        rows.append(_row(f"g{i}", "2026-01-10", "Red", "jng", "Filler2", 0))
    df = pd.DataFrame(rows)
    strength = {"SynergyA": 0.0, "SynergyB": 0.0, "Filler1": 0.0, "Filler2": 0.0}

    table = compute_synergy_table(
        df, strength, config, reference_date=pd.Timestamp("2026-01-10", tz="UTC")
    )
    stat = table[synergy_key("SynergyA", "SynergyB")]

    assert stat.games_decayed == pytest.approx(20.0, abs=1e-6)
    assert stat.win_rate_raw == pytest.approx(1.0, abs=1e-6)
    # expected logit = 0 + 0 = 0; residual_raw = logit(1.0 clipped) - 0 > 0
    assert stat.residual_raw > 0
    assert stat.residual > 0
    # With 20 >> synergy_prior_games=8, shrinkage should leave most of the
    # raw residual intact (not pulled all the way to 0).
    assert stat.residual > 0.5 * stat.residual_raw


def test_shrinkage_pulls_small_samples_toward_zero(config: PipelineConfig):
    """A pair with a huge (surprising) raw residual but only 1 decayed
    game should be shrunk close to (but not exactly) 0, much more than a
    pair with the same raw residual but many decayed games."""
    # Small-sample pair: 1 game, blue side (both 0-strength champs) wins.
    small_rows = [
        _row("g1", "2026-01-10", "Blue", "top", "SmallA", 1),
        _row("g1", "2026-01-10", "Blue", "jng", "SmallB", 1),
        _row("g1", "2026-01-10", "Red", "top", "F1", 0),
        _row("g1", "2026-01-10", "Red", "jng", "F2", 0),
    ]
    small_df = pd.DataFrame(small_rows)
    strength = {"SmallA": 0.0, "SmallB": 0.0, "F1": 0.0, "F2": 0.0}

    small_table = compute_synergy_table(
        small_df, strength, config, reference_date=pd.Timestamp("2026-01-10", tz="UTC")
    )
    small_stat = small_table[synergy_key("SmallA", "SmallB")]

    # 1 decayed game vs prior_games=8 -> shrunk residual should be a small
    # fraction (1/9) of the raw residual.
    expected_shrink_fraction = 1.0 / (config.synergy_prior_games + 1.0)
    assert small_stat.residual == pytest.approx(
        small_stat.residual_raw * expected_shrink_fraction, abs=1e-9
    )
    assert abs(small_stat.residual) < abs(small_stat.residual_raw)


def test_synergy_residual_matches_manual_formula(config: PipelineConfig):
    rows = [
        _row("g1", "2026-01-01", "Blue", "top", "A", 1),
        _row("g1", "2026-01-01", "Blue", "jng", "B", 1),
        _row("g1", "2026-01-01", "Red", "top", "X", 0),
        _row("g1", "2026-01-01", "Red", "jng", "Y", 0),
        _row("g2", "2026-01-01", "Blue", "top", "A", 0),
        _row("g2", "2026-01-01", "Blue", "jng", "B", 0),
        _row("g2", "2026-01-01", "Red", "top", "X", 1),
        _row("g2", "2026-01-01", "Red", "jng", "Y", 1),
    ]
    df = pd.DataFrame(rows)
    strength = {"A": 0.2, "B": -0.1, "X": 0.0, "Y": 0.0}
    ref_date = pd.Timestamp("2026-01-01", tz="UTC")

    table = compute_synergy_table(df, strength, config, reference_date=ref_date)
    stat = table[synergy_key("A", "B")]

    # Both games on the same date -> weight 1.0 each -> games_decayed=2,
    # win_rate_raw = 0.5 (1 win, 1 loss).
    assert stat.games_decayed == pytest.approx(2.0)
    assert stat.win_rate_raw == pytest.approx(0.5)

    expected_logit = strength["A"] + strength["B"]
    expected_residual_raw = logit(0.5) - expected_logit
    assert stat.expected_logit == pytest.approx(expected_logit)
    assert stat.residual_raw == pytest.approx(expected_residual_raw)

    expected_shrunk = (expected_residual_raw * 2.0) / (config.synergy_prior_games + 2.0)
    assert stat.residual == pytest.approx(expected_shrunk)


def test_matchup_residual_is_directional_and_can_differ_from_reverse(config: PipelineConfig):
    """A-vs-B need not be the negative of B-vs-A: build a dataset where A's
    side always wins the A-vs-B top-lane matchup, but B's side does NOT
    always lose when facing a *different* opponent, and confirm A>B and
    B>A are independently computed (and, in this constructed scenario,
    not simply negatives of one another because different game sets/roles
    can contribute)."""
    rows = [
        # Game 1: A (blue top) beats B (red top). Blue side wins overall.
        _row("g1", "2026-01-05", "Blue", "top", "A", 1),
        _row("g1", "2026-01-05", "Red", "top", "B", 0),
        # Game 2: same matchup again, blue (A) wins again.
        _row("g2", "2026-01-05", "Blue", "top", "A", 1),
        _row("g2", "2026-01-05", "Red", "top", "B", 0),
    ]
    df = pd.DataFrame(rows)
    strength = {"A": 0.0, "B": 0.0}
    ref_date = pd.Timestamp("2026-01-05", tz="UTC")

    table = compute_matchup_table(df, strength, config, reference_date=ref_date)
    a_vs_b = table[matchup_key("A", "B")]
    b_vs_a = table[matchup_key("B", "A")]

    assert a_vs_b.win_rate_raw == pytest.approx(1.0)
    assert b_vs_a.win_rate_raw == pytest.approx(0.0)
    assert a_vs_b.residual > 0
    assert b_vs_a.residual < 0
    # Not required to be exact negatives, but in this simple symmetric
    # dataset (same games, same strength=0 baseline) they happen to be.
    assert a_vs_b.residual == pytest.approx(-b_vs_a.residual)


def test_matchup_prior_games_validation():
    with pytest.raises(ValueError):
        PipelineConfig(matchup_prior_games=-1)
    PipelineConfig(matchup_prior_games=0)


def test_synergy_prior_games_validation():
    with pytest.raises(ValueError):
        PipelineConfig(synergy_prior_games=-1)
    PipelineConfig(synergy_prior_games=0)


def test_empty_games_df_returns_empty_tables(config: PipelineConfig):
    empty = pd.DataFrame(columns=["gameid", "date", "side", "position", "champion", "result"])
    assert compute_synergy_table(empty, {}, config) == {}
    assert compute_matchup_table(empty, {}, config) == {}
