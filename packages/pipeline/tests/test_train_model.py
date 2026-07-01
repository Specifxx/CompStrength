"""Tests for the meta-presence (pick+ban rate) model feature."""

from __future__ import annotations

import pandas as pd

from compstrength_pipeline.train_model import (
    FEATURE_COLUMNS,
    build_training_frame,
    compute_presence_diff,
    train_model,
)


def test_compute_presence_diff():
    presence = {"Ahri": 0.6, "Zed": 0.2, "Jinx": 0.4}
    # Blue has 0.6 + 0.2 = 0.8, red has 0.4 + (missing -> 0) = 0.4.
    assert compute_presence_diff(presence, ["Ahri", "Zed"], ["Jinx", "Unknown"]) == (
        0.8 - 0.4
    )
    # Empty presence map -> everything 0.
    assert compute_presence_diff({}, ["Ahri"], ["Zed"]) == 0.0


def _tiny_games(n_games: int = 30) -> pd.DataFrame:
    """Alternating-outcome 5v5 games over two champion pools."""
    rows = []
    blue_pool = ["A1", "A2", "A3", "A4", "A5"]
    red_pool = ["B1", "B2", "B3", "B4", "B5"]
    positions = ["top", "jng", "mid", "bot", "sup"]
    for g in range(n_games):
        blue_win = g % 2
        for i in range(5):
            rows.append(
                {"gameid": f"g{g}", "side": "Blue", "position": positions[i],
                 "champion": blue_pool[i], "result": blue_win,
                 "date": pd.Timestamp("2026-06-01", tz="UTC")}
            )
            rows.append(
                {"gameid": f"g{g}", "side": "Red", "position": positions[i],
                 "champion": red_pool[i], "result": 1 - blue_win,
                 "date": pd.Timestamp("2026-06-01", tz="UTC")}
            )
    return pd.DataFrame(rows)


def test_training_frame_has_presence_column():
    games = _tiny_games()
    presence = {"A1": 0.9, "B1": 0.1}
    frame = build_training_frame(games, {}, None, None, presence)
    assert "presence_diff" in frame.columns
    assert "presence_diff" in FEATURE_COLUMNS
    # Every game: blue presence 0.9, red 0.1 -> diff 0.8.
    assert (frame["presence_diff"] == 0.8).all()


def test_presence_feature_off_gives_zero_weight():
    """With no presence map the feature column is all zeros, so the fitted
    presenceWeight must be exactly 0 and the model is backward compatible."""
    games = _tiny_games()
    result = train_model(games, {"A1": 0.1, "B1": -0.1})
    assert result.coefficients["presenceWeight"] == 0.0
    # All required coefficient keys present.
    for key in ("scoreDiffWeight", "synergyWeight", "matchupWeight",
                "presenceWeight", "blueSideBias", "intercept"):
        assert key in result.coefficients
