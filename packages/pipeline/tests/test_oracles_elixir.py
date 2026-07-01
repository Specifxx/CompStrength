"""Tests for the Oracle's Elixir source: datacompleteness filtering and the
downloaded-file sanity checks (the live Google Drive download itself is only
exercised on GitHub Actions -- drive.google.com is blocked in the sandbox)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from compstrength_pipeline.sources.oracles_elixir import (
    DataSourceUnavailableError,
    _drop_incomplete,
    _validate_downloaded_csv,
    download_oracles_elixir_csv,
    extract_bans,
)


def _raw_game(gameid, completeness):
    """One synthetic OE game: 10 player rows + 2 team rows, all sharing a
    datacompleteness flag."""
    rows = []
    for side in ("Blue", "Red"):
        for pos in ("top", "jng", "mid", "bot", "sup"):
            rows.append(
                {"gameid": gameid, "datacompleteness": completeness, "position": pos,
                 "side": side, "champion": "Ahri", "result": 1 if side == "Blue" else 0,
                 "teamname": f"T_{side}"}
            )
        rows.append(
            {"gameid": gameid, "datacompleteness": completeness, "position": "team",
             "side": side, "champion": "", "result": 1 if side == "Blue" else 0,
             "teamname": f"T_{side}", "ban1": "Zed"}
        )
    return rows


def test_drop_incomplete_removes_whole_partial_game():
    raw = pd.DataFrame(_raw_game("complete1", "complete") + _raw_game("partial1", "partial"))
    kept = _drop_incomplete(raw)
    assert set(kept["gameid"]) == {"complete1"}
    # Every row of the partial game is gone, not just some.
    assert (kept["datacompleteness"] == "complete").all()


def test_drop_incomplete_noop_without_column():
    raw = pd.DataFrame([{"gameid": "g", "position": "top", "champion": "Ahri"}])
    out = _drop_incomplete(raw)
    assert len(out) == len(raw)


def test_extract_bans_excludes_partial_games():
    raw = pd.DataFrame(_raw_game("c", "complete") + _raw_game("p", "partial"))
    bans = extract_bans(raw)
    assert set(bans["gameid"]) == {"c"}


def test_validate_rejects_too_small_file(tmp_path: Path):
    small = tmp_path / "small.csv"
    small.write_text("gameid,side\nABC,Blue\n")  # valid header but tiny
    with pytest.raises(DataSourceUnavailableError, match="bytes"):
        _validate_downloaded_csv(small, 2026, "fakeid")


def test_validate_rejects_html_interstitial(tmp_path: Path):
    html = tmp_path / "page.html"
    # >1MB but an HTML page (no 'gameid' header) -- Drive quota interstitial.
    html.write_text("<html><body>Quota exceeded</body></html>\n" + "x" * 1_100_000)
    with pytest.raises(DataSourceUnavailableError, match="gameid"):
        _validate_downloaded_csv(html, 2026, "fakeid")


def test_validate_accepts_real_looking_csv(tmp_path: Path):
    csv = tmp_path / "ok.csv"
    csv.write_text("gameid,date,patch,side,position,champion,result\n" + "x" * 1_100_000)
    _validate_downloaded_csv(csv, 2026, "fakeid")  # should not raise


def test_download_unknown_year_raises():
    with pytest.raises(DataSourceUnavailableError, match="No known Oracle's Elixir"):
        download_oracles_elixir_csv(1999)
