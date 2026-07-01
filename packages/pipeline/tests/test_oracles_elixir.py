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


# ---------------------------------------------------------------------------
# Past-season cache (COMPSTRENGTH_OE_CACHE_DIR)
# ---------------------------------------------------------------------------

_FAKE_CSV = "gameid,date,patch,side,position,champion,result\n" + "x" * 1_100_000


def test_past_season_served_from_cache_without_downloading(tmp_path, monkeypatch):
    """A valid cached past-season CSV short-circuits all download attempts."""
    from compstrength_pipeline.sources import oracles_elixir as oe

    cache_dir = tmp_path / "oecache"
    cache_dir.mkdir()
    cached = cache_dir / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
    cached.write_text(_FAKE_CSV)
    monkeypatch.setenv("COMPSTRENGTH_OE_CACHE_DIR", str(cache_dir))

    def boom(*args, **kwargs):  # any download attempt = test failure
        raise AssertionError("download attempted despite valid cache")

    monkeypatch.setattr(oe, "_download_via_gdown", boom)
    monkeypatch.setattr(oe, "_download_via_requests", boom)
    monkeypatch.setattr(oe, "_download_via_url", boom)

    result = oe.download_oracles_elixir_csv(2025)
    assert result == cached


def test_successful_past_season_download_is_saved_to_cache(tmp_path, monkeypatch):
    from compstrength_pipeline.sources import oracles_elixir as oe

    cache_dir = tmp_path / "oecache"  # not created yet -> mkdir path exercised
    monkeypatch.setenv("COMPSTRENGTH_OE_CACHE_DIR", str(cache_dir))

    def fake_gdown(file_id, dest):
        Path(dest).write_text(_FAKE_CSV)
        return True

    monkeypatch.setattr(oe, "_download_via_gdown", fake_gdown)

    result = oe.download_oracles_elixir_csv(2025, dest=tmp_path / "dl.csv")
    assert result.exists()
    cached = cache_dir / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
    assert cached.exists() and cached.read_text() == _FAKE_CSV


def test_current_season_never_served_from_cache(tmp_path, monkeypatch):
    """The current year's file updates daily -- a cached copy must be ignored
    and a real download attempted."""
    from datetime import datetime, timezone

    from compstrength_pipeline.sources import oracles_elixir as oe

    current_year = datetime.now(timezone.utc).year
    cache_dir = tmp_path / "oecache"
    cache_dir.mkdir()
    stale = cache_dir / f"{current_year}_LoL_esports_match_data_from_OraclesElixir.csv"
    stale.write_text(_FAKE_CSV)
    monkeypatch.setenv("COMPSTRENGTH_OE_CACHE_DIR", str(cache_dir))

    fresh_marker = "gameid,fresh\n" + "y" * 1_100_000
    downloads = []

    def fake_gdown(file_id, dest):
        downloads.append(file_id)
        Path(dest).write_text(fresh_marker)
        return True

    monkeypatch.setattr(oe, "_download_via_gdown", fake_gdown)

    result = oe.download_oracles_elixir_csv(current_year, dest=tmp_path / "dl.csv")
    assert downloads, "current season must be re-downloaded, not cache-served"
    assert result != stale
    assert result.read_text() == fresh_marker


def test_corrupt_cache_is_ignored_and_redownloaded(tmp_path, monkeypatch):
    from compstrength_pipeline.sources import oracles_elixir as oe

    cache_dir = tmp_path / "oecache"
    cache_dir.mkdir()
    corrupt = cache_dir / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
    corrupt.write_text("<html>quota page</html>" + "x" * 1_100_000)  # no gameid header
    monkeypatch.setenv("COMPSTRENGTH_OE_CACHE_DIR", str(cache_dir))

    def fake_gdown(file_id, dest):
        Path(dest).write_text(_FAKE_CSV)
        return True

    monkeypatch.setattr(oe, "_download_via_gdown", fake_gdown)

    with pytest.warns(UserWarning, match="corrupt cached"):
        result = oe.download_oracles_elixir_csv(2025, dest=tmp_path / "dl.csv")
    assert result.read_text() == _FAKE_CSV
    # And the good download replaced the corrupt cache entry.
    assert corrupt.read_text() == _FAKE_CSV
