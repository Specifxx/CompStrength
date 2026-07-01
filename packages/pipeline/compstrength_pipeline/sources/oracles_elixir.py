"""Fetcher/normalizer for Oracle's Elixir competitive match data.

Oracle's Elixir (https://oracleselixir.com/tools/downloads) publishes one
wide-format CSV per year, refreshed daily, with ~150-160+ columns and
(nominally) 12 rows per game: 10 individual player rows plus 2 team-summary
rows (one per side). All 12 rows of a game share the same ``gameid``, which
is the primary key for grouping a game's rows together.

Key columns (per our schema research; see the project README for citations):

- Identifiers: ``gameid`` (joins all rows of a game), ``datacompleteness``
  ("complete"/"partial" -- partial rows are missing some stats and should
  generally be dropped), ``url``.
- Date/competition: ``date``, ``game`` (game number within a Bo3/Bo5 series),
  ``patch``, ``league``, ``year``, ``split``, ``playoffs``.
- Side/team/player: ``side`` ("Blue"/"Red"), ``position`` (one of
  "top"/"jng"/"mid"/"bot"/"sup" for player rows, or the literal string
  "team" for the two team-summary rows per game), ``playerid``,
  ``playername``, ``teamid``, ``teamname``.
- Draft: ``champion`` (per player row only). The two team rows carry the
  draft order info: ``ban1``..``ban5`` (bans in the order they were banned)
  and ``pick1``..``pick5`` (picks in draft order for that team).
- Result: ``result`` (1 = win, 0 = loss), plus many per-player/team combat
  and objective stat columns not modeled here.

This module is best-effort with respect to *live* fetching: network egress
to oracleselixir.com (and the S3 bucket that serves the actual CSVs) may be
blocked in some environments (e.g. this sandbox). When that happens we raise
a ``DataSourceUnavailableError`` with a clear, actionable message rather than
letting a low-level connection error propagate. The function works
unmodified against a local CSV path (e.g. a bundled test fixture) with no
network involved at all.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from compstrength_pipeline.config import (
    ORACLES_ELIXIR_DRIVE_DOWNLOAD_URL,
    ORACLES_ELIXIR_DRIVE_IDS,
    ORACLES_ELIXIR_MIRROR_URLS,
)

# A real Oracle's Elixir season CSV is tens of MB; anything much smaller is
# almost certainly a Google-Drive HTML quota/interstitial page rather than
# the data, so we reject it (see download_oracles_elixir_csv).
_MIN_CSV_BYTES = 1_000_000

# Canonical output schema for the per-player-game table.
CANONICAL_GAME_COLUMNS = [
    "gameid",
    "date",
    "league",
    "patch",
    "side",
    "team",
    "player",
    "position",
    "champion",
    "result",
]

# Canonical output schema for the normalized bans table.
CANONICAL_BAN_COLUMNS = ["gameid", "team", "champion", "ban_number"]

# Raw Oracle's Elixir column names we depend on.
_BAN_COLUMNS = [f"ban{i}" for i in range(1, 6)]
_TEAM_ROW_POSITION = "team"

# CRITICAL: force ``patch`` to be read as a string. Oracle's Elixir stores it
# zero-padded ("16.01".."16.10".."16.12"), but pandas otherwise auto-detects
# the all-numeric column as float64, which is silently destructive:
#   - "16.10" (a huge, near-newest patch) becomes the float 16.1, which parses
#     to (16, 1) -- i.e. it looks like an *ancient* patch, corrupting every
#     patch-recency comparison and the min_patch floor.
#   - features.parse_patch() only accepts strings, so a float column makes the
#     25.1 floor a silent no-op and the per-patch decay ordering unreliable.
# Reading it as text preserves the exact "16.10" form so parse_patch, the
# floor, and the patch-recency decay all work correctly.
_READ_CSV_KWARGS: dict = {"low_memory": False, "dtype": {"patch": str}}


class DataSourceUnavailableError(RuntimeError):
    """Raised when a live data source cannot be reached.

    This is intentionally a distinct, easy-to-catch exception type so
    callers (e.g. the CLI in build.py) can print a helpful message about
    network policy instead of a raw ``requests`` traceback.
    """


def _is_url(path_or_url: str) -> bool:
    parsed = urlparse(str(path_or_url))
    return parsed.scheme in ("http", "https")


def _read_csv_from_url(url: str) -> pd.DataFrame:
    """Best-effort download of a remote Oracle's Elixir CSV.

    Wrapped so that any network failure (blocked egress, DNS failure,
    timeout, non-200 response, etc.) is converted into a
    ``DataSourceUnavailableError`` with guidance, rather than leaking a raw
    ``requests``/``urllib`` exception. This function is expected to work
    unmodified in environments with open network egress (e.g. GitHub
    Actions); it is not exercised over the network in this sandbox.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a hard dependency
        raise DataSourceUnavailableError(
            "The 'requests' package is required to fetch remote Oracle's Elixir "
            "data. Install it via `pip install requests`."
        ) from exc

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        raise DataSourceUnavailableError(
            f"Could not fetch Oracle's Elixir CSV from {url!r}. This is expected "
            "in network-restricted sandboxes (oracleselixir.com / its S3 CDN is "
            "blocked here). This code path is designed to work unmodified in an "
            "unrestricted environment such as GitHub Actions. Original error: "
            f"{exc!r}"
        ) from exc

    return pd.read_csv(io.StringIO(response.text), **_READ_CSV_KWARGS)


def fetch_oracles_elixir(path_or_url: str) -> pd.DataFrame:
    """Load Oracle's Elixir match data and normalize it into a canonical schema.

    Args:
        path_or_url: Either a local filesystem path to a CSV (e.g. a test
            fixture, or a downloaded snapshot) or an ``http(s)://`` URL
            pointing at a live Oracle's Elixir export.

    Returns:
        A DataFrame with one row per player-game and exactly the columns
        ``CANONICAL_GAME_COLUMNS`` = [gameid, date, league, patch, side,
        team, player, position, champion, result]. Team-summary rows
        (``position == "team"``) are excluded from this table since they
        have no ``champion``/``player``; use :func:`extract_bans` to pull
        ban data from the raw frame instead.

    Raises:
        DataSourceUnavailableError: if ``path_or_url`` is a URL and the
            network request fails (including because egress is blocked).
        FileNotFoundError: if ``path_or_url`` is a local path that does not
            exist.
    """
    if _is_url(path_or_url):
        raw = _read_csv_from_url(path_or_url)
    else:
        local_path = Path(path_or_url)
        if not local_path.exists():
            raise FileNotFoundError(f"Oracle's Elixir CSV not found at {local_path}")
        raw = pd.read_csv(local_path, **_READ_CSV_KWARGS)

    return _normalize_player_games(raw)


def download_oracles_elixir_csv(year: int, dest: str | Path | None = None) -> Path:
    """Download the Oracle's Elixir season CSV for ``year``.

    Tries, in order (first one that yields a valid CSV wins):
      1. Google Drive via ``gdown`` (freshest -- OE updates the Drive file
         ~daily -- but the shared file is often over Google's per-file
         download quota).
      2. Google Drive direct-download endpoint via ``requests``.
      3. Plain-HTTP GitHub mirrors (``config.ORACLES_ELIXIR_MIRROR_URLS``) --
         no per-file quota, reachable from essentially anywhere; may lag the
         Drive file by a few days.

    Each candidate is validated (a real season CSV is tens of MB and its
    header starts with ``gameid``); a too-small file or an HTML quota
    interstitial is rejected and the next candidate is tried, so garbage is
    never fed downstream.

    Args:
        year: Season year, e.g. ``2026``.
        dest: Where to write the CSV. Defaults to a temp file.

    Returns:
        The path to the downloaded CSV.

    Raises:
        DataSourceUnavailableError: if the year is unknown, or every source
            fails / returns something that isn't the real CSV.
    """
    file_id = ORACLES_ELIXIR_DRIVE_IDS.get(year)
    mirror_urls = ORACLES_ELIXIR_MIRROR_URLS.get(year, [])
    if file_id is None and not mirror_urls:
        raise DataSourceUnavailableError(
            f"No known Oracle's Elixir source for year {year}. Known Drive years: "
            f"{sorted(ORACLES_ELIXIR_DRIVE_IDS)}; mirror years: {sorted(ORACLES_ELIXIR_MIRROR_URLS)}."
        )

    import os

    # Persistent cache for PAST seasons (opt-in via COMPSTRENGTH_OE_CACHE_DIR;
    # the CI refresh workflow points this at an actions/cache-backed dir). A
    # finished season's CSV is frozen forever, so once ANY run manages to
    # download it -- the shared Drive file is frequently over Google's
    # per-file quota -- every later run reuses the cached copy instead of
    # gambling on Drive again. The CURRENT season's file updates daily and is
    # therefore never served from (or saved to) this cache.
    cache_dir = os.environ.get("COMPSTRENGTH_OE_CACHE_DIR")
    is_past_season = year < datetime.now(timezone.utc).year
    cache_path: Path | None = None
    if cache_dir and is_past_season:
        cache_path = (
            Path(os.path.expanduser(cache_dir))
            / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        )
        if cache_path.exists():
            try:
                _validate_downloaded_csv(cache_path, year, "past-season cache")
                print(f"Using cached Oracle's Elixir {year} CSV: {cache_path}")
                return cache_path
            except DataSourceUnavailableError as exc:
                warnings.warn(f"Ignoring corrupt cached {year} CSV: {exc}")

    if dest is None:
        fd, dest_str = tempfile.mkstemp(prefix=f"oracleselixir_{year}_", suffix=".csv")
        os.close(fd)
        dest = dest_str
    dest = Path(dest)

    # (label, download-callable) attempts, in priority order.
    attempts: list[tuple[str, "callable"]] = []
    if file_id is not None:
        attempts.append((f"gdown Drive id {file_id!r}", lambda: _download_via_gdown(file_id, dest)))
        attempts.append((f"requests Drive id {file_id!r}", lambda: _download_via_requests(file_id, dest)))
    for url in mirror_urls:
        attempts.append((f"mirror {url}", lambda u=url: _download_via_url(u, dest)))

    for label, fn in attempts:
        if not fn():
            continue
        try:
            _validate_downloaded_csv(dest, year, label)
        except DataSourceUnavailableError as exc:
            warnings.warn(f"Discarding {label}: {exc}")
            continue
        if cache_path is not None:
            # Frozen past season: persist so future runs never need Drive.
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dest, cache_path)
                print(f"Cached Oracle's Elixir {year} CSV for future runs: {cache_path}")
            except OSError as exc:  # cache failure must never fail the download
                warnings.warn(f"Could not cache {year} CSV: {exc}")
        return dest

    raise DataSourceUnavailableError(
        f"Could not obtain a valid Oracle's Elixir {year} CSV from any source "
        f"(tried: {[label for label, _ in attempts]}). In this dev sandbox "
        "drive.google.com is blocked; the GitHub raw mirror should work here "
        "and on GitHub Actions."
    )


def _download_via_url(url: str, dest: Path) -> bool:
    """Stream a plain-HTTP CSV (e.g. a raw.githubusercontent.com mirror) to
    ``dest``. Returns True on a non-trivial download, False on any failure."""
    try:
        import requests
    except ImportError:  # pragma: no cover
        return False
    try:
        with requests.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
    except Exception as exc:  # pragma: no cover - network path
        warnings.warn(f"Could not download Oracle's Elixir mirror {url!r}: {exc!r}")
        return False
    return dest.exists() and dest.stat().st_size > _MIN_CSV_BYTES


def _download_via_gdown(file_id: str, dest: Path) -> bool:
    """Best-effort download using the ``gdown`` package. Returns True on
    success, False on ANY failure (gdown not installed, or a Drive
    error such as the shared-file "too many users / quota" block) so the
    caller can fall through to the next strategy rather than aborting."""
    try:
        import gdown
    except ImportError:
        return False
    try:
        gdown.download(id=file_id, output=str(dest), quiet=True)
    except Exception as exc:  # pragma: no cover - network path
        warnings.warn(f"gdown could not download Drive id {file_id!r}: {exc!r}")
        return False
    return dest.exists() and dest.stat().st_size > _MIN_CSV_BYTES


def _download_via_requests(file_id: str, dest: Path) -> bool:
    """Fallback download via the drive.usercontent direct-download endpoint
    (``confirm=t`` skips the >25MB virus-scan interstitial). Returns True on
    success, False on any failure (so the caller can fall through). A too-
    small response is treated as failure (Drive serves a small HTML quota/
    interstitial page instead of the file when over quota)."""
    try:
        import requests
    except ImportError:  # pragma: no cover
        return False

    url = ORACLES_ELIXIR_DRIVE_DOWNLOAD_URL.format(file_id=file_id)
    try:
        with requests.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
    except Exception as exc:  # pragma: no cover - network path
        warnings.warn(f"requests could not download Drive id {file_id!r} from {url!r}: {exc!r}")
        return False
    return dest.exists() and dest.stat().st_size > _MIN_CSV_BYTES


def _validate_downloaded_csv(path: Path, year: int, source_label: str) -> None:
    """Reject a too-small file or an HTML interstitial masquerading as CSV."""
    size = path.stat().st_size if path.exists() else 0
    if size < _MIN_CSV_BYTES:
        raise DataSourceUnavailableError(
            f"Downloaded Oracle's Elixir {year} CSV via {source_label} is only "
            f"{size} bytes (< {_MIN_CSV_BYTES}); this is almost certainly a quota/"
            "interstitial HTML page or a truncated file, not the real data."
        )
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    if "gameid" not in header:
        raise DataSourceUnavailableError(
            f"Downloaded Oracle's Elixir {year} file via {source_label} does not "
            f"look like the CSV: header {header[:120]!r} is missing 'gameid'."
        )


def fetch_oracles_elixir_for_year(year: int) -> pd.DataFrame:
    """Convenience wrapper: download the live per-year CSV from Google Drive
    (see :func:`download_oracles_elixir_csv`) and normalize it.

    Not exercised over the network in this dev sandbox (drive.google.com is
    blocked here); designed to run unmodified on GitHub Actions.
    """
    csv_path = download_oracles_elixir_csv(year)
    return fetch_oracles_elixir(str(csv_path))


def _drop_incomplete(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop rows from games flagged partial/incomplete by Oracle's Elixir.

    Oracle's Elixir marks each row's ``datacompleteness`` as ``complete`` or
    ``partial`` (a partial/forfeit/remade game can still have 10 player rows
    with a valid ``result``, so the "exactly 10 rows" gate alone does not
    exclude it). We drop every row belonging to any game that has a
    non-``complete`` marker, so partial games never reach training. If the
    column is absent (e.g. the Leaguepedia source or the synthetic fixture),
    this is a no-op.
    """
    if "datacompleteness" not in raw.columns or "gameid" not in raw.columns:
        return raw
    completeness = raw["datacompleteness"].astype(str).str.strip().str.lower()
    incomplete_gameids = set(raw.loc[completeness != "complete", "gameid"])
    if not incomplete_gameids:
        return raw
    return raw[~raw["gameid"].isin(incomplete_gameids)].copy()


def _normalize_player_games(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw Oracle's Elixir frame into the canonical player-game schema."""
    missing = [c for c in ("gameid", "position") if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Input does not look like an Oracle's Elixir export; missing "
            f"required columns: {missing}"
        )

    df = _drop_incomplete(raw.copy())
    # Player rows only (team-summary rows carry no champion/player and are
    # handled separately by extract_bans()).
    player_rows = df[df["position"].astype(str).str.lower() != _TEAM_ROW_POSITION].copy()

    rename_map = {"teamname": "team", "playername": "player"}
    player_rows = player_rows.rename(columns=rename_map)

    for col in CANONICAL_GAME_COLUMNS:
        if col not in player_rows.columns:
            player_rows[col] = pd.NA

    normalized = player_rows[CANONICAL_GAME_COLUMNS].copy()

    # Standardize types/casing.
    normalized["champion"] = normalized["champion"].astype(str).str.strip()
    normalized["position"] = normalized["position"].astype(str).str.strip().str.lower()
    normalized["side"] = normalized["side"].astype(str).str.strip().str.title()
    normalized["result"] = pd.to_numeric(normalized["result"], errors="coerce")
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce", utc=True)

    return normalized.reset_index(drop=True)


def extract_bans(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract a normalized bans table from a raw Oracle's Elixir frame.

    Oracle's Elixir stores each team's 5 bans as wide columns (``ban1``..
    ``ban5``) on that team's *team-summary* row (``position == "team"``),
    one such row per side per game. This melts those wide columns into a
    long/tidy table.

    Args:
        raw: The raw (un-normalized) DataFrame as read directly from the
            Oracle's Elixir CSV (i.e. including the team-summary rows).

    Returns:
        DataFrame with columns ``CANONICAL_BAN_COLUMNS`` =
        [gameid, team, champion, ban_number]. Empty/NaN bans (e.g. a game
        with a "No Bans" ruleset) are dropped.
    """
    missing = [c for c in ("gameid", "position") if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Input does not look like an Oracle's Elixir export; missing "
            f"required columns: {missing}"
        )

    complete = _drop_incomplete(raw)
    team_rows = complete[
        complete["position"].astype(str).str.lower() == _TEAM_ROW_POSITION
    ].copy()
    if "teamname" in team_rows.columns and "team" not in team_rows.columns:
        team_rows = team_rows.rename(columns={"teamname": "team"})

    available_ban_cols = [c for c in _BAN_COLUMNS if c in team_rows.columns]
    if not available_ban_cols:
        return pd.DataFrame(columns=CANONICAL_BAN_COLUMNS)

    id_cols = ["gameid", "team"]
    # Team rows carry an (empty) "champion" column too, which would collide
    # with melt()'s value_name="champion" below -- select only the columns
    # we actually need before melting.
    long = team_rows[id_cols + available_ban_cols].melt(
        id_vars=id_cols,
        value_vars=available_ban_cols,
        var_name="ban_col",
        value_name="champion",
    )
    long["ban_number"] = long["ban_col"].str.replace("ban", "", regex=False).astype(int)
    long["champion"] = long["champion"].astype(str).str.strip()
    long = long[(long["champion"] != "") & (long["champion"].str.lower() != "nan")]

    return long[CANONICAL_BAN_COLUMNS].sort_values(["gameid", "team", "ban_number"]).reset_index(
        drop=True
    )
