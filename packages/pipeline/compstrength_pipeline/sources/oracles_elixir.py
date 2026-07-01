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
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from compstrength_pipeline.config import ORACLES_ELIXIR_CSV_URL_TEMPLATE

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

    return pd.read_csv(io.StringIO(response.text), low_memory=False)


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
        raw = pd.read_csv(local_path, low_memory=False)

    return _normalize_player_games(raw)


def fetch_oracles_elixir_for_year(year: int) -> pd.DataFrame:
    """Convenience wrapper: fetch the live per-year CSV for ``year``.

    Uses :data:`compstrength_pipeline.config.ORACLES_ELIXIR_CSV_URL_TEMPLATE`.
    Best-effort / not exercised over the network in this sandbox -- see
    module docstring.
    """
    url = ORACLES_ELIXIR_CSV_URL_TEMPLATE.format(year=year)
    return fetch_oracles_elixir(url)


def _normalize_player_games(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw Oracle's Elixir frame into the canonical player-game schema."""
    missing = [c for c in ("gameid", "position") if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Input does not look like an Oracle's Elixir export; missing "
            f"required columns: {missing}"
        )

    df = raw.copy()
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

    team_rows = raw[raw["position"].astype(str).str.lower() == _TEAM_ROW_POSITION].copy()
    if "teamname" in team_rows.columns and "team" not in team_rows.columns:
        team_rows = team_rows.rename(columns={"teamname": "team"})

    available_ban_cols = [c for c in _BAN_COLUMNS if c in team_rows.columns]
    if not available_ban_cols:
        return pd.DataFrame(columns=CANONICAL_BAN_COLUMNS)

    id_cols = ["gameid", "team"]
    long = team_rows.melt(
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
