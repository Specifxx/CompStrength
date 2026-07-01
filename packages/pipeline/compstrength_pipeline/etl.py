"""ETL cleaning/validation for normalized Oracle's Elixir game and ban tables.

Takes the canonical (but not yet validated) DataFrames produced by
``sources/oracles_elixir.py`` (or ``sources/leaguepedia.py`` as a
cross-check) and produces cleaned tables ready for feature computation:

- Drops incomplete games (a valid game must have exactly 10 player rows:
  5 per side).
- Parses/validates dates.
- Standardizes champion name casing (title case, trimmed whitespace) so
  that e.g. "kai'sa", "Kai'Sa", "KAI'SA" all collapse to one canonical
  spelling.
"""

from __future__ import annotations

import functools

import pandas as pd

from compstrength_pipeline.champions import get_full_champion_roster

EXPECTED_PLAYER_ROWS_PER_GAME = 10


@functools.lru_cache(maxsize=1)
def _canonical_name_by_casefold() -> dict[str, str]:
    """Map each known champion's casefolded name -> its canonical spelling.

    Built from the full champion roster (Data Dragon live, else the static
    fallback in ``champions.py``), which uses Riot's official casing
    ("Jarvan IV", "LeBlanc", "Kai'Sa", "Dr. Mundo", ...). This is the single
    source of truth we canonicalize raw champion strings against.
    """
    return {name.casefold(): name for name in get_full_champion_roster()}


def standardize_champion_name(name: object) -> object:
    """Canonicalize a SINGLE champion name to its roster spelling.

    Oracle's Elixir already uses Riot's official spellings, but naive
    ``str.title()`` corrupts irregular names -- "Jarvan IV" -> "Jarvan Iv",
    "LeBlanc" -> "Leblanc" -- which would then fail to match the roster's
    canonical spelling and split ALL of that champion's game/synergy/matchup
    data under a mangled key (while the correctly-spelled roster entry gets
    zero data). To avoid that, we first map the name (case-insensitively)
    onto the known roster's canonical spelling; only names the roster doesn't
    know fall back to title-casing, with the apostrophe special-case
    (Kai'Sa, Kog'Maw, Cho'Gath, Vel'Koz, Kha'Zix, Rek'Sai) preserved.

    This is deliberately the SINGLE canonicalization used for BOTH the game
    champion column (via :func:`_standardize_champion_name`) AND the
    solo-queue win-rate keys (see ``features.compute_champion_features``), so
    the two sides can never diverge onto different spellings of one champion.

    Non-string inputs (NaN/None -- real Oracle's Elixir data has them in empty
    ban slots / "No Ban" games) are returned unchanged.
    """
    if not isinstance(name, str):
        return name
    stripped = name.strip()
    # 1. Canonical roster match (case-insensitive) -- the common path for real
    #    data, and the fix for "Jarvan IV"/"LeBlanc"-style names.
    canon = _canonical_name_by_casefold().get(stripped.casefold())
    if canon is not None:
        return canon
    # 2. Unknown champion (e.g. a brand-new release not yet in the roster):
    #    title-case with apostrophe fixup so casing is at least consistent.
    titled = stripped.title()
    if "'" in titled:
        parts = titled.split("'")
        parts = [parts[0]] + [p[:1].upper() + p[1:] if p else p for p in parts[1:]]
        return "'".join(parts)
    return titled


def _standardize_champion_name(series: pd.Series) -> pd.Series:
    """Vectorized :func:`standardize_champion_name` over a champion-name column."""
    return series.map(standardize_champion_name)


def build_raw_tables(
    games_df: pd.DataFrame, bans_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and validate the normalized games/bans tables.

    Args:
        games_df: Canonical per-player-game DataFrame, as produced by
            ``sources.oracles_elixir.fetch_oracles_elixir`` (columns:
            gameid, date, league, patch, side, team, player, position,
            champion, result).
        bans_df: Canonical bans DataFrame, as produced by
            ``sources.oracles_elixir.extract_bans`` (columns: gameid,
            team, champion, ban_number).

    Returns:
        A tuple ``(clean_games_df, clean_bans_df)``:
        - ``clean_games_df`` contains only rows belonging to games with
          exactly 10 player rows, dates parsed as timezone-aware UTC
          timestamps, and standardized champion name casing.
        - ``clean_bans_df`` is filtered to only reference gameids that
          survived the games-table cleaning, with standardized champion
          name casing.

    Raises a ValueError if required columns are missing.
    """
    required_game_cols = {"gameid", "date", "champion", "result", "position"}
    missing = required_game_cols - set(games_df.columns)
    if missing:
        raise ValueError(f"games_df missing required columns: {sorted(missing)}")

    games = games_df.copy()

    # Parse dates defensively (idempotent if already parsed).
    games["date"] = pd.to_datetime(games["date"], errors="coerce", utc=True)

    # Standardize champion casing.
    games["champion"] = _standardize_champion_name(games["champion"])

    # Drop rows with unparseable dates or missing champion/result -- these
    # can't be used for feature computation anyway.
    games = games.dropna(subset=["date", "champion", "result", "gameid"])

    # Keep only games with exactly 10 player rows (5v5, no partial data).
    game_sizes = games.groupby("gameid").size()
    complete_game_ids = game_sizes[game_sizes == EXPECTED_PLAYER_ROWS_PER_GAME].index
    clean_games = games[games["gameid"].isin(complete_game_ids)].reset_index(drop=True)

    # Bans: filter to complete games only, standardize champion casing.
    clean_bans = bans_df.copy() if bans_df is not None else pd.DataFrame(
        columns=["gameid", "team", "champion", "ban_number"]
    )
    if not clean_bans.empty:
        clean_bans = clean_bans[clean_bans["gameid"].isin(complete_game_ids)].copy()
        clean_bans["champion"] = _standardize_champion_name(clean_bans["champion"])
        clean_bans = clean_bans.reset_index(drop=True)

    return clean_games, clean_bans
