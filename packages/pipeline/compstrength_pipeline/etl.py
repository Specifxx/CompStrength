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

import pandas as pd

EXPECTED_PLAYER_ROWS_PER_GAME = 10


def _standardize_champion_name(series: pd.Series) -> pd.Series:
    """Standardize champion name casing/whitespace.

    We title-case the name but special-case apostrophe-containing names
    (Kai'Sa, Kog'Maw, Cho'Gath, Vel'Koz, Kha'Zix, Rek'Sai) so the letter
    right after the apostrophe is also capitalized, matching Riot's
    official spelling.
    """
    cleaned = series.astype(str).str.strip()
    titled = cleaned.str.title()

    def _fix_apostrophe(name: str) -> str:
        if "'" in name:
            parts = name.split("'")
            parts = [parts[0]] + [p[:1].upper() + p[1:] if p else p for p in parts[1:]]
            return "'".join(parts)
        return name

    return titled.map(_fix_apostrophe)


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
