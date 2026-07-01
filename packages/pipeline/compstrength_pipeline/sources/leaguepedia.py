"""Fetcher for Leaguepedia's Cargo query API (lol.fandom.com).

Leaguepedia (the Fandom-hosted LoL esports wiki) exposes structured match
data via the MediaWiki "Cargo" extension's query API:
``https://lol.fandom.com/api.php?action=cargoquery&...&format=json``.

This is used in CompStrength as a **cross-check / supplement** to Oracle's
Elixir, primarily for draft (bans/picks) and side data, since Leaguepedia's
Cargo tables are independently maintained and can catch data-entry
discrepancies.

Relevant Cargo tables (schema per ``Module:CargoDeclare`` pages on the
wiki):

- ``ScoreboardGames``: one row per game. Key fields include ``GameId``
  (primary join key, also used by ``ScoreboardPlayers`` and
  ``PicksAndBansS7``), ``MatchId`` (joins to ``MatchSchedule``),
  ``RiotPlatformGameId``, ``RiotGameId``, ``RiotVersion`` (patch), plus
  team names, side, winner, and date fields.
- ``ScoreboardPlayers``: one row per player-per-game. Key fields include
  ``GameId``, ``MatchId``, ``GameTeamId``, ``GameRoleId`` /
  ``GameRoleIdVs`` (role, keyed by in-game role rather than a positional
  index), ``UniqueRole``, champion played, side, and team link.
- ``PicksAndBansS7``: current-format draft table. Fields are
  ``Team1Ban1``..``Team1Ban5``, ``Team2Ban1``..``Team2Ban5``,
  ``Team1Pick1``..``Team1Pick5``, ``Team2Pick1``..``Team2Pick5``
  (``Team1`` = blue side, ``Team2`` = red side, numbered in real draft
  order), plus ``GameId``/``MatchId`` to join back to
  ``ScoreboardGames``/``MatchScheduleGame``.
- Tournament/date/patch metadata lives on ``Tournaments`` and
  ``MatchSchedule``, joined via ``OverviewPage``/``MatchId``.

This module is a **stub-but-correct** implementation: the emphasis is on
getting the Cargo query construction right (table names, join keys, field
names, ``where``/``join_on`` semantics) rather than on exhaustively
supporting every Cargo table. It is best-effort with respect to live
network calls -- lol.fandom.com is blocked from this sandbox's egress, so
this function is not exercised over the network here, but is written to
work unmodified in an unrestricted environment (e.g. GitHub Actions).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from compstrength_pipeline.config import LEAGUEPEDIA_API_BASE

DEFAULT_USER_AGENT = "CompStrength/0.1 (hobby esports analytics; contact via GitHub repo)"


class DataSourceUnavailableError(RuntimeError):
    """Raised when the Leaguepedia Cargo API cannot be reached."""


def build_cargoquery_params(
    tables: str,
    fields: str,
    where: str | None = None,
    join_on: str | None = None,
    order_by: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any]:
    """Build the query-string params for a ``action=cargoquery`` request.

    Args:
        tables: Comma-separated Cargo table names, e.g.
            ``"ScoreboardGames=SG, ScoreboardPlayers=SP"`` (aliases are
            supported and recommended when joining tables that share
            column names like ``GameId``).
        fields: Comma-separated fields to select, e.g.
            ``"SG.GameId=GameId, SG.Team1=Team1, SG.Team2=Team2"``.
        where: Optional SQL-like filter, e.g. ``"SG.RiotVersion='14.1'"``.
        join_on: Optional join condition when querying multiple tables,
            e.g. ``"SG.GameId=SP.GameId"``.
        order_by: Optional ORDER BY clause.
        limit: Max rows per page (Cargo caps this at 500).
        offset: Pagination offset.

    Returns:
        A dict of query parameters ready to pass as ``params=`` to
        ``requests.get``.
    """
    params: dict[str, Any] = {
        "action": "cargoquery",
        "tables": tables,
        "fields": fields,
        "format": "json",
        "limit": min(limit, 500),
        "offset": offset,
    }
    if where:
        params["where"] = where
    if join_on:
        params["join_on"] = join_on
    if order_by:
        params["order_by"] = order_by
    return params


def fetch_leaguepedia_cargo(
    tables: str,
    fields: str,
    where: str | None = None,
    join_on: str | None = None,
    order_by: str | None = None,
    limit: int = 500,
    offset: int = 0,
    api_base: str = LEAGUEPEDIA_API_BASE,
) -> pd.DataFrame:
    """Run a Cargo query against Leaguepedia and return the results as a DataFrame.

    This is a thin, best-effort wrapper: any network failure (blocked
    egress, timeout, non-200, malformed JSON) is converted into a
    ``DataSourceUnavailableError`` with a clear, actionable message rather
    than a raw ``requests`` traceback.

    Example (bans cross-check for a patch):
        >>> fetch_leaguepedia_cargo(  # doctest: +SKIP
        ...     tables="ScoreboardGames=SG",
        ...     fields="SG.GameId=GameId, SG.Team1=Team1, SG.Team2=Team2, "
        ...            "SG.WinTeam=WinTeam, SG.Patch=Patch",
        ...     where="SG.Patch='14.1'",
        ... )

    Args:
        tables: See :func:`build_cargoquery_params`.
        fields: See :func:`build_cargoquery_params`.
        where: See :func:`build_cargoquery_params`.
        join_on: See :func:`build_cargoquery_params`.
        order_by: See :func:`build_cargoquery_params`.
        limit: See :func:`build_cargoquery_params`.
        offset: See :func:`build_cargoquery_params`.
        api_base: Override for the API base URL (mainly for testing).

    Returns:
        A DataFrame of the ``cargoquery`` result rows (each row's "title"
        wrapper is flattened away).

    Raises:
        DataSourceUnavailableError: if the API cannot be reached or
            returns an error payload.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise DataSourceUnavailableError(
            "The 'requests' package is required to query Leaguepedia's Cargo "
            "API. Install it via `pip install requests`."
        ) from exc

    params = build_cargoquery_params(tables, fields, where, join_on, order_by, limit, offset)

    try:
        response = requests.get(
            api_base,
            params=params,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise DataSourceUnavailableError(
            f"Could not query Leaguepedia Cargo API at {api_base!r} "
            f"(tables={tables!r}). This is expected in network-restricted "
            "sandboxes (lol.fandom.com is blocked here). This code path is "
            "designed to work unmodified in an unrestricted environment such "
            f"as GitHub Actions. Original error: {exc!r}"
        ) from exc

    if "error" in payload:
        raise DataSourceUnavailableError(
            f"Leaguepedia Cargo API returned an error for tables={tables!r}: "
            f"{payload['error']}"
        )

    rows = [item.get("title", item) for item in payload.get("cargoquery", [])]
    return pd.DataFrame(rows)


def fetch_picks_and_bans(patch: str | None = None, limit: int = 500) -> pd.DataFrame:
    """Convenience wrapper: fetch ``PicksAndBansS7`` rows, optionally filtered by patch.

    Joins in ``ScoreboardGames`` to get the patch (``RiotVersion``) since
    ``PicksAndBansS7`` itself does not carry patch info directly.
    """
    where = f"SG.RiotVersion='{patch}'" if patch else None
    return fetch_leaguepedia_cargo(
        tables="PicksAndBansS7=PB, ScoreboardGames=SG",
        fields=(
            "PB.GameId=GameId, SG.RiotVersion=Patch, "
            "PB.Team1Ban1=Team1Ban1, PB.Team1Ban2=Team1Ban2, PB.Team1Ban3=Team1Ban3, "
            "PB.Team1Ban4=Team1Ban4, PB.Team1Ban5=Team1Ban5, "
            "PB.Team2Ban1=Team2Ban1, PB.Team2Ban2=Team2Ban2, PB.Team2Ban3=Team2Ban3, "
            "PB.Team2Ban4=Team2Ban4, PB.Team2Ban5=Team2Ban5, "
            "PB.Team1Pick1=Team1Pick1, PB.Team1Pick2=Team1Pick2, PB.Team1Pick3=Team1Pick3, "
            "PB.Team1Pick4=Team1Pick4, PB.Team1Pick5=Team1Pick5, "
            "PB.Team2Pick1=Team2Pick1, PB.Team2Pick2=Team2Pick2, PB.Team2Pick3=Team2Pick3, "
            "PB.Team2Pick4=Team2Pick4, PB.Team2Pick5=Team2Pick5"
        ),
        where=where,
        join_on="PB.GameId=SG.GameId",
        limit=limit,
    )
