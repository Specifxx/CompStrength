"""Fetcher for Leaguepedia's Cargo query API (lol.fandom.com).

Leaguepedia (the Fandom-hosted LoL esports wiki) exposes structured match
data via the MediaWiki "Cargo" extension's query API:
``https://lol.fandom.com/api.php?action=cargoquery&...&format=json``.

Confirmed (live, against real 2026 data) field names and shapes:

- ``ScoreboardGames``: one row per game. ``GameId`` (primary join key),
  ``Team1``/``Team2`` (team names -- ``Team1`` is blue side, ``Team2`` is
  red side, per Leaguepedia's standard convention, consistent across this
  table and ``PicksAndBansS7``), ``WinTeam``, ``DateTime_UTC``, ``Patch``
  (e.g. ``"26.13"``), ``Tournament`` (a tournament *page name*, e.g.
  ``"2026 Mid-Season Invitational/Play-In Day 4"`` -- not a clean league
  acronym, see :func:`infer_league`).
- ``ScoreboardPlayers``: one row per player-per-game. ``GameId``, ``Link``
  (player name), ``Champion``, ``Side`` (``"1"`` = blue/Team1, ``"2"`` =
  red/Team2), ``Role`` (``"Top"``/``"Jungle"``/``"Mid"``/``"Bot"``/
  ``"Support"``), ``PlayerWin`` (``"Yes"``/``"No"``), ``Team``.
- ``PicksAndBansS7``: ``GameId``, ``Team1Ban1``..``Team1Ban5``,
  ``Team2Ban1``..``Team2Ban5``, ``Team1Pick1``..``Team1Pick5``,
  ``Team2Pick1``..``Team2Pick5`` (``Team1``/``Team2`` here are the same
  blue/red convention as ``ScoreboardGames``, NOT team names -- resolve
  the actual team name via a join on ``GameId`` against
  ``ScoreboardGames.Team1``/``Team2``).

This is used in CompStrength as the **primary live pro-match data source**
(``fetch_recent_games``) and doubles as a bans/side cross-check against
Oracle's Elixir when both are available. This module is not exercised over
the network in the dev sandbox (lol.fandom.com is blocked there), but is
written to work -- and has been validated live -- from an unrestricted
environment (GitHub Actions).

Rate limiting
--------------
The Cargo API rate-limits aggressively from shared-IP CI runners, and the
block can be sustained (observed to outlast 15+ minutes of in-job retries)
rather than a simple per-second throttle -- likely shared across whatever
else is hitting lol.fandom.com from the same GitHub Actions egress IP
range at the time, not purely a function of our own request rate.
``fetch_recent_games`` batches player/ban lookups into chunked
``GameId IN (...)`` queries (rather than one request per game) to keep the
total request count low, but when a sustained block is in effect, no
amount of in-job backoff will out-wait it -- ``_request_with_retry`` fails
fast (a few short retries, not tens of minutes) so a blocked run ends
quickly and the next attempt (the next day's scheduled run, or a
deliberately-later manual retry) gets a fresh chance instead.
"""

from __future__ import annotations

import random
import time
from typing import Any

import pandas as pd

from compstrength_pipeline.config import LEAGUEPEDIA_API_BASE

DEFAULT_USER_AGENT = "CompStrength/0.1 (hobby esports analytics; contact via GitHub repo)"

# Cargo's hard per-request row cap.
CARGO_MAX_LIMIT = 500

# Retry/backoff for the API's rate limiting. Confirmed (2026-07-01, via a
# targeted diagnostic) that this is a SUSTAINED, IP-wide block, not a
# query-complexity issue: a plain ScoreboardGames query with no IN-clause
# -- identical to one that succeeded minutes earlier -- got rate-limited
# too, after a burst of testing against this API from the same shared
# GitHub Actions egress IP range. In-job backoff cannot fix a block like
# this; only real wall-clock time between separate job runs can (the
# window appears to be well over 15 minutes, based on an 8-retry/~15-minute
# attempt that never recovered). So: keep retries here SHORT (fail in a
# couple of minutes, not tens of minutes) and rely on the scheduled job's
# once-a-day cadence -- plus genuinely spaced-out manual retries -- to find
# an open window, rather than hammering harder within a single run.
MAX_RETRIES = 3
RATE_LIMIT_BACKOFF_BASE_SECONDS = 20
RATE_LIMIT_BACKOFF_MAX_SECONDS = 45
REQUEST_DELAY_SECONDS = 12


def _backoff_seconds(attempt: int) -> float:
    """Capped, jittered backoff: grows with ``attempt`` but never past
    ``RATE_LIMIT_BACKOFF_MAX_SECONDS`` (bounds worst-case total wait), with
    random jitter so retries from many concurrent callers don't stay
    synchronized against a shared rate-limit window.
    """
    base = min(RATE_LIMIT_BACKOFF_BASE_SECONDS * (attempt + 1), RATE_LIMIT_BACKOFF_MAX_SECONDS)
    return base + random.uniform(0, 10)

ROLE_TO_POSITION = {
    "Top": "top",
    "Jungle": "jng",
    "Mid": "mid",
    "Bot": "bot",
    "Support": "sup",
}
SIDE_TO_LABEL = {"1": "Blue", "2": "Red"}
RESULT_TO_INT = {"Yes": 1, "No": 0}


class DataSourceUnavailableError(RuntimeError):
    """Raised when the Leaguepedia Cargo API cannot be reached."""


def infer_league(tournament: str) -> str:
    """Best-effort mapping from a Leaguepedia tournament *page name* (e.g.
    ``"2026 Mid-Season Invitational/Play-In Day 4"``) to a short league/
    event code. Only the international-event names need to be recognized
    precisely (they drive ``PipelineConfig.international_weight_multiplier``);
    anything else falls back to the leading path segment of the tournament
    name, which is good enough for display/pick-rate grouping purposes.
    """
    if not isinstance(tournament, str) or not tournament.strip():
        return "UNKNOWN"
    t = tournament.lower()
    if "mid-season invitational" in t:
        return "MSI"
    if "world championship" in t or t.strip().startswith("worlds"):
        return "WORLDS"
    for acronym in ("lck", "lpl", "lec", "lcs", "cblol", "pcs", "vcs", "lcp", "ewc"):
        if acronym in t:
            return acronym.upper()
    return tournament.split("/")[0].strip() or "UNKNOWN"


def build_cargoquery_params(
    tables: str,
    fields: str,
    where: str | None = None,
    join_on: str | None = None,
    order_by: str | None = None,
    limit: int = CARGO_MAX_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """Build the query-string params for a ``action=cargoquery`` request."""
    params: dict[str, Any] = {
        "action": "cargoquery",
        "tables": tables,
        "fields": fields,
        "format": "json",
        "limit": min(limit, CARGO_MAX_LIMIT),
        "offset": offset,
    }
    if where:
        params["where"] = where
    if join_on:
        params["join_on"] = join_on
    if order_by:
        params["order_by"] = order_by
    return params


def _request_with_retry(api_base: str, params: dict[str, Any]) -> dict:
    """GET ``api_base`` with ``params``, retrying on rate-limit responses.

    Returns the parsed JSON payload. Raises ``DataSourceUnavailableError``
    after exhausting retries or on any non-rate-limit error.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise DataSourceUnavailableError(
            "The 'requests' package is required to query Leaguepedia's Cargo "
            "API. Install it via `pip install requests`."
        ) from exc

    last_error: Exception | str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                api_base,
                params=params,
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - converted to DataSourceUnavailableError below
            last_error = exc
            time.sleep(_backoff_seconds(attempt))
            continue

        if isinstance(payload, dict) and payload.get("error", {}).get("code") == "ratelimited":
            last_error = "ratelimited"
            time.sleep(_backoff_seconds(attempt))
            continue

        if isinstance(payload, dict) and "error" in payload:
            raise DataSourceUnavailableError(
                f"Leaguepedia Cargo API returned an error: {payload['error']}"
            )

        return payload

    raise DataSourceUnavailableError(
        f"Leaguepedia Cargo API: exceeded {MAX_RETRIES} retries "
        f"(mostly likely persistent rate limiting). Last error: {last_error!r}. "
        "This is expected in network-restricted sandboxes (lol.fandom.com is "
        "blocked here); this code path is designed to work -- and has been "
        "validated -- from an unrestricted environment such as GitHub Actions, "
        "though it may still need to retry there too."
    )


def fetch_leaguepedia_cargo(
    tables: str,
    fields: str,
    where: str | None = None,
    join_on: str | None = None,
    order_by: str | None = None,
    limit: int = CARGO_MAX_LIMIT,
    offset: int = 0,
    api_base: str = LEAGUEPEDIA_API_BASE,
) -> pd.DataFrame:
    """Run a Cargo query against Leaguepedia and return the results as a DataFrame.

    See :func:`build_cargoquery_params` for argument semantics. Any network
    failure or API error (after retries) raises ``DataSourceUnavailableError``.
    """
    params = build_cargoquery_params(tables, fields, where, join_on, order_by, limit, offset)
    payload = _request_with_retry(api_base, params)
    rows = [item.get("title", item) for item in payload.get("cargoquery", [])]
    return pd.DataFrame(rows)


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _cargo_in_clause(values: list[str]) -> str:
    escaped = [v.replace('"', '\\"') for v in values]
    return ",".join(f'"{v}"' for v in escaped)


def fetch_recent_games(
    target_games: int = 1000,
    chunk_size: int = 50,
    api_base: str = LEAGUEPEDIA_API_BASE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the most recent ``target_games`` games from Leaguepedia, fully
    normalized into CompStrength's canonical schema.

    Returns:
        ``(games_df, bans_df)`` where ``games_df`` has columns
        ``[gameid, date, league, patch, side, team, player, position,
        champion, result]`` (one row per player-game, matching
        ``sources.oracles_elixir.CANONICAL_GAME_COLUMNS``) and ``bans_df``
        has ``[gameid, team, champion, ban_number]`` (matching
        ``CANONICAL_BAN_COLUMNS``).

    This issues a small, bounded number of requests regardless of
    ``target_games`` (paginated ``ScoreboardGames`` queries, plus chunked
    ``GameId IN (...)`` queries against ``ScoreboardPlayers`` and
    ``PicksAndBansS7``) to stay within the API's aggressive rate limiting.
    """
    game_pages: list[pd.DataFrame] = []
    offset = 0
    while sum(len(p) for p in game_pages) < target_games:
        remaining = target_games - sum(len(p) for p in game_pages)
        page = fetch_leaguepedia_cargo(
            tables="ScoreboardGames=SG",
            fields=(
                "SG.GameId=GameId, SG.Team1=Team1, SG.Team2=Team2, "
                "SG.WinTeam=WinTeam, SG.DateTime_UTC=DateTimeUTC, "
                "SG.Patch=Patch, SG.Tournament=Tournament"
            ),
            order_by="SG.DateTime_UTC DESC",
            limit=min(CARGO_MAX_LIMIT, remaining),
            offset=offset,
            api_base=api_base,
        )
        if page.empty:
            break
        game_pages.append(page)
        offset += len(page)
        time.sleep(REQUEST_DELAY_SECONDS)
        if len(page) < min(CARGO_MAX_LIMIT, remaining):
            break  # fewer rows than asked for -- exhausted the table

    if not game_pages:
        empty_games = pd.DataFrame(
            columns=["gameid", "date", "league", "patch", "side", "team", "player", "position", "champion", "result"]
        )
        empty_bans = pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])
        return empty_games, empty_bans

    games_meta = pd.concat(game_pages, ignore_index=True).drop_duplicates(subset="GameId")
    game_ids = games_meta["GameId"].tolist()

    player_pages: list[pd.DataFrame] = []
    ban_pages: list[pd.DataFrame] = []
    for chunk in _chunked(game_ids, chunk_size):
        in_clause = _cargo_in_clause(chunk)

        players = fetch_leaguepedia_cargo(
            tables="ScoreboardPlayers=SP",
            fields=(
                "SP.GameId=GameId, SP.Link=Player, SP.Champion=Champion, "
                "SP.Side=Side, SP.Role=Role, SP.PlayerWin=PlayerWin, SP.Team=Team"
            ),
            where=f"SP.GameId IN ({in_clause})",
            limit=CARGO_MAX_LIMIT,
            api_base=api_base,
        )
        if not players.empty:
            player_pages.append(players)
        time.sleep(REQUEST_DELAY_SECONDS)

        ban_fields = ", ".join(
            [f"PB.Team1Ban{n}=Team1Ban{n}" for n in range(1, 6)]
            + [f"PB.Team2Ban{n}=Team2Ban{n}" for n in range(1, 6)]
        )
        bans = fetch_leaguepedia_cargo(
            tables="PicksAndBansS7=PB",
            fields=f"PB.GameId=GameId, {ban_fields}",
            where=f"PB.GameId IN ({in_clause})",
            limit=CARGO_MAX_LIMIT,
            api_base=api_base,
        )
        if not bans.empty:
            ban_pages.append(bans)
        time.sleep(REQUEST_DELAY_SECONDS)

    games_df = _build_canonical_games(games_meta, player_pages)
    bans_df = _build_canonical_bans(games_meta, ban_pages)
    return games_df, bans_df


def _build_canonical_games(
    games_meta: pd.DataFrame, player_pages: list[pd.DataFrame]
) -> pd.DataFrame:
    if not player_pages:
        return pd.DataFrame(
            columns=["gameid", "date", "league", "patch", "side", "team", "player", "position", "champion", "result"]
        )

    players = pd.concat(player_pages, ignore_index=True)
    merged = players.merge(
        games_meta[["GameId", "DateTimeUTC", "Patch", "Tournament"]], on="GameId", how="left"
    )

    return pd.DataFrame(
        {
            "gameid": merged["GameId"],
            "date": pd.to_datetime(merged["DateTimeUTC"], utc=True, errors="coerce"),
            "league": merged["Tournament"].map(infer_league),
            "patch": merged["Patch"],
            "side": merged["Side"].map(SIDE_TO_LABEL),
            "team": merged["Team"],
            "player": merged["Player"],
            "position": merged["Role"].map(ROLE_TO_POSITION).fillna(
                merged["Role"].astype(str).str.lower()
            ),
            "champion": merged["Champion"].astype(str).str.strip(),
            "result": merged["PlayerWin"].map(RESULT_TO_INT),
        }
    )


def _build_canonical_bans(
    games_meta: pd.DataFrame, ban_pages: list[pd.DataFrame]
) -> pd.DataFrame:
    if not ban_pages:
        return pd.DataFrame(columns=["gameid", "team", "champion", "ban_number"])

    bans_wide = pd.concat(ban_pages, ignore_index=True).merge(
        games_meta[["GameId", "Team1", "Team2"]], on="GameId", how="left"
    )

    records = []
    for _, row in bans_wide.iterrows():
        for side_key, team_col in (("Team1", "Team1"), ("Team2", "Team2")):
            team_name = row.get(team_col)
            for n in range(1, 6):
                champ = row.get(f"{side_key}Ban{n}")
                if isinstance(champ, str) and champ.strip():
                    records.append(
                        {
                            "gameid": row["GameId"],
                            "team": team_name,
                            "champion": champ.strip(),
                            "ban_number": n,
                        }
                    )

    return pd.DataFrame(records, columns=["gameid", "team", "champion", "ban_number"])


def fetch_picks_and_bans(patch: str | None = None, limit: int = CARGO_MAX_LIMIT) -> pd.DataFrame:
    """Convenience wrapper: fetch raw ``PicksAndBansS7`` rows, optionally
    filtered by patch (joins in ``ScoreboardGames`` for the ``Patch`` field).
    """
    where = f"SG.Patch='{patch}'" if patch else None
    return fetch_leaguepedia_cargo(
        tables="PicksAndBansS7=PB, ScoreboardGames=SG",
        fields=(
            "PB.GameId=GameId, SG.Patch=Patch, "
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
