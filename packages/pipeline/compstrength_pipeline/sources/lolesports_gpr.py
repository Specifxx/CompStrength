"""Riot's official Global Power Rankings (GPR) as a team-strength source.

`lolesports.com/gpr/<year>/current <https://lolesports.com/en-GB/gpr/2026/current>`_
publishes Riot's own tier-1 team-strength rating, computed by a model that
sees things this pipeline cannot: in-game execution metrics (not just the
win/loss this repo's Elo pass consumes), an explicit regional-strength
score, and strength-of-schedule via average opponent rating. That makes it
a genuinely INDEPENDENT second opinion on team strength rather than a
re-derivation of the same signal, which is the only reason it's worth
adding next to ``teams.py``'s own Elo.

Why this is usable in a walk-forward backtest
---------------------------------------------
The page ships each team's full ``teamGPRHistory`` -- roughly one snapshot
every 10 days back to the season opener, each stamped with the
``dateCalculated`` the rating was computed on. So, exactly like
``soloqueue.py``'s timestamped history, a game played on date ``d`` can be
scored with the newest snapshot computed STRICTLY BEFORE ``d`` and nothing
leaks from the future. A "current standings" table with no history would be
useless here (every backtest fold would be reading tomorrow's answer).

How the data is obtained
------------------------
There is no documented public GPR API. The page is a React Server
Components app, and the rankings arrive inside the RSC flight payload as
plain JSON under a ``"teamGPR":[...]`` key -- so a single ordinary GET of
the page HTML yields the whole season's history, with no private endpoint,
no auth, and no per-request rate limit to worry about. ``fetch_gpr_season``
does that one request and brace-matches the JSON array out of the response.

ToS note: same good-citizen posture as the other sources in this package --
one request per season per daily refresh, no scraping loop.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from compstrength_pipeline.config import LOLESPORTS_GPR_URL_TEMPLATE


class GprUnavailableError(RuntimeError):
    """Raised when the GPR page can't be fetched or parsed."""


@dataclass(frozen=True)
class GprSnapshot:
    """One GPR rating for one team at one ``dateCalculated``."""

    team: str
    code: str
    league: str
    date: str
    gpr_score: float
    elo: float
    rank: int


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------


def _extract_json_array(html: str, key: str) -> list:
    """Brace-match the JSON array that follows ``"<key>":`` in ``html``.

    The RSC payload is one enormous line, so a regex can't reliably find the
    array's end; we scan forward tracking bracket depth (and string state, so
    brackets inside team names don't confuse the count). Returns the first
    array that parses; raises if none does.
    """
    for match in re.finditer(rf'"{re.escape(key)}"\s*:\s*\[', html):
        start = html.index("[", match.start())
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(html)):
            char = html[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(html[start : i + 1])
                    except json.JSONDecodeError:
                        break  # try the next occurrence of the key
    raise GprUnavailableError(f"No parseable {key!r} array found in the GPR page")


def parse_gpr_html(html: str) -> list[GprSnapshot]:
    """Parse one season's GPR page HTML into a flat list of snapshots."""
    records = _extract_json_array(html, "teamGPR")
    out: list[GprSnapshot] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        team = record.get("team") or {}
        name = (team.get("name") or "").strip()
        if not name:
            continue
        code = (team.get("code") or "").strip()
        league = ((team.get("homeLeague") or {}).get("name") or "").strip()
        history = record.get("teamGPRHistory") or []
        # ``currentTeamGPR`` is normally also the newest history entry, but
        # include it defensively -- dedup below keeps one per (team, date).
        current = record.get("currentTeamGPR")
        if isinstance(current, dict):
            history = [*history, current]
        for point in history:
            if not isinstance(point, dict):
                continue
            date = point.get("dateCalculated")
            score = point.get("gprScore")
            elo = point.get("elo")
            if not date or score is None or elo is None:
                continue
            out.append(
                GprSnapshot(
                    team=name,
                    code=code,
                    league=league,
                    date=str(date),
                    gpr_score=float(score),
                    elo=float(elo),
                    rank=int(point.get("rank") or 0),
                )
            )
    if not out:
        raise GprUnavailableError("GPR page parsed but contained no rating history")
    return out


def fetch_gpr_season(year: int, locale: str = "en-GB", timeout: int = 30) -> list[GprSnapshot]:
    """Fetch and parse one season's GPR history (one HTTP GET)."""
    url = LOLESPORTS_GPR_URL_TEMPLATE.format(locale=locale, year=year)
    try:
        import requests

        response = requests.get(
            url,
            timeout=timeout,
            # The page 403s bare programmatic clients; a normal UA string is
            # all it wants (this is public, unauthenticated page content).
            headers={"User-Agent": "Mozilla/5.0 (compatible; CompStrength/1.0)"},
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - any transport failure is "unavailable"
        raise GprUnavailableError(f"Could not fetch {url!r}: {exc!r}") from exc
    return parse_gpr_html(response.text)


# ---------------------------------------------------------------------------
# Timestamped history file (data/gpr_history.json)
# ---------------------------------------------------------------------------
#
# Shape, mirroring data/soloqueue_history.json:
#
#     {"snapshots": [{"ts": "<dateCalculated>",
#                     "teams": {"<GPR team name>": [gprScore, elo, rank]}}, ...],
#      "teamMeta": {"<GPR team name>": {"code": "T1", "league": "LCK"}}}
#
# One snapshot per distinct dateCalculated, sorted chronologically.


def build_history(snapshots: list[GprSnapshot]) -> dict:
    """Group flat snapshots into the ``data/gpr_history.json`` payload."""
    by_date: dict[str, dict[str, list[float]]] = {}
    meta: dict[str, dict[str, str]] = {}
    for snap in snapshots:
        by_date.setdefault(snap.date, {})[snap.team] = [
            snap.gpr_score,
            snap.elo,
            snap.rank,
        ]
        meta.setdefault(snap.team, {"code": snap.code, "league": snap.league})
    return {
        "snapshots": [
            {"ts": ts, "teams": by_date[ts]} for ts in sorted(by_date)
        ],
        "teamMeta": dict(sorted(meta.items())),
    }


def merge_history(existing: dict | None, fresh: dict) -> dict:
    """Union two history payloads, newer values winning per (date, team).

    Lets the daily refresh keep older seasons' snapshots after the live page
    has rolled over to a new season and stopped serving them.
    """
    merged: dict[str, dict[str, list[float]]] = {}
    meta: dict[str, dict[str, str]] = {}
    for payload in (existing or {"snapshots": [], "teamMeta": {}}, fresh):
        for snap in payload.get("snapshots", []):
            merged.setdefault(snap["ts"], {}).update(snap.get("teams", {}))
        meta.update(payload.get("teamMeta", {}))
    return {
        "snapshots": [{"ts": ts, "teams": merged[ts]} for ts in sorted(merged)],
        "teamMeta": dict(sorted(meta.items())),
    }


def load_gpr_history(path: str | Path) -> list[tuple[pd.Timestamp, dict[str, list[float]]]]:
    """Load the history file -> chronologically sorted
    ``[(ts, {team: [gprScore, elo, rank]}), ...]``. Empty when absent, so a
    missing file simply zeroes the feature rather than failing the build."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out = []
    for snap in payload.get("snapshots", []):
        try:
            ts = pd.Timestamp(snap["ts"])
        except (KeyError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        out.append((ts, snap.get("teams", {})))
    out.sort(key=lambda item: item[0])
    return out


# ---------------------------------------------------------------------------
# Name reconciliation: GPR names -> Oracle's Elixir team names
# ---------------------------------------------------------------------------
#
# The two sources name the same org differently. Most pairs fall out of
# aggressive normalization (case, accents, punctuation, "Esports"/"Gaming"
# suffixes); the rest are LPL teams that GPR prefixes with a host city
# ("Suzhou LNG Esports" vs OE's "LNG Esports"), plus a handful of genuine
# rebrands/sponsor names. Anything still unmatched just doesn't get a GPR
# rating, which zeroes the feature for that team -- never a wrong join.

# Words dropped entirely when normalizing a team name.
_FILLER_WORDS = (
    "esports",
    "esport",
    "gaming",
    "club",
    "team",
)

# Host-city / sponsor prefixes GPR prepends that OE doesn't use.
_STRIPPED_PREFIXES = (
    "beijing",
    "shenzhen",
    "suzhou",
    "xian",
    "hangzhou",
    "shanghai",
    "chengdu",
    "ninebot",
    "fukuoka",
    "tphcm",
    "the",
    "relove",
)

# Residual pairs normalization can't reach: sponsor suffixes GPR carries and
# OE doesn't (Kia, Alienware), and orgs the two sources spell differently
# enough that no rule connects them ("Beijing JDG Esports" vs "JD Gaming").
# GPR name -> Oracle's Elixir name.
GPR_TEAM_ALIASES: dict[str, str] = {
    "Beijing JDG Esports": "JD Gaming",
    "Cloud9 Kia": "Cloud9",
    "KaBuM!": "KaBuM! Ilha das Lendas",
    "NRG Kia": "NRG",
    "RED Kalunga": "RED Canids",
    "TP.HCM Team Flash": "Team Flash",
    "Team Liquid Alienware": "Team Liquid",
}


def normalize_team_name(name: str) -> str:
    """Aggressively normalize a team name for cross-source matching.

    Strips accents/case, drops org filler words ("Esports"/"Gaming"/"Club"/
    "Team") and GPR's host-city prefixes, so "Xi'an Team WE" and "Team WE"
    both reduce to ``we``. Apostrophes are deleted rather than treated as
    separators (otherwise "Xi'an" splits into two tokens and the city prefix
    stops matching), and filler words are stripped from the joined string as
    well as token-wise, so GPR's unspaced "WeiboGaming" still reaches OE's
    "Weibo Gaming".
    """
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = stripped.lower().replace("ø", "o").replace("æ", "ae").replace("ß", "ss")
    lowered = lowered.replace("'", "").replace("’", "")
    tokens = [t for t in re.split(r"[^a-z0-9]+", lowered) if t]
    while tokens and tokens[0] in _STRIPPED_PREFIXES:
        tokens = tokens[1:]
    kept = [t for t in tokens if t not in _FILLER_WORDS]
    # Never normalize a name down to nothing (e.g. an org literally called
    # "Gaming Team" would lose both tokens); fall back to the raw tokens.
    joined = "".join(kept or tokens)
    for filler in _FILLER_WORDS:
        if joined.endswith(filler) and len(joined) > len(filler):
            joined = joined[: -len(filler)]
            break
    return joined


def build_team_name_map(
    gpr_team_names, oe_team_names
) -> dict[str, str]:
    """``{GPR team name: Oracle's Elixir team name}`` for names present in both.

    Explicit aliases win; otherwise names are matched on
    :func:`normalize_team_name`. An OE name that several GPR names normalize
    onto is left to the first match (ordered), and unmatched GPR names are
    simply absent from the result.
    """
    oe_by_norm: dict[str, str] = {}
    for name in oe_team_names:
        if isinstance(name, str) and name.strip():
            oe_by_norm.setdefault(normalize_team_name(name), name.strip())

    mapping: dict[str, str] = {}
    for gpr_name in gpr_team_names:
        alias = GPR_TEAM_ALIASES.get(gpr_name)
        if alias is not None:
            resolved = oe_by_norm.get(normalize_team_name(alias))
            if resolved is not None:
                mapping[gpr_name] = resolved
                continue
        resolved = oe_by_norm.get(normalize_team_name(gpr_name))
        if resolved is not None:
            mapping[gpr_name] = resolved
    return mapping


def gpr_ratings_asof(
    history: list[tuple[pd.Timestamp, dict[str, list[float]]]],
    as_of,
    field: str = "gpr",
) -> dict[str, float]:
    """``{GPR team name: rating}`` from the newest snapshot computed STRICTLY
    BEFORE ``as_of`` -- the leakage rule that makes this usable as a
    backtest feature.

    ``field`` selects ``"gpr"`` (the published headline power score) or
    ``"elo"`` (GPR's underlying raw Elo). Empty dict when no snapshot
    predates ``as_of``.
    """
    index = 1 if field == "elo" else 0
    chosen: dict[str, list[float]] | None = None
    for ts, teams in history:
        if ts < as_of:
            chosen = teams
        else:
            break
    if not chosen:
        return {}
    ratings: dict[str, float] = {}
    for team, values in chosen.items():
        try:
            ratings[str(team)] = float(values[index])
        except (TypeError, ValueError, IndexError):
            continue
    return ratings


# ---------------------------------------------------------------------------
# CLI: refresh data/gpr_history.json (used by the scheduled data refresh)
# ---------------------------------------------------------------------------


def refresh_history_file(path: str | Path, years, locale: str = "en-GB") -> dict:
    """Fetch ``years`` and merge them into the history file at ``path``.

    Merging (rather than overwriting) is what keeps older seasons after the
    live page rolls over and stops serving them. Seasons that fail to fetch
    are skipped with a warning: a partial refresh is strictly better than
    losing the committed history.
    """
    import warnings

    file_path = Path(path)
    payload: dict | None = None
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

    fetched = 0
    for year in years:
        try:
            payload = merge_history(payload, build_history(fetch_gpr_season(year, locale)))
            fetched += 1
        except GprUnavailableError as exc:
            warnings.warn(f"GPR {year}: {exc}")
    if not fetched:
        raise GprUnavailableError(f"No GPR season could be fetched (tried {list(years)})")

    payload = payload or {"snapshots": [], "teamMeta": {}}
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    return payload


def main(argv: list[str] | None = None) -> None:
    import argparse
    from datetime import datetime, timezone

    # .../packages/pipeline/compstrength_pipeline/sources/lolesports_gpr.py
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(repo_root / "data" / "gpr_history.json")
    )
    parser.add_argument(
        "--years",
        default="",
        help=(
            "Comma-separated seasons to fetch (default: the current calendar "
            "year and the one before it -- enough to keep the training window "
            "covered without re-fetching frozen history every day)."
        ),
    )
    parser.add_argument("--locale", default="en-GB")
    args = parser.parse_args(argv)

    if args.years.strip():
        years = [int(y) for y in args.years.split(",") if y.strip()]
    else:
        this_year = datetime.now(timezone.utc).year
        years = [this_year - 1, this_year]

    payload = refresh_history_file(args.output, years, args.locale)
    latest = payload["snapshots"][-1]["ts"] if payload["snapshots"] else "n/a"
    print(
        f"Wrote {args.output}: {len(payload['snapshots'])} snapshots, "
        f"{len(payload['teamMeta'])} teams, latest {latest}"
    )


if __name__ == "__main__":
    main()
