"""Solo-queue champion win rate sources.

CompStrength blends pro (competitive) win rates with solo-queue win rates
to build an informative empirical-Bayes prior (see ``features.py``). This
module defines a small abstract interface, ``SoloQueueSource``, plus:

- ``LolalyticsSoloQueueSource``: a best-effort concrete adapter for
  lolalytics.com, the source recommended by our research (see
  packages/pipeline/README.md for the full comparison of lolalytics vs
  u.gg vs op.gg vs the Riot API).
- ``StaticSoloQueueSource``: reads pre-computed winrate/games data from a
  local JSON fixture. This is what tests, demos, and offline/CI runs
  should use by default.

IMPORTANT ToS / stability caveat
---------------------------------
lolalytics.com does not publish an official public API. The endpoint used
by ``LolalyticsSoloQueueSource`` (see
``compstrength_pipeline.config.LOLALYTICS_TIERLIST_URL_TEMPLATE``) is a
reverse-engineered internal JSON endpoint that has been used by several
hobby projects (e.g. the ``lolalytics-api`` PyPI package, khorn89's
``LolAlytics.py`` on GitHub), but:

- It is not covered by a documented Terms of Service for third-party
  programmatic use; treat this adapter as best-effort and be prepared for
  it to break without notice if lolalytics changes their internal API.
- Respect reasonable rate limits (this adapter makes a single request per
  ``get_champion_winrates(patch)`` call; do not hammer the endpoint in a
  tight loop across many patches/ranks/regions).
- For a more defensible, ToS-compliant alternative, see the Riot Developer
  API path documented in the project README: pull match-v5 data at scale
  yourself and aggregate win rates. This is more "legitimate" but requires
  building your own crawler/aggregation pipeline and is much heavier
  (rate limits, storage, compute).
- OP.GG's official MCP server (https://github.com/opgginc/opgg-mcp) is
  another officially-sanctioned option worth evaluating as a secondary/
  fallback adapter, though it's an MCP interface rather than a plain
  REST/JSON endpoint, so it is not implemented here.

This module never calls the network at import time, and the concrete
lolalytics adapter is not exercised over the network in this sandbox
(lolalytics.com is blocked here) -- it is written to work unmodified in an
unrestricted environment such as GitHub Actions.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from compstrength_pipeline.config import LOLALYTICS_TIERLIST_URL_TEMPLATE

# type alias: champion name -> (winrate, games)
ChampionWinrates = dict[str, tuple[float, int]]


class DataSourceUnavailableError(RuntimeError):
    """Raised when a live solo-queue data source cannot be reached."""


class SoloQueueSource(ABC):
    """Abstract interface for a solo-queue champion win rate provider."""

    @abstractmethod
    def get_champion_winrates(self, patch: str) -> ChampionWinrates:
        """Return ``{champion_name: (winrate, games)}`` for the given patch.

        Implementations should return an empty dict (rather than raising)
        when they simply have no data for a champion/patch combination;
        callers (``features.py``) are responsible for falling back to
        ``GLOBAL_MEAN`` in that case. Raising is reserved for genuine
        failures to reach/parse the source at all.
        """
        raise NotImplementedError


class LolalyticsSoloQueueSource(SoloQueueSource):
    """Best-effort adapter for lolalytics.com's unofficial tier-list JSON endpoint.

    See the module docstring for the ToS / stability caveat. This adapter
    performs exactly one HTTP GET per ``get_champion_winrates`` call.
    """

    def __init__(self, tier: str = "platinum_plus", region: str = "all"):
        """
        Args:
            tier: Rank tier filter as accepted by lolalytics (e.g.
                "platinum_plus", "diamond_plus", "master_plus").
            region: Region filter (lolalytics uses "all" for global stats).
        """
        self.tier = tier
        self.region = region

    def get_champion_winrates(self, patch: str) -> ChampionWinrates:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise DataSourceUnavailableError(
                "The 'requests' package is required to fetch live lolalytics "
                "data. Install it via `pip install requests`."
            ) from exc

        url = LOLALYTICS_TIERLIST_URL_TEMPLATE.format(patch=patch, tier=self.tier)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise DataSourceUnavailableError(
                f"Could not fetch solo-queue win rates from lolalytics for "
                f"patch={patch!r} (url={url!r}). This is expected in "
                "network-restricted sandboxes (lolalytics.com is blocked "
                "here). This code path is designed to work unmodified in an "
                "unrestricted environment such as GitHub Actions. For "
                "offline/test use, use StaticSoloQueueSource instead. "
                f"Original error: {exc!r}"
            ) from exc

        return self._parse_payload(payload)

    @staticmethod
    def _parse_payload(payload: dict) -> ChampionWinrates:
        """Parse lolalytics' tier-list payload into ``{champion: (winrate, games)}``.

        The exact shape of lolalytics' internal JSON is not officially
        documented and has changed over time; this parses the commonly
        observed shape (a list of per-champion records under a "champions"
        or top-level list key, each with name/winrate/count-ish fields)
        defensively, skipping anything it can't confidently parse rather
        than raising.
        """
        records = payload.get("champions") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return {}

        result: ChampionWinrates = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            name = rec.get("name") or rec.get("champion")
            winrate = rec.get("winrate") or rec.get("wins")
            games = rec.get("count") or rec.get("games") or rec.get("n")
            if name is None or winrate is None or games is None:
                continue
            try:
                winrate_f = float(winrate)
                # lolalytics reports winrate as a percentage (e.g. 51.2).
                if winrate_f > 1.0:
                    winrate_f /= 100.0
                games_i = int(games)
            except (TypeError, ValueError):
                continue
            result[str(name)] = (winrate_f, games_i)
        return result


class NullSoloQueueSource(SoloQueueSource):
    """Always returns no data (every champion falls back to ``GLOBAL_MEAN``).

    Used when running against real pro-match data without a working live
    solo-queue adapter wired up yet: rather than blending real pro results
    with the *synthetic* fixture's made-up solo-queue numbers, this keeps
    the prior an honest, data-free neutral baseline (50%) until a real
    solo-queue source (e.g. ``LolalyticsSoloQueueSource``, once its
    endpoint/params are confirmed working) is available.
    """

    def get_champion_winrates(self, patch: str) -> ChampionWinrates:
        return {}


class StaticSoloQueueSource(SoloQueueSource):
    """Reads solo-queue win rates from a local JSON fixture.

    Intended for tests, demos, and any offline/CI run where live network
    access to a solo-queue stats provider is unavailable or undesired.

    Expected JSON fixture shape::

        {
          "<patch>": {
            "<ChampionName>": {"winrate": 0.512, "games": 18342},
            ...
          },
          ...
        }
    """

    def __init__(self, fixture_path: str | Path):
        self.fixture_path = Path(fixture_path)
        if not self.fixture_path.exists():
            raise FileNotFoundError(f"Solo queue fixture not found at {self.fixture_path}")
        with self.fixture_path.open("r", encoding="utf-8") as f:
            self._data: dict[str, dict[str, dict]] = json.load(f)

    def get_champion_winrates(self, patch: str) -> ChampionWinrates:
        patch_data = self._data.get(patch, {})
        result: ChampionWinrates = {}
        for champion, stats in patch_data.items():
            winrate = stats.get("winrate")
            games = stats.get("games")
            if winrate is None or games is None:
                continue
            result[champion] = (float(winrate), int(games))
        return result

    def available_patches(self) -> list[str]:
        """Return the list of patches present in the fixture."""
        return list(self._data.keys())


# ---------------------------------------------------------------------------
# Timestamped solo-queue history (data/soloqueue_history.json)
# ---------------------------------------------------------------------------
#
# Built once by .github/workflows/build-soloqueue-history.yml from the git
# history of a public repo that archives lolalytics-derived per-champion
# solo-queue stats ~twice a day. Each snapshot carries the COMMIT TIMESTAMP
# at which the data verifiably existed, so the walk-forward backtest can do
# a leakage-free "as of" join: a fold may only read snapshots committed
# strictly before its boundary. Snapshot champion entries are
# [wins, matches, previousWins, previousMatches] (current patch-to-date at
# the commit + the full prior patch), so (wins+previousWins) /
# (matches+previousMatches) is a large, purely-backward-looking sample.


def load_solo_history(path: str | Path) -> list[tuple[pd.Timestamp, dict]]:
    """Load the history fixture -> chronologically sorted
    ``[(commit_ts, {champion: [w, m, pw, pm]}), ...]``. Empty list if the
    file is absent (callers then keep the flat prior)."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        fixture = json.load(f)
    out = []
    for snap in fixture.get("snapshots", []):
        try:
            ts = pd.Timestamp(snap["ts"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
        except (KeyError, ValueError):
            continue
        out.append((ts, snap.get("champions", {})))
    out.sort(key=lambda x: x[0])
    return out


def solo_winrates_asof(
    history: list[tuple[pd.Timestamp, dict]], as_of, patch_offset: int | None = None
) -> ChampionWinrates:
    """``{champion: (winrate, games)}`` from the LATEST snapshot committed at
    or before ``as_of`` (leakage rule: never read data from the future).
    Empty dict when no snapshot predates ``as_of``.

    Each snapshot stores ``[wins, matches, previousWins, previousMatches]``
    per champion -- the live patch's numbers AND the patch before it.
    ``patch_offset`` selects which:

    * ``None`` (default): pool both patches. Biggest sample, but it smears
      two different metas together.
    * ``0``: the live solo-queue patch only.
    * ``1``: the PREVIOUS solo-queue patch only. Solo queue runs ahead of pro
      play -- leagues lock a patch for a week or a whole split, so when a new
      patch ships, solo queue moves immediately and pro does not. Measured on
      2026 data, the live solo patch is one ahead of the patch pro is
      actually playing in 18 of 63 snapshots (29%) and level the rest of the
      time. Offset 1 is therefore the right alignment exactly when it
      matters (just after a patch drops) and one patch stale otherwise --
      which is why the choice is a measured knob rather than an assumption.
    """
    chosen = None
    for ts, champs in history:
        if ts <= as_of:
            chosen = champs
        else:
            break
    if not chosen:
        return {}
    result: ChampionWinrates = {}
    for name, vals in chosen.items():
        try:
            w, m, pw, pm = (int(v) for v in vals[:4])
        except (TypeError, ValueError):
            continue
        if patch_offset == 0:
            wins, games = w, m
        elif patch_offset == 1:
            wins, games = pw, pm
        else:
            wins, games = w + pw, m + pm
        if games <= 0:
            continue
        result[str(name)] = (wins / games, games)
    return result
