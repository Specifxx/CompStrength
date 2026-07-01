"""Canonical League of Legends champion roster.

The champion *ratings* table used to only include champions that actually
appeared in the ingested match data (typically a few dozen, since a single
patch window of pro games only sees a fraction of the roster played). That
meant unconventional/off-meta picks simply weren't selectable on the site at
all, even though a real draft can legally lock in any champion for any role.

This module is the single source of truth for "every champion that exists",
independent of whether we have any pro-play data for them yet. ``features.py``
unions this roster with whatever champions actually appear in the games/
solo-queue data, so every champion always gets an entry in
``champion_ratings.json`` -- champions with zero pro games simply fall back
to their solo-queue-informed prior (see ``features.compute_champion_shrinkage``),
same as any other sparse-data champion.

Two ways to get the roster, tried in order:

1. **Live**: Riot's Data Dragon (``versions.json`` + ``champion.json``) is
   the authoritative, always-current list -- new champion releases show up
   here immediately with no code change needed. This requires outbound
   network access, which is unavailable in this development sandbox but
   works fine in an unrestricted environment (e.g. GitHub Actions).
2. **Fallback**: ``STATIC_CHAMPION_ROLES`` below, a hand-maintained snapshot
   used when the live fetch fails (offline dev, sandboxed CI, Data Dragon
   downtime) or as a floor that's unioned with whatever the live fetch
   returns. It will inevitably go stale as new champions release -- that's
   fine, since it's a fallback, not the primary path, and any champion
   missing from it entirely still works correctly if it shows up in the
   match/solo-queue data (see ``features.py``'s role fallback to "MID").

Primary role here is a best-effort classification of a champion's most
common competitive lane, used only as a sort-priority hint in the UI (e.g.
so a mid-lane search shows mid-lane champions first) -- it does NOT restrict
which role a champion can be selected for. Many champions are legitimately
flexible (jungle/mid, support/mid, etc.); where a champion splits roles we
picked the historically more common competitive assignment.
"""

from __future__ import annotations

import functools
import json
import urllib.request
from urllib.error import URLError

DATA_DRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DATA_DRAGON_CHAMPIONS_URL_TEMPLATE = (
    "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
)

_FETCH_TIMEOUT_SECONDS = 5

# Hand-maintained fallback snapshot, name -> most-common competitive role.
# Not exhaustive/permanent by design -- see module docstring.
STATIC_CHAMPION_ROLES: dict[str, str] = {
    # Top
    "Aatrox": "TOP", "Ambessa": "TOP", "Camille": "TOP", "Cho'Gath": "TOP",
    "Darius": "TOP", "Dr. Mundo": "TOP", "Fiora": "TOP", "Garen": "TOP",
    "Gnar": "TOP", "Gwen": "TOP", "Illaoi": "TOP", "Irelia": "TOP",
    "Jax": "TOP", "Jayce": "TOP", "K'Sante": "TOP", "Kennen": "TOP",
    "Kled": "TOP", "Malphite": "TOP", "Mordekaiser": "TOP", "Nasus": "TOP",
    "Olaf": "TOP", "Ornn": "TOP", "Pantheon": "TOP", "Poppy": "TOP",
    "Quinn": "TOP", "Renekton": "TOP", "Riven": "TOP", "Rumble": "TOP",
    "Sett": "TOP", "Shen": "TOP", "Singed": "TOP", "Sion": "TOP",
    "Teemo": "TOP", "Tryndamere": "TOP", "Urgot": "TOP", "Volibear": "TOP",
    "Yorick": "TOP", "Zaahen": "TOP",
    # Jungle
    "Amumu": "JUNGLE", "Bel'Veth": "JUNGLE", "Briar": "JUNGLE",
    "Elise": "JUNGLE", "Evelynn": "JUNGLE", "Fiddlesticks": "JUNGLE",
    "Graves": "JUNGLE", "Hecarim": "JUNGLE", "Ivern": "JUNGLE",
    "Jarvan Iv": "JUNGLE", "Jarvan IV": "JUNGLE", "Karthus": "JUNGLE",
    "Kayn": "JUNGLE", "Kha'Zix": "JUNGLE", "Kindred": "JUNGLE",
    "Lee Sin": "JUNGLE", "Lillia": "JUNGLE", "Master Yi": "JUNGLE",
    "Nidalee": "JUNGLE", "Nocturne": "JUNGLE", "Nunu & Willump": "JUNGLE",
    "Rammus": "JUNGLE", "Rek'Sai": "JUNGLE", "Rengar": "JUNGLE",
    "Sejuani": "JUNGLE", "Shaco": "JUNGLE", "Shyvana": "JUNGLE",
    "Skarner": "JUNGLE", "Udyr": "JUNGLE", "Vi": "JUNGLE", "Viego": "JUNGLE",
    "Warwick": "JUNGLE", "Wukong": "JUNGLE", "Xin Zhao": "JUNGLE",
    "Zac": "JUNGLE",
    # Mid
    "Ahri": "MID", "Akali": "MID", "Anivia": "MID", "Annie": "MID",
    "Aurelion Sol": "MID", "Aurora": "MID", "Azir": "MID", "Cassiopeia": "MID",
    "Corki": "MID", "Diana": "MID", "Ekko": "MID", "Fizz": "MID",
    "Galio": "MID", "Heimerdinger": "MID", "Hwei": "MID", "Kassadin": "MID",
    "Katarina": "MID", "Leblanc": "MID", "LeBlanc": "MID", "Lissandra": "MID",
    "Malzahar": "MID", "Mel": "MID", "Naafiri": "MID", "Neeko": "MID",
    "Orianna": "MID", "Qiyana": "MID", "Ryze": "MID", "Syndra": "MID",
    "Sylas": "MID", "Talon": "MID", "Taliyah": "MID", "Twisted Fate": "MID",
    "Veigar": "MID", "Vex": "MID", "Viktor": "MID", "Vladimir": "MID",
    "Yasuo": "MID", "Yone": "MID", "Zed": "MID", "Ziggs": "MID", "Zoe": "MID",
    # Bottom (ADC)
    "Aphelios": "BOTTOM", "Ashe": "BOTTOM", "Caitlyn": "BOTTOM",
    "Draven": "BOTTOM", "Ezreal": "BOTTOM", "Jhin": "BOTTOM",
    "Jinx": "BOTTOM", "Kai'Sa": "BOTTOM", "Kalista": "BOTTOM",
    "Kog'Maw": "BOTTOM", "Miss Fortune": "BOTTOM", "Nilah": "BOTTOM",
    "Samira": "BOTTOM", "Sivir": "BOTTOM", "Smolder": "BOTTOM",
    "Tristana": "BOTTOM", "Twitch": "BOTTOM", "Varus": "BOTTOM",
    "Vayne": "BOTTOM", "Xayah": "BOTTOM", "Yunara": "BOTTOM", "Zeri": "BOTTOM",
    # Support
    "Alistar": "SUPPORT", "Bard": "SUPPORT", "Blitzcrank": "SUPPORT",
    "Brand": "SUPPORT", "Braum": "SUPPORT", "Janna": "SUPPORT",
    "Karma": "SUPPORT", "Leona": "SUPPORT", "Lulu": "SUPPORT",
    "Maokai": "SUPPORT", "Milio": "SUPPORT", "Morgana": "SUPPORT",
    "Nami": "SUPPORT", "Nautilus": "SUPPORT", "Pyke": "SUPPORT",
    "Rakan": "SUPPORT", "Rell": "SUPPORT", "Renata Glasc": "SUPPORT",
    "Senna": "SUPPORT", "Seraphine": "SUPPORT", "Sona": "SUPPORT",
    "Soraka": "SUPPORT", "Swain": "SUPPORT", "Taric": "SUPPORT",
    "Thresh": "SUPPORT", "Vel'Koz": "SUPPORT", "Xerath": "SUPPORT",
    "Yuumi": "SUPPORT", "Zilean": "SUPPORT", "Zyra": "SUPPORT",
}


def _fetch_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


@functools.lru_cache(maxsize=1)
def fetch_live_champion_names() -> frozenset[str] | None:
    """Best-effort live fetch of every current champion name from Data Dragon.

    Returns ``None`` (never raises) if the network is unavailable or the
    request fails for any reason -- callers should fall back to
    :data:`STATIC_CHAMPION_ROLES` in that case. This is expected to fail in
    this development sandbox (network egress to ddragon is blocked here) and
    succeed in an unrestricted environment such as GitHub Actions.

    Cached for the lifetime of the process (``lru_cache``): the roster is
    effectively static within a single pipeline run, and this avoids one
    network round-trip (successful or failed) per backtest fold / per
    ``compute_champion_features`` call, which would otherwise add up fast.
    """
    try:
        versions = _fetch_json(DATA_DRAGON_VERSIONS_URL)
        latest = versions[0]
        champions_data = _fetch_json(
            DATA_DRAGON_CHAMPIONS_URL_TEMPLATE.format(version=latest)
        )
        return frozenset(entry["name"] for entry in champions_data["data"].values())
    except (URLError, TimeoutError, KeyError, IndexError, ValueError, OSError):
        return None


@functools.lru_cache(maxsize=1)
def get_full_champion_roster() -> dict[str, str]:
    """Return ``{champion_name: primaryRole}`` for every known champion.

    Combines the live Data Dragon roster (if reachable) with the static
    fallback list, so a champion released after this file was last updated
    still appears (with a "MID" role placeholder if it's not in
    :data:`STATIC_CHAMPION_ROLES`), and the site never loses champions that
    happen to be temporarily missing from a live fetch.
    """
    roster = dict(STATIC_CHAMPION_ROLES)
    live_names = fetch_live_champion_names()
    if live_names:
        for name in live_names:
            roster.setdefault(name, STATIC_CHAMPION_ROLES.get(name, "MID"))
    return roster
