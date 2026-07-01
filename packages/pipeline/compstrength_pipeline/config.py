"""Configuration constants and hyperparameters for the CompStrength pipeline.

All tunable hyperparameters used by ``features.py`` and ``train_model.py`` live
here so they can be adjusted in one place and are easy to reason about /
unit-test against. Source URLs/endpoints are documented as constants but are
NOT called anywhere by default — callers must opt in explicitly (see
``sources/`` module docstrings for the ToS / rate-limit caveats).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Hyperparameters controlling the empirical-Bayes blending of pro and
    solo-queue win rates, and the recency decay applied to pro games.

    Attributes:
        patch_half_life_days: Half-life (in days) used for exponential decay
            of a pro game's weight based on how long ago it was played.
            After this many days, a game's contribution to the decayed
            sample size is halved.
        solo_queue_weight: Weight (0..1) given to the solo-queue win rate
            when constructing the informative prior mean. The remainder
            weight goes to ``GLOBAL_MEAN``.
        prior_games: The "pseudo-count" of games behind the prior mean.
            This is the shrinkage strength: a champion needs roughly this
            many decayed pro games before the raw pro win rate dominates
            the blended estimate.
        pro_window_days: The trailing window (in days) of pro games
            considered for a given patch's rating. If there are too few
            games in this window, the window is extended backward across
            patches to find more data (see ``features.py``).
        global_mean: The assumed baseline win rate for any champion in a
            balanced 5v5 game with no other information (50%).
        num_recent_patches: The number of most-recent distinct patches
            (ordered by the max game date on that patch, not lexicographic/
            semver order) to hard-restrict all feature computation to.
            Games on any older patch are dropped entirely before any
            window selection, decay, or pick/ban-rate computation happens.
            This enforces "only look at the last few patches at all" on
            top of (not instead of) the day-based exponential recency
            decay, per explicit product requirement.
        synergy_prior_games: Empirical-Bayes shrinkage strength (in
            decayed "pseudo-games") for the within-team pairwise synergy
            residual computed in ``pairwise.py``.
        matchup_prior_games: Empirical-Bayes shrinkage strength (in
            decayed "pseudo-games") for the cross-team same-role matchup
            residual computed in ``pairwise.py``.
    """

    patch_half_life_days: int = 21
    solo_queue_weight: float = 0.35
    prior_games: int = 15
    pro_window_days: int = 90
    global_mean: float = 0.5
    num_recent_patches: int = 3
    synergy_prior_games: int = 8
    matchup_prior_games: int = 10

    def __post_init__(self) -> None:
        if not (0.0 <= self.solo_queue_weight <= 1.0):
            raise ValueError("solo_queue_weight must be in [0, 1]")
        if self.patch_half_life_days <= 0:
            raise ValueError("patch_half_life_days must be positive")
        if self.prior_games < 0:
            raise ValueError("prior_games must be non-negative")
        if self.pro_window_days <= 0:
            raise ValueError("pro_window_days must be positive")
        if not (0.0 < self.global_mean < 1.0):
            raise ValueError("global_mean must be in (0, 1)")
        if self.num_recent_patches < 1:
            raise ValueError("num_recent_patches must be >= 1")
        if self.synergy_prior_games < 0:
            raise ValueError("synergy_prior_games must be non-negative")
        if self.matchup_prior_games < 0:
            raise ValueError("matchup_prior_games must be non-negative")


# Module-level default instance, importable directly as
# ``from compstrength_pipeline.config import DEFAULT_CONFIG``.
DEFAULT_CONFIG = PipelineConfig()

# Convenience module-level aliases matching the naming used in the project
# spec (some callers may prefer plain constants over the dataclass).
PATCH_HALF_LIFE_DAYS = DEFAULT_CONFIG.patch_half_life_days
SOLO_QUEUE_WEIGHT = DEFAULT_CONFIG.solo_queue_weight
PRIOR_GAMES = DEFAULT_CONFIG.prior_games
PRO_WINDOW_DAYS = DEFAULT_CONFIG.pro_window_days
GLOBAL_MEAN = DEFAULT_CONFIG.global_mean


# ---------------------------------------------------------------------------
# Data source endpoints (documented, NOT called automatically).
#
# These are provided so `sources/*.py` modules have a single source of truth
# for real endpoints/URLs. Network egress to these hosts is blocked in this
# development sandbox; they are intended to work unmodified in an
# unrestricted environment (e.g. GitHub Actions).
# ---------------------------------------------------------------------------

# Oracle's Elixir publishes one CSV per year, refreshed daily. The exact
# per-year download URL pattern is not officially documented via a stable
# REST API; the canonical entry point is the tools/downloads page which
# links to Google Sheets / CSV exports per year.
ORACLES_ELIXIR_DOWNLOADS_PAGE = "https://oracleselixir.com/tools/downloads"
# Best-effort direct CSV pattern (subject to change upstream); year must be
# supplied by the caller, e.g. ORACLES_ELIXIR_CSV_URL_TEMPLATE.format(year=2026)
ORACLES_ELIXIR_CSV_URL_TEMPLATE = (
    "https://oracleselixir-downloadable-match-data.s3.us-east-2.amazonaws.com/"
    "{year}_LoL_esports_match_data_from_OraclesElixir.csv"
)

# Leaguepedia Cargo query API (MediaWiki extension). Documented at
# https://lol.fandom.com/wiki/Special:CargoTables
LEAGUEPEDIA_API_BASE = "https://lol.fandom.com/api.php"
LEAGUEPEDIA_CARGO_TABLES = ("ScoreboardGames", "ScoreboardPlayers", "PicksAndBansS7")

# Solo-queue champion win rate source (best-effort, unofficial). See
# sources/soloqueue.py module docstring for the full ToS / stability caveat.
LOLALYTICS_TIERLIST_URL_TEMPLATE = (
    "https://a1.lolalytics.com/mega/?ep=champion&patch={patch}&tier={tier}&queue=ranked"
)

# Riot Data Dragon, used to fetch the live champion list (roles, ids) so we
# never have to hardcode a champion roster that goes stale when new
# champions release.
DATA_DRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DATA_DRAGON_CHAMPIONS_URL_TEMPLATE = (
    "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
)
