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
        target_training_games: The number of most-recent games (by date,
            regardless of which patch they're on) to hard-restrict all
            feature computation to -- i.e. a target *sample size*, not a
            patch-count cutoff. Older games are still included (unlike the
            previous patch-count-based cutoff) as long as there are fewer
            than this many more-recent games available; they just count
            for less thanks to the day-based exponential recency decay
            below, rather than being excluded outright. If fewer than this
            many games exist in total, all of them are used. This is what
            lets the model train on a statistically meaningful sample
            (default target: 1000 games) even when the last patch or two
            alone wouldn't have nearly enough.
        synergy_prior_games: Empirical-Bayes shrinkage strength (in
            decayed "pseudo-games") for the within-team pairwise synergy
            residual computed in ``pairwise.py``.
        matchup_prior_games: Empirical-Bayes shrinkage strength (in
            decayed "pseudo-games") for the cross-team same-role matchup
            residual computed in ``pairwise.py``.
        international_leagues: ``league`` column values (matched
            case-insensitively) treated as top-level international events
            (MSI, Worlds, EWC, ...) rather than regular regional season
            play. These pit the best teams from every region against each
            other on a single current patch, which makes them an
            unusually concentrated, high-signal sample of how the current
            meta actually resolves at the highest level -- worth weighting
            up relative to an average regional-split game.
        international_weight_multiplier: Extra multiplier (on top of the
            normal patch-recency decay weight) applied to games whose
            ``league`` is in ``international_leagues``. ``1.0`` disables
            this entirely.
        patch_decay_base: Per-patch geometric decay base applied on top of
            the day-based recency decay. Each game is additionally weighted
            by ``patch_decay_base ** patch_ordinal_distance``, where the
            newest patch has distance 0, the previous patch distance 1, and
            so on (patches ordered by their most recent game date, NOT by
            lexically sorting the patch string). With the default 0.5 the
            current patch counts at full weight, the previous patch at half,
            two patches back at a quarter -- so the latest patch(es)
            dominate the rating, while older patches still contribute a
            shrinking-but-nonzero amount. ``1.0`` disables patch weighting
            entirely (pure calendar-day decay, the old behavior). This is
            the knob for "weight the latest patches more heavily".
    """

    patch_half_life_days: int = 21
    solo_queue_weight: float = 0.35
    prior_games: int = 15
    pro_window_days: int = 90
    global_mean: float = 0.5
    target_training_games: int = 1000
    synergy_prior_games: int = 30
    matchup_prior_games: int = 35
    international_leagues: frozenset[str] = frozenset(
        {"MSI", "WLDS", "WORLDS", "EWC"}
    )
    international_weight_multiplier: float = 1.5
    patch_decay_base: float = 0.5

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
        if self.target_training_games < 1:
            raise ValueError("target_training_games must be >= 1")
        if self.synergy_prior_games < 0:
            raise ValueError("synergy_prior_games must be non-negative")
        if self.matchup_prior_games < 0:
            raise ValueError("matchup_prior_games must be non-negative")
        if self.international_weight_multiplier <= 0:
            raise ValueError("international_weight_multiplier must be positive")
        if not (0.0 < self.patch_decay_base <= 1.0):
            raise ValueError("patch_decay_base must be in (0, 1]")


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

# Oracle's Elixir publishes one CSV per season (year), refreshed ~daily.
# It MIGRATED OFF its old AWS S3 bucket (which now returns NoSuchBucket) to
# Google Drive: the tools/downloads page links a public Drive folder with one
# file per year, named "<YEAR>_LoL_esports_match_data_from_OraclesElixir.csv".
# The stable machine handle is the per-year Drive file ID; fetch it with
# gdown (which handles Drive's >25MB virus-scan confirm-token automatically),
# or the drive.usercontent.google.com direct-download endpoint as a fallback.
# These IDs are corroborated across many community repos (2026 verified across
# 6 independent 2026 projects). oracleselixir.com itself is Cloudflare/bot
# -blocked, but the Drive endpoints are directly reachable -- which is exactly
# why every community pipeline hits Drive rather than scraping the site.
ORACLES_ELIXIR_DOWNLOADS_PAGE = "https://oracleselixir.com/tools/downloads"
ORACLES_ELIXIR_DRIVE_IDS: dict[int, str] = {
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    2023: "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    2022: "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy",
}
# Direct-download endpoint for a large public Drive file without a browser
# (the ``confirm=t`` skips the >25MB virus-scan interstitial). Used as the
# fallback path when the ``gdown`` package isn't available.
ORACLES_ELIXIR_DRIVE_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
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
