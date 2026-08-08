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
    # Safety cap on how many most-recent games to train on. With min_patch
    # defining the real window (25.1 -> current), this is set high enough not
    # to trim that window (~1.5 seasons of all-league pro play), so it only
    # guards against pathological inputs. A smoke test can still pass a small
    # --target-games to fetch/process fewer games quickly.
    target_training_games: int = 20000
    # Oldest patch to include, inclusive, in Oracle's Elixir's `patch` column
    # numbering (the GAME-CLIENT version, not Riot's season.patch marketing
    # name). OE labels the 2026 season "16.xx" and 2025 "15.xx" (client major
    # = year - 2010), so the owner's "no older than 2025 patch 1 (25.1)" is
    # "15.1" here, and the current "26.13" is "16.13". Games on older patches
    # are dropped entirely; the newest patch present is weighted most and each
    # older patch exponentially less (patch_decay_base). Compared numerically,
    # not lexically. None disables the floor; and if the floor would drop every
    # game (e.g. the synthetic fixture's 14.x patches) it is skipped so
    # offline/dev runs still work.
    min_patch: str | None = "15.1"
    synergy_prior_games: int = 30
    matchup_prior_games: int = 35
    international_leagues: frozenset[str] = frozenset(
        {"MSI", "WLDS", "WORLDS", "EWC"}
    )
    international_weight_multiplier: float = 1.5
    # Tier-1 regional leagues (Oracle's Elixir ``league`` codes). Games from
    # these leagues get ``major_league_weight_multiplier`` extra weight in all
    # decayed statistics: the best teams and most disciplined drafting produce
    # a cleaner champion-strength signal than minor/academy leagues, which
    # make up the majority of raw game volume (the 2026 dataset spans 36
    # leagues, mostly sub-tier-1). Multiplier 1.0 disables this entirely.
    major_leagues: frozenset[str] = frozenset(
        {"LCK", "LPL", "LEC", "LCS", "LTA", "LTA N", "LTA S", "LCP"}
    )
    major_league_weight_multiplier: float = 1.0
    patch_decay_base: float = 0.5
    # Feed the meta-presence feature (per-champion pickRate + banRate over the
    # training window) into the logistic model as a 4th predictor. The theory
    # (teams reveal champion strength through bans; presence needs no game
    # outcomes so it's low-noise) did NOT survive measurement: a walk-forward
    # A/B on 4,112 held-out real 2026 pro games showed presence made held-out
    # accuracy/log-loss slightly WORSE at every regularization strength tried
    # (off: acc .5319 / ll .6907; on: .5233/.6912 at C=.001, monotonically
    # worse at lighter C). Likely because in pro play both teams draft from
    # the same contested pool, so the presence differential is small and
    # uninformative. Kept as machinery (False zeroes the feature so its
    # fitted weight is exactly 0) in case future data changes the picture.
    use_presence_feature: bool = False
    # Team-strength feature: sequential pre-game Elo over the game history
    # (see teams.py), consumed as (blue_elo - red_elo) / 400. This is the one
    # feature class the literature consistently shows lifts pro-match
    # prediction well above draft-only (~60%+ vs ~53-55%): team identity
    # carries most of the predictable signal. Unlike synergy/matchup it is
    # NOT in-sample-leaky (a game's feature only uses strictly-earlier
    # games), so it tolerates lighter regularization. False disables it
    # (feature zeroed, weight exactly 0), reverting to draft-only. On the
    # site, teams are OPTIONAL inputs: leaving them blank feeds a 0 gap
    # ("equal teams"), which reproduces the draft-only prediction.
    use_team_feature: bool = True
    # Premier leagues (owner directive): the strongest leagues' games should
    # dominate the champion/synergy/matchup statistics. When
    # premier_league_target_share is set, a per-run multiplier is computed so
    # these leagues' combined share of total decayed training weight hits the
    # target (e.g. 0.7 = LCK+LPL carry 70% of the weight), regardless of how
    # many raw games each league contributes that window. None disables.
    # Note this drives the same weighting machinery as major_leagues (which
    # it overrides when active); international events keep their own boost.
    # MEASURED COST (walk-forward, 4,451 held-out 2026 games, teams known):
    # premier ON = 63.9% acc / 0.639 log-loss vs premier OFF = 64.1% / 0.632;
    # draft-only mode pays more (52.5%/0.701 vs 53.1%/0.690). Kept ON per the
    # owner's explicit directive; set premier_league_target_share=None to
    # revert to the empirically-best uniform league weighting.
    premier_leagues: frozenset[str] = frozenset({"LCK", "LPL"})
    premier_league_target_share: float | None = 0.7
    # Champion-strength definition (the draft-only score_diff feature). An
    # agent-fleet experiment sweep on the walk-forward harness found that a
    # simple SHRUNK DECAYED WIN RATE -- (weighted wins + shrink/2) /
    # (weighted games + shrink) - 0.5, with a LONG calendar half-life and NO
    # patch decay -- beats both the old EB-logit strength and one-hot
    # champion regression: draft-only held-out accuracy 55.4% vs 52.5%
    # (log-loss 0.684 vs 0.701) with two seasons of history. Training rows
    # use LEAVE-ONE-GAME-OUT winrates (a game's own result is excluded from
    # its feature) so the joint model's calibration is honest.
    champion_wr_half_life_days: float = 150.0
    champion_wr_shrink_games: float = 25.0
    # Multiplier turning (winrate - 0.5) into strengthScore. Like
    # elo_feature_scale, a per-feature regularization knob: big enough that
    # the non-leaky score_diff survives the heavy global C that keeps the
    # leaky synergy/matchup features crushed. Swept 10/25/40 on the
    # two-season production backtest; 40 gave the best draft-only accuracy
    # with identical calibration.
    strength_feature_scale: float = 40.0
    # Solo-queue-informed prior for the champion-strength signal: instead of
    # shrinking a champion's pro winrate toward a FLAT 0.5, shrink toward
    # 0.5 + wr_prior_solo_weight * (solo_wr - solo_mean), clipped to
    # +-wr_prior_clip. Centering on the games-weighted solo mean cancels any
    # rank-tier bias in the source; the clip bounds how far solo queue alone
    # can move a champion with no pro games. Only champions with at least
    # wr_prior_min_solo_games solo games get a non-flat prior.
    #
    # MEASURED (K=8 walk-forward, 15,973 games, timestamped leakage-free solo
    # history data/soloqueue_history.json): weights 0.5/1.0/1.5 all land
    # WITHIN NOISE of the flat prior -- current-season draft-only 55.40-55.50%
    # vs 55.57% flat, log-loss 0.6836 vs 0.6837, with-teams unchanged. Reason:
    # with two seasons of pro data and 25-pseudo-game shrinkage, nearly every
    # champion that actually gets DRAFTED in pro has ample pro evidence, so
    # the prior only binds for champions who almost never appear in the test
    # set. Shipped OFF (0.0 = exact flat-prior behavior). The machinery,
    # fixture, and daily snapshot append stay: a BRAND-NEW champion release
    # (zero pro games, plentiful solo games) is the one real beneficiary, and
    # flipping this weight is a one-line change if that case matters later.
    wr_prior_solo_weight: float = 0.0
    wr_prior_min_solo_games: int = 2000
    wr_prior_clip: float = 0.06
    # Player-level features (players.py), an OPTIONAL ADD-ON to the team
    # feature: per-player Elo (mean of the 5 starters, tracks WHO is actually
    # playing across roster moves) and player-champion proficiency (shrunk
    # pre-game winrate edge on the champion each player is on -- the comfort
    # -pick signal). Both are gated behind the same optional team selection
    # as team Elo: no teams selected -> features are 0 and the prediction is
    # identical to draft-only. MEASURED (8-fold walk-forward, 5,935 held-out
    # 2026 games): team Elo only 63.6%/0.633 -> +player Elo 64.7%/0.628 ->
    # +proficiency 65.0%/0.628. Requires use_team_feature (rosters are
    # resolved through the selected teams).
    use_player_features: bool = True
    # Per-player Elo K. Smaller than the team K=32 was the a-priori guess
    # (5 players share one game outcome), but the sweep said otherwise:
    # 16 -> 64.45%, 24 -> 64.99%, 32 -> 65.17%, 40 -> 65.11%, 48 -> 64.92%
    # held-out at carryover 0.8; 32 is the peak.
    player_elo_k: float = 32.0
    # Season-boundary carryover for player Elo. Players keep more of their
    # rating across seasons than teams (0.8 vs the teams' 0.7): a roster is
    # rebuilt, a person is not. 0.8 beat 0.7 (64.84%) and edged 0.9 (65.07%)
    # in the sweep at K=24.
    player_season_carryover: float = 0.8
    # Pseudo-games shrinking a player-champion winrate toward the player's
    # OWN overall winrate (so proficiency isolates champion-specific skill
    # from general skill, which player Elo already carries). 4/8/16 all
    # within noise (64.94-65.02% at K=24); 8 ships.
    prof_shrink_games: float = 8.0
    # Elo K-factor (rating movement per game) for the team feature. 32 beat
    # 16/24/48 on the walk-forward backtest at the shipped feature scale for
    # the ORIGINAL binary Elo; with margin-of-victory scoring enabled (see
    # elo_mov_tau) K and tau were re-swept jointly and 64 won -- continuous
    # scores sit closer to the expected value, so each update is a smaller
    # step and a larger K restores the same effective learning pace.
    elo_k: float = 64.0
    # Margin-of-victory scoring for team Elo. None = classic binary Elo (a
    # win is a win). When set, a game's observed score becomes a continuous
    # dominance value derived from the blue-minus-red GOLD PER MINUTE
    # differential, squashed through a logistic with this tau: a lopsided
    # 25-minute win teaches the ratings far more than a 45-minute coinflip.
    # Measured on the walk-forward backtest (15,060 held-out games):
    # current-season accuracy 65.73% -> 66.42%, log-loss 0.6218 -> 0.6125,
    # with log-loss improving in 7/7 folds. Hyperparameters were also chosen
    # on EARLY folds only and re-checked on later folds they never saw
    # (65.96%/0.6193 -> 66.56%/0.6128), so the gain is not grid-selection
    # luck. The effect is broad, not a knife-edge: every tau in [400, 1600]
    # crossed with every K in [32, 96] beat the binary baseline on both
    # metrics.
    elo_mov_tau: float | None = 800.0
    # Fraction of a team's Elo deviation kept across a season boundary
    # (rosters change). Measured: 0.7 beats 1.0/0.85/0.5/0.25 on 2026
    # pre-game Elo predictions (64.11%/0.6326 vs 63.86%/0.6357 at 1.0).
    elo_season_carryover: float = 0.7
    # Extra Elo K multiplier for INTER-REGION games -- a game is inter-region
    # when its league is an international event (see international_leagues) OR
    # the two teams' most-recently-seen leagues differ. These are the ONLY
    # games that calibrate Elo across regions (a domestic split never tells
    # you how LCK stacks up against LPL), so they carry more information and
    # should move ratings more. MEASURED (walk-forward, 2025+2026): boosting
    # this lifts held-out accuracy on international events (MSI/Worlds/EWC)
    # from 55.7% toward ~58% AND nudges overall accuracy/log-loss the right
    # way, because better cross-region anchoring helps every prediction
    # involving a team whose strength was set partly by cross-region play.
    # 1.0 disables (uniform K, the old behaviour).
    international_elo_k_multiplier: float = 3.0
    # Divisor turning an Elo gap into the model feature. A regularization
    # knob, not an Elo parameter: under L2, a bigger feature (smaller
    # divisor) is penalized effectively less, letting the NON-leaky Elo
    # feature carry real weight while the same heavy C keeps crushing the
    # leaky synergy/matchup features. Swept 400 -> 200 -> 100 -> 50 -> 25 ->
    # 15 on the real backtest: held-out log-loss improves monotonically to
    # ~15-25 then flattens; 15 ships. Shipped in teams.json as params.eloScale
    # so the frontend computes the identical feature.
    elo_feature_scale: float = 15.0

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
        if self.major_league_weight_multiplier <= 0:
            raise ValueError("major_league_weight_multiplier must be positive")
        if self.elo_k <= 0:
            raise ValueError("elo_k must be positive")
        if self.elo_mov_tau is not None and self.elo_mov_tau <= 0:
            raise ValueError("elo_mov_tau must be positive when set")
        if self.international_elo_k_multiplier <= 0:
            raise ValueError("international_elo_k_multiplier must be positive")
        if self.player_elo_k <= 0:
            raise ValueError("player_elo_k must be positive")
        if not (0.0 <= self.player_season_carryover <= 1.0):
            raise ValueError("player_season_carryover must be in [0, 1]")
        if self.prof_shrink_games <= 0:
            raise ValueError("prof_shrink_games must be positive")
        if not (0.0 <= self.wr_prior_solo_weight <= 2.0):
            raise ValueError("wr_prior_solo_weight must be in [0, 2]")
        if self.wr_prior_min_solo_games < 0:
            raise ValueError("wr_prior_min_solo_games must be non-negative")
        if not (0.0 < self.wr_prior_clip <= 0.5):
            raise ValueError("wr_prior_clip must be in (0, 0.5]")
        if not (0.0 <= self.elo_season_carryover <= 1.0):
            raise ValueError("elo_season_carryover must be in [0, 1]")
        if self.champion_wr_half_life_days <= 0:
            raise ValueError("champion_wr_half_life_days must be positive")
        if self.champion_wr_shrink_games < 0:
            raise ValueError("champion_wr_shrink_games must be non-negative")
        if self.strength_feature_scale <= 0:
            raise ValueError("strength_feature_scale must be positive")
        if self.elo_feature_scale <= 0:
            raise ValueError("elo_feature_scale must be positive")
        if self.premier_league_target_share is not None and not (
            0.0 < self.premier_league_target_share < 1.0
        ):
            raise ValueError("premier_league_target_share must be in (0, 1) or None")
        if not (0.0 < self.patch_decay_base <= 1.0):
            raise ValueError("patch_decay_base must be in (0, 1]")
        if self.min_patch is not None:
            # Validate it parses as major[.minor] (compared numerically, not
            # lexically, in features.parse_patch). Inlined here rather than
            # importing features to avoid a circular import at module load.
            head = str(self.min_patch).strip().split(".")[0]
            if not head.isdigit():
                raise ValueError(
                    f"min_patch must be a patch string like '25.1' (got {self.min_patch!r})"
                )


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
# Plain-HTTP mirrors of the same per-season Oracle's Elixir CSV, committed to
# public GitHub repos by community projects. raw.githubusercontent.com has NO
# per-file download quota (unlike the shared Google Drive file, which is so
# widely used it's often over Google's "too many users" quota) and is
# reachable from essentially anywhere, so this is the reliable workhorse when
# the Drive download is quota-locked. Mirrors can lag the daily Drive file by
# a few days, so the Drive endpoints are still tried first for freshness.
ORACLES_ELIXIR_MIRROR_URLS: dict[int, list[str]] = {
    2026: [
        "https://raw.githubusercontent.com/HKB06/ProjectPredictionLOl/HEAD/"
        "lol-predictor/data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv",
    ],
    # Self-mirror: the frozen 2025 season CSV published to this repo's
    # data-cache branch by .github/workflows/mirror-past-season.yml. A
    # finished season's file never changes, so this link never goes stale --
    # and raw.githubusercontent.com is reachable both from CI and from the
    # dev sandbox (where Google Drive is egress-blocked).
    2025: [
        "https://raw.githubusercontent.com/Specifxx/CompStrength/data-cache/"
        "2025_LoL_esports_match_data_from_OraclesElixir.csv",
    ],
}

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
