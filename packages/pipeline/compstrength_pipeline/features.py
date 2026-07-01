"""Core champion-strength feature computation: empirical-Bayes blending of
pro and solo-queue win rates, with exponential recency decay.

Why empirical-Bayes shrinkage?
-------------------------------
Professional LoL game counts per champion per patch are tiny -- a champion
might be picked only a handful of times in a 90-day window on a given
patch. The raw (unweighted) pro win rate for such a champion is extremely
noisy: a champion that goes 2-0 in its only two pro games has an observed
100% win rate, which obviously shouldn't be taken at face value.

Empirical-Bayes shrinkage addresses this by blending the noisy raw
estimate with an *informative prior* -- here, a mix of the global baseline
(50%) and the champion's solo-queue win rate (which has a much larger,
more stable sample size, though it reflects a different skill
distribution/meta than pro play). The blend is a precision-weighted
average:

    blended = (prior_mean * prior_games + raw_pro_mean * pro_games_decayed)
              / (prior_games + pro_games_decayed)

``prior_games`` (``PRIOR_GAMES``) is the shrinkage strength, expressed as
a "pseudo-count" of games behind the prior: it's the number of decayed pro
games at which the prior and the observed pro data are weighted equally.
As ``pro_games_decayed`` grows far beyond ``prior_games``, the blended
estimate converges to the raw pro win rate (the data dominates). As it
shrinks toward zero, the blended estimate converges to the prior mean
(the prior dominates) -- exactly the desired behavior for a champion with
zero or very few pro games.

Recency decay
--------------
Because patches change balance frequently, older pro games are less
informative about a champion's *current* strength than recent ones. We
apply exponential decay to each game's weight based on how long ago it
was played, with a configurable half-life (``PATCH_HALF_LIFE_DAYS``):

    weight = 0.5 ** (days_since_game / PATCH_HALF_LIFE_DAYS)

This is used both to compute a "decayed sample size" (``proGamesDecayed``,
the sum of weights, which plays the role of ``pro_games`` above) and a
weighted mean win rate (``proWinRateRaw``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from compstrength_pipeline.champions import get_full_champion_roster
from compstrength_pipeline.config import PipelineConfig


def parse_patch(patch: object) -> tuple[int, int] | None:
    """Parse a patch string like ``"26.13"`` into a ``(major, minor)`` tuple
    for correct chronological comparison (``(26, 13) > (26, 2) > (25, 1)``).

    Lexical/string comparison is wrong here ("26.13" < "26.2" as text), so
    ratings/floors must compare these tuples, never the raw strings. Returns
    ``None`` for anything that doesn't parse to at least a numeric major
    (e.g. NaN, empty, or a non-numeric label).
    """
    if not isinstance(patch, str):
        return None
    parts = patch.strip().split(".")
    if not parts or not parts[0]:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
    except ValueError:
        return None
    return (major, minor)


def restrict_to_min_patch(
    games_df: pd.DataFrame, bans_df: pd.DataFrame, min_patch: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop games on patches OLDER than ``min_patch`` (inclusive floor).

    Patches are compared as parsed ``(major, minor)`` tuples (see
    :func:`parse_patch`), never lexically. If ``min_patch`` is None/unparseable
    or applying it would drop every game (e.g. the synthetic fixture's 14.x
    patches against a "25.1" floor), the filter is skipped and all games are
    kept -- so offline/dev/test runs still work while real runs get the floor.
    """
    if min_patch is None or games_df.empty or "patch" not in games_df.columns:
        return games_df, bans_df
    floor = parse_patch(min_patch)
    if floor is None:
        return games_df, bans_df

    parsed = games_df["patch"].map(parse_patch)
    keep = parsed.map(lambda t: t is not None and t >= floor)
    if not keep.any():
        warnings.warn(
            f"min_patch floor {min_patch!r} would drop every game "
            f"(newest present patch parses below it); skipping the floor. "
            "This is expected offline on the synthetic fixture; on real data "
            "it would mean the source has no games at or after the floor."
        )
        return games_df, bans_df

    restricted_games = games_df[keep].reset_index(drop=True)
    if bans_df is None or bans_df.empty:
        restricted_bans = bans_df
    else:
        surviving = set(restricted_games["gameid"])
        restricted_bans = bans_df[bans_df["gameid"].isin(surviving)].reset_index(drop=True)
    return restricted_games, restricted_bans


def logit(p: float) -> float:
    """Log-odds transform: ``ln(p / (1 - p))``, with ``p`` clipped to [0.01, 0.99]."""
    p_clipped = min(max(p, 0.01), 0.99)
    return math.log(p_clipped / (1.0 - p_clipped))


def decay_weight(days_since_game: float, half_life_days: float) -> float:
    """Exponential recency-decay weight for a game played ``days_since_game`` ago.

    Returns ``0.5 ** (days_since_game / half_life_days)``. A game played
    today gets weight 1.0; a game played ``half_life_days`` ago gets
    weight 0.5; a game played ``2 * half_life_days`` ago gets weight 0.25;
    etc. Negative ``days_since_game`` (a game "in the future" relative to
    the reference date) is clipped to 0 days.
    """
    days = max(days_since_game, 0.0)
    return 0.5 ** (days / half_life_days)


@dataclass(frozen=True)
class ShrinkageResult:
    """Result of blending a champion's raw pro win rate with its prior."""

    pro_games_decayed: float
    pro_win_rate_raw: float
    solo_win_rate: float
    solo_games: int
    prior_mean_win_rate: float
    blended_win_rate: float
    strength_score: float
    sample_confidence: str


def compute_prior_mean_win_rate(
    solo_win_rate: float, global_mean: float, solo_queue_weight: float
) -> float:
    """Blend the global mean and solo-queue win rate into an informative prior.

    ``priorMeanWinRate = GLOBAL_MEAN * (1 - SOLO_QUEUE_WEIGHT) + soloWinRate * SOLO_QUEUE_WEIGHT``
    """
    return global_mean * (1.0 - solo_queue_weight) + solo_win_rate * solo_queue_weight


def compute_blended_win_rate(
    prior_mean_win_rate: float,
    prior_games: float,
    pro_win_rate_raw: float,
    pro_games_decayed: float,
) -> float:
    """Precision-weighted (empirical-Bayes) blend of prior and observed pro win rate.

    ``blended = (prior_mean * prior_games + raw_pro_mean * pro_games_decayed)
                / (prior_games + pro_games_decayed)``

    If both ``prior_games`` and ``pro_games_decayed`` are zero (degenerate
    case), returns ``prior_mean_win_rate`` unchanged.
    """
    denom = prior_games + pro_games_decayed
    if denom <= 0:
        return prior_mean_win_rate
    return (
        prior_mean_win_rate * prior_games + pro_win_rate_raw * pro_games_decayed
    ) / denom


def compute_strength_score(blended_win_rate: float, global_mean: float) -> float:
    """``strengthScore = logit(blendedWinRate) - logit(GLOBAL_MEAN)``.

    Centers the score at 0 for a champion exactly at the global mean win
    rate; positive scores indicate an above-average (stronger) champion,
    negative scores a below-average (weaker) champion, on a log-odds
    scale (so scores are roughly additive/comparable in a linear model,
    which is exactly how ``train_model.py`` uses them).
    """
    return logit(blended_win_rate) - logit(global_mean)


def sample_confidence_label(pro_games_decayed: float) -> str:
    """"low" if < 5 decayed pro games, "medium" if < 20, else "high"."""
    if pro_games_decayed < 5:
        return "low"
    if pro_games_decayed < 20:
        return "medium"
    return "high"


def compute_champion_shrinkage(
    pro_games_decayed: float,
    pro_win_rate_raw: float,
    solo_win_rate: float,
    solo_games: int,
    config: PipelineConfig,
) -> ShrinkageResult:
    """Run the full empirical-Bayes shrinkage pipeline for a single champion.

    See the module docstring for the statistical intuition. This is the
    single source of truth for the blending math; ``compute_champion_features``
    calls this after computing the raw decayed pro stats from the games
    table.
    """
    prior_mean = compute_prior_mean_win_rate(
        solo_win_rate, config.global_mean, config.solo_queue_weight
    )
    blended = compute_blended_win_rate(
        prior_mean, config.prior_games, pro_win_rate_raw, pro_games_decayed
    )
    strength = compute_strength_score(blended, config.global_mean)
    confidence = sample_confidence_label(pro_games_decayed)

    return ShrinkageResult(
        pro_games_decayed=pro_games_decayed,
        pro_win_rate_raw=pro_win_rate_raw,
        solo_win_rate=solo_win_rate,
        solo_games=solo_games,
        prior_mean_win_rate=prior_mean,
        blended_win_rate=blended,
        strength_score=strength,
        sample_confidence=confidence,
    )


def international_league_multiplier(
    league: object,
    international_leagues: frozenset[str],
    multiplier: float,
) -> float:
    """``multiplier`` if ``league`` (matched case-insensitively) is an
    international event, else ``1.0``. Missing/non-string ``league`` values
    (e.g. NaN) are treated as a regular (non-international) game.
    """
    if not isinstance(league, str):
        return 1.0
    return multiplier if league.strip().upper() in international_leagues else 1.0


def patch_weight_series(
    patches: pd.Series,
    patch_distances: dict[str, int] | None,
    patch_decay_base: float,
) -> pd.Series:
    """Per-row ``patch_decay_base ** ordinal_distance`` multiplier.

    Games on the newest patch (distance 0) get 1.0, the previous patch
    ``patch_decay_base``, two-back ``patch_decay_base**2``, etc. Patches not
    present in ``patch_distances`` (shouldn't happen if it was built from the
    same games) fall back to distance 0 (full weight). Returns all-ones when
    patch weighting is disabled (``patch_distances`` is None or
    ``patch_decay_base >= 1``), so the multiplier is a no-op for callers/tests
    that don't opt in.
    """
    if patch_distances is None or patch_decay_base >= 1.0:
        return pd.Series(1.0, index=patches.index)
    return patches.map(lambda p: patch_decay_base ** patch_distances.get(p, 0))


def compute_decayed_pro_stats(
    champion_games: pd.DataFrame,
    reference_date: pd.Timestamp,
    half_life_days: float,
    international_leagues: frozenset[str] = frozenset(),
    international_weight_multiplier: float = 1.0,
    patch_distances: dict[str, int] | None = None,
    patch_decay_base: float = 1.0,
) -> tuple[float, float]:
    """Compute (pro_games_decayed, pro_win_rate_raw) for one champion's games.

    Args:
        champion_games: Rows (already filtered to a single champion) with
            at least ``date`` (tz-aware timestamp) and ``result`` (0/1)
            columns. Should already be restricted to the desired trailing
            window by the caller. A ``league`` column is used if present
            (see ``international_leagues``); its absence is treated as "no
            international boost for any row". A ``patch`` column is used if
            present together with ``patch_distances`` (see below).
        reference_date: The "current" date used to compute recency
            (typically the latest date in the whole dataset, i.e. "today"
            for the purposes of this pipeline run).
        half_life_days: Passed through to :func:`decay_weight`.
        international_leagues: ``league`` values (case-insensitive) that
            get the international weight boost -- see
            ``PipelineConfig.international_leagues``.
        international_weight_multiplier: Extra multiplier applied on top of
            the recency-decay weight for games in ``international_leagues``
            (e.g. MSI, Worlds) -- these concentrate the best teams from
            every region playing on one current patch, so they're an
            unusually high-signal sample of the current meta and are
            weighted up relative to a typical regional-split game.
        patch_distances: ``{patch: ordinal_distance_from_newest}`` (see
            :func:`patch_ordinal_distances`). When provided together with a
            ``patch_decay_base`` < 1, each game is additionally weighted by
            ``patch_decay_base ** distance`` so the latest patch(es) dominate
            -- this is *patch*-recency weighting, on top of the *calendar-day*
            recency decay. ``None`` disables it (old day-decay-only behavior).
        patch_decay_base: See ``PipelineConfig.patch_decay_base``.

    Returns:
        ``(pro_games_decayed, pro_win_rate_raw)``. If ``champion_games``
        is empty, returns ``(0.0, 0.0)``.
    """
    if champion_games.empty:
        return 0.0, 0.0

    days_since = (reference_date - champion_games["date"]).dt.total_seconds() / 86400.0
    weights = days_since.map(lambda d: decay_weight(d, half_life_days))

    if "league" in champion_games.columns:
        league_boost = champion_games["league"].map(
            lambda lg: international_league_multiplier(
                lg, international_leagues, international_weight_multiplier
            )
        )
        weights = weights * league_boost

    if "patch" in champion_games.columns:
        weights = weights * patch_weight_series(
            champion_games["patch"], patch_distances, patch_decay_base
        )

    total_weight = float(weights.sum())
    if total_weight <= 0:
        return 0.0, 0.0

    weighted_win_sum = float((weights * champion_games["result"]).sum())
    win_rate_raw = weighted_win_sum / total_weight
    return total_weight, win_rate_raw


def _select_pro_window(
    champion_games: pd.DataFrame,
    reference_date: pd.Timestamp,
    pro_window_days: int,
    min_games: int = 1,
) -> pd.DataFrame:
    """Select games within the trailing window, extending backward if too sparse.

    Starts with the ``pro_window_days`` trailing window. If that window
    contains fewer than ``min_games`` rows for this champion (but the
    champion has *some* games at all, just outside the window), doubles
    the window repeatedly (up to a generous cap) until it finds at least
    ``min_games`` rows or exhausts all available history. This implements
    the spec's "extend window backward across patches if needed to get
    any data" behavior.
    """
    if champion_games.empty:
        return champion_games

    window = pro_window_days
    # Cap how far back we'll look: 16x the base window (plenty for a
    # hobby-scale fixture/dataset; avoids unbounded loops on pathological
    # inputs).
    max_window = pro_window_days * 16
    selected = champion_games[
        champion_games["date"] >= reference_date - pd.Timedelta(days=window)
    ]
    while len(selected) < min_games and window < max_window:
        window *= 2
        selected = champion_games[
            champion_games["date"] >= reference_date - pd.Timedelta(days=window)
        ]

    if selected.empty:
        # Still nothing (e.g. champion truly has zero games) -- fall back
        # to whatever the widest window found (could be empty).
        return selected

    return selected


def _patches_by_recency(games_df: pd.DataFrame) -> list[str]:
    """Distinct patches present in ``games_df``, ordered newest-first by patch
    NUMBER (parsed ``(major, minor)`` -- see :func:`parse_patch`), not by
    lexicographic/semver text sort and not by max game date.

    Ordering by patch number (rather than by the max game *date* of each
    patch) matters because pro regions do not all move to a new patch on the
    same calendar day: an older patch can carry games with later timestamps
    than a newer patch (e.g. one region still on 16.11 plays after another has
    started 16.12). Date-ranking would then rank the older patch as "newest"
    and hand it distance 0 / full patch-decay weight -- the exact opposite of
    the intended "latest patch weighted most". Numeric patch order is the
    correct notion of "which patch is newer". Patches whose string doesn't
    parse to a number are sorted last (by max date among themselves) so they
    never masquerade as the newest patch.
    """
    if games_df.empty or "patch" not in games_df.columns:
        return []
    patch_max_dates = games_df.groupby("patch")["date"].max()
    # Sort by (parsed patch tuple, max date) descending. Unparseable patches
    # get a -inf sentinel so they rank after every real patch.
    def _key(patch: object) -> tuple:
        parsed = parse_patch(patch)
        max_date = patch_max_dates[patch]
        date_ord = max_date.value if pd.notna(max_date) else -1
        if parsed is None:
            return (0, -1, -1, date_ord)
        return (1, parsed[0], parsed[1], date_ord)

    return sorted(patch_max_dates.index, key=_key, reverse=True)


def newest_patch(games_df: pd.DataFrame) -> str | None:
    """Return the numerically-newest patch present in ``games_df`` (the one
    used as the "current patch" label and solo-queue lookup key), or ``None``
    if there are no patches. This is ``_patches_by_recency(...)[0]`` -- i.e.
    the newest by patch NUMBER, so it never reports an older patch as current
    just because that patch happens to have a later-dated game (see
    :func:`_patches_by_recency`)."""
    ordered = _patches_by_recency(games_df)
    return ordered[0] if ordered else None


def patch_ordinal_distances(games_df: pd.DataFrame) -> dict[str, int]:
    """Map each patch in ``games_df`` to its ordinal distance from the newest.

    The most recent patch (by max game date) gets distance 0, the previous
    patch 1, two-back 2, and so on -- so a game's patch weighting can be
    ``patch_decay_base ** distance``, making the latest patch(es) dominate.
    Patches are ordered by date, never by lexically sorting the patch string
    (see :func:`_patches_by_recency`).
    """
    return {patch: distance for distance, patch in enumerate(_patches_by_recency(games_df))}


def select_recent_games(games_df: pd.DataFrame, target_games: int) -> list:
    """Return the up-to-``target_games`` most recent distinct ``gameid``s.

    Games are ordered by each gameid's max ``date`` (most recent first),
    with no regard for which patch they're on -- unlike a patch-count
    cutoff, this guarantees a target *sample size* rather than an
    unpredictable one that depends on how many games happen to exist on
    the last patch or two. If fewer than ``target_games`` distinct games
    exist at all, all of them are returned.

    If ``games_df`` is empty or has no ``gameid``/``date`` columns, returns
    an empty list.
    """
    if games_df.empty or "gameid" not in games_df.columns:
        return []

    game_max_dates = games_df.groupby("gameid")["date"].max().sort_values(ascending=False)
    return list(game_max_dates.index[:target_games])


def restrict_to_recent_games(
    games_df: pd.DataFrame, bans_df: pd.DataFrame, target_games: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Hard-filter ``games_df``/``bans_df`` down to the most recent
    ``target_games`` games, regardless of which patch(es) they're on.

    Older games beyond ``target_games`` are excluded outright (to bound
    compute and avoid ancient, barely-relevant history), but -- per
    ``PipelineConfig.target_training_games`` -- games are no longer
    dropped just for being on an older patch: as long as they're within
    the ``target_games`` most recent, they're included and simply count
    for less via the existing day-based exponential recency decay.

    Args:
        games_df: Cleaned per-player-game table with ``gameid``/``date``/
            ``patch`` columns.
        bans_df: Cleaned bans table (filtered to the same surviving
            gameids as ``games_df``).
        target_games: See ``PipelineConfig.target_training_games``.

    Returns:
        ``(restricted_games_df, restricted_bans_df, patches_used)`` where
        ``patches_used`` is the most-recent-first list of every distinct
        patch present among the surviving games (see
        :func:`_patches_by_recency`) -- this can span many patches, since
        selection is by game count, not patch count.
    """
    selected_gameids = select_recent_games(games_df, target_games)
    if not selected_gameids:
        return games_df, bans_df, _patches_by_recency(games_df)

    selected_gameids_set = set(selected_gameids)
    restricted_games = games_df[games_df["gameid"].isin(selected_gameids_set)].reset_index(
        drop=True
    )
    patches_used = _patches_by_recency(restricted_games)

    if bans_df is None or bans_df.empty:
        restricted_bans = bans_df
    else:
        restricted_bans = bans_df[bans_df["gameid"].isin(selected_gameids_set)].reset_index(
            drop=True
        )

    return restricted_games, restricted_bans, patches_used


def compute_champion_features(
    games_df: pd.DataFrame,
    bans_df: pd.DataFrame,
    solo_winrates: dict[str, tuple[float, int]],
    config: PipelineConfig,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute the full per-champion feature table for the current patch.

    Args:
        games_df: Cleaned per-player-game table (post ``etl.build_raw_tables``),
            with at least [gameid, date, patch, champion, result] columns.
            Before anything else, this is hard-restricted to the
            ``config.target_training_games`` most recent games overall,
            regardless of patch (see :func:`restrict_to_recent_games`) --
            window selection, decay, and pick/ban rates are all computed
            over that restricted set only.
        bans_df: Cleaned bans table (post ``etl.build_raw_tables``), with
            at least [gameid, champion] columns.
        solo_winrates: ``{champion: (winrate, games)}`` for the current
            patch, e.g. from ``SoloQueueSource.get_champion_winrates``.
            Champions absent from this dict fall back to
            ``(config.global_mean, 0)``.
        config: Hyperparameters (see ``config.PipelineConfig``).
        reference_date: The "as of" date for recency decay and the
            trailing pro-games window. Defaults to the max date present
            in the restricted ``games_df``.

    Returns:
        A DataFrame indexed by champion name with columns: primaryRole,
        proGames (decayed, rounded to nearest int), proWinRate, soloGames,
        soloWinRate, blendedWinRate, strengthScore, pickRate, banRate,
        sampleConfidence.
    """
    if games_df.empty:
        return pd.DataFrame(
            columns=[
                "primaryRole", "proGames", "proWinRate", "soloGames",
                "soloWinRate", "blendedWinRate", "strengthScore", "pickRate",
                "banRate", "sampleConfidence",
            ]
        )

    games_df, bans_df = restrict_to_min_patch(games_df, bans_df, config.min_patch)
    games_df, bans_df, _patches_used = restrict_to_recent_games(
        games_df, bans_df, config.target_training_games
    )

    if games_df.empty:
        return pd.DataFrame(
            columns=[
                "primaryRole", "proGames", "proWinRate", "soloGames",
                "soloWinRate", "blendedWinRate", "strengthScore", "pickRate",
                "banRate", "sampleConfidence",
            ]
        )

    if reference_date is None:
        reference_date = games_df["date"].max()

    # Canonicalize solo-queue keys onto the roster's spelling so they join with
    # the (already-canonicalized) game champion names. Without this, a solo
    # source that spells a champion differently ("Jarvan Iv"/"Leblanc" vs the
    # canonical "Jarvan IV"/"LeBlanc") would (a) miss the prior lookup below and
    # (b) leak a phantom duplicate champion row via the ``data_champions``
    # union -- the very split-key bug the game-side canonicalization fixed, at
    # the game<->solo boundary. We only remap names that hit the KNOWN roster
    # (case-insensitively) and leave unrecognized names untouched, so this never
    # mangles a name the roster doesn't know.
    if solo_winrates:
        _canon = {name.casefold(): name for name in get_full_champion_roster()}
        solo_winrates = {
            _canon.get(str(name).strip().casefold(), name): stats
            for name, stats in solo_winrates.items()
        }

    # Patch-recency weighting: newest patch (by number) = distance 0, previous
    # = 1, etc. Games are additionally weighted by patch_decay_base**distance
    # so the latest patch(es) dominate the ratings (see compute_decayed_pro_stats).
    patch_distances = patch_ordinal_distances(games_df)

    total_games = games_df["gameid"].nunique()
    total_bans = 0 if bans_df is None or bans_df.empty else len(bans_df)

    # The champion universe is the UNION of the full known roster (so every
    # champion is selectable on the site, including ones with zero pro games
    # in this window -- they simply fall back to their solo-queue-informed
    # prior, same as any other sparse-data champion) and whatever actually
    # appears in the data (so a brand-new champion missing from the roster
    # snapshot still gets rated correctly rather than silently dropped).
    full_roster = get_full_champion_roster()
    data_champions = set(games_df["champion"].dropna().unique()) | set(solo_winrates.keys())
    champions = sorted(set(full_roster.keys()) | data_champions)
    rows: dict[str, dict] = {}

    for champion in champions:
        champ_games_all = games_df[games_df["champion"] == champion]
        champ_games = _select_pro_window(champ_games_all, reference_date, config.pro_window_days)

        pro_games_decayed, pro_win_rate_raw = compute_decayed_pro_stats(
            champ_games,
            reference_date,
            config.patch_half_life_days,
            international_leagues=config.international_leagues,
            international_weight_multiplier=config.international_weight_multiplier,
            patch_distances=patch_distances,
            patch_decay_base=config.patch_decay_base,
        )

        solo_win_rate, solo_games = solo_winrates.get(champion, (config.global_mean, 0))

        shrinkage = compute_champion_shrinkage(
            pro_games_decayed=pro_games_decayed,
            pro_win_rate_raw=pro_win_rate_raw,
            solo_win_rate=solo_win_rate,
            solo_games=solo_games,
            config=config,
        )

        pick_count = len(champ_games_all["gameid"].unique()) if not champ_games_all.empty else 0
        pick_rate = pick_count / total_games if total_games else 0.0

        ban_count = 0
        if bans_df is not None and not bans_df.empty:
            ban_count = int((bans_df["champion"] == champion).sum())
        ban_rate = ban_count / total_games if total_games else 0.0

        rows[champion] = {
            "primaryRole": full_roster.get(champion, "MID"),
            "proGames": int(round(shrinkage.pro_games_decayed)),
            "proWinRate": shrinkage.pro_win_rate_raw,
            "soloGames": solo_games,
            "soloWinRate": solo_win_rate,
            "blendedWinRate": shrinkage.blended_win_rate,
            "strengthScore": shrinkage.strength_score,
            "pickRate": pick_rate,
            "banRate": ban_rate,
            "sampleConfidence": shrinkage.sample_confidence,
        }

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "champion"
    return result
