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
from dataclasses import dataclass

import numpy as np
import pandas as pd

from compstrength_pipeline.config import PipelineConfig

# Champion primary-role classification. This is a best-effort static map
# covering the champions used in our fixtures/tests; in a live pipeline
# this would ideally be sourced from Data Dragon's tags or from the
# Oracle's Elixir / Leaguepedia position column's mode per champion.
_ROLE_MAP: dict[str, str] = {
    "Aatrox": "TOP", "Camille": "TOP", "Renekton": "TOP", "Gnar": "TOP",
    "Jax": "TOP", "Ornn": "TOP", "Fiora": "TOP", "Gwen": "TOP",
    "K'Sante": "TOP", "Rumble": "TOP", "Sion": "TOP", "Kennen": "TOP",
    "Lee Sin": "JUNGLE", "Vi": "JUNGLE", "Viego": "JUNGLE", "Sejuani": "JUNGLE",
    "Nidalee": "JUNGLE", "Kindred": "JUNGLE", "Wukong": "JUNGLE",
    "Elise": "JUNGLE", "Jarvan Iv": "JUNGLE", "Graves": "JUNGLE",
    "Azir": "MID", "Ahri": "MID", "Orianna": "MID", "Syndra": "MID",
    "Akali": "MID", "Zoe": "MID", "Sylas": "MID", "Corki": "MID",
    "Leblanc": "MID", "Taliyah": "MID", "Viktor": "MID",
    "Jinx": "BOTTOM", "Kai'Sa": "BOTTOM", "Aphelios": "BOTTOM",
    "Ezreal": "BOTTOM", "Xayah": "BOTTOM", "Zeri": "BOTTOM",
    "Varus": "BOTTOM", "Samira": "BOTTOM", "Caitlyn": "BOTTOM",
    "Nautilus": "SUPPORT", "Renata Glasc": "SUPPORT", "Rakan": "SUPPORT",
    "Braum": "SUPPORT", "Karma": "SUPPORT", "Lulu": "SUPPORT",
    "Yuumi": "SUPPORT", "Thresh": "SUPPORT", "Nami": "SUPPORT",
}


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


def compute_decayed_pro_stats(
    champion_games: pd.DataFrame,
    reference_date: pd.Timestamp,
    half_life_days: float,
) -> tuple[float, float]:
    """Compute (pro_games_decayed, pro_win_rate_raw) for one champion's games.

    Args:
        champion_games: Rows (already filtered to a single champion) with
            at least ``date`` (tz-aware timestamp) and ``result`` (0/1)
            columns. Should already be restricted to the desired trailing
            window by the caller.
        reference_date: The "current" date used to compute recency
            (typically the latest date in the whole dataset, i.e. "today"
            for the purposes of this pipeline run).
        half_life_days: Passed through to :func:`decay_weight`.

    Returns:
        ``(pro_games_decayed, pro_win_rate_raw)``. If ``champion_games``
        is empty, returns ``(0.0, 0.0)``.
    """
    if champion_games.empty:
        return 0.0, 0.0

    days_since = (reference_date - champion_games["date"]).dt.total_seconds() / 86400.0
    weights = days_since.map(lambda d: decay_weight(d, half_life_days))

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


def select_recent_patches(games_df: pd.DataFrame, num_recent_patches: int) -> list[str]:
    """Return the ``num_recent_patches`` most-recent distinct patches present.

    Patches are ordered by the max ``date`` of games on that patch (not by
    lexicographic/semver sort, since patch strings like "14.2" vs "14.10"
    aren't reliably sortable as text). Returned most-recent-first.

    If ``games_df`` is empty or has no ``patch``/``date`` columns, returns
    an empty list.
    """
    if games_df.empty or "patch" not in games_df.columns:
        return []

    patch_max_dates = games_df.groupby("patch")["date"].max().sort_values(ascending=False)
    return list(patch_max_dates.index[:num_recent_patches])


def restrict_to_recent_patches(
    games_df: pd.DataFrame, bans_df: pd.DataFrame, num_recent_patches: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Hard-filter ``games_df``/``bans_df`` to only the most recent patches.

    Args:
        games_df: Cleaned per-player-game table with a ``patch`` column.
        bans_df: Cleaned bans table (filtered to the same surviving
            gameids as ``games_df``).
        num_recent_patches: See ``PipelineConfig.num_recent_patches``.

    Returns:
        ``(restricted_games_df, restricted_bans_df, patches_used)`` where
        ``patches_used`` is the most-recent-first list of patch strings
        that survived the filter (see :func:`select_recent_patches`).
    """
    patches_used = select_recent_patches(games_df, num_recent_patches)
    if not patches_used:
        return games_df, bans_df, patches_used

    restricted_games = games_df[games_df["patch"].isin(patches_used)].reset_index(drop=True)

    if bans_df is None or bans_df.empty:
        restricted_bans = bans_df
    else:
        surviving_gameids = set(restricted_games["gameid"])
        restricted_bans = bans_df[bans_df["gameid"].isin(surviving_gameids)].reset_index(drop=True)

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
            Before anything else, this is hard-restricted to games on the
            ``config.num_recent_patches`` most recent distinct patches
            (see :func:`restrict_to_recent_patches`) -- window selection,
            decay, and pick/ban rates are all computed over that
            patch-restricted set only.
        bans_df: Cleaned bans table (post ``etl.build_raw_tables``), with
            at least [gameid, champion] columns.
        solo_winrates: ``{champion: (winrate, games)}`` for the current
            patch, e.g. from ``SoloQueueSource.get_champion_winrates``.
            Champions absent from this dict fall back to
            ``(config.global_mean, 0)``.
        config: Hyperparameters (see ``config.PipelineConfig``).
        reference_date: The "as of" date for recency decay and the
            trailing pro-games window. Defaults to the max date present
            in the patch-restricted ``games_df``.

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

    games_df, bans_df, _patches_used = restrict_to_recent_patches(
        games_df, bans_df, config.num_recent_patches
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

    total_games = games_df["gameid"].nunique()
    total_bans = 0 if bans_df is None or bans_df.empty else len(bans_df)

    champions = sorted(games_df["champion"].dropna().unique())
    rows: dict[str, dict] = {}

    for champion in champions:
        champ_games_all = games_df[games_df["champion"] == champion]
        champ_games = _select_pro_window(champ_games_all, reference_date, config.pro_window_days)

        pro_games_decayed, pro_win_rate_raw = compute_decayed_pro_stats(
            champ_games, reference_date, config.patch_half_life_days
        )

        solo_win_rate, solo_games = solo_winrates.get(champion, (config.global_mean, 0))

        shrinkage = compute_champion_shrinkage(
            pro_games_decayed=pro_games_decayed,
            pro_win_rate_raw=pro_win_rate_raw,
            solo_win_rate=solo_win_rate,
            solo_games=solo_games,
            config=config,
        )

        pick_count = len(champ_games_all["gameid"].unique())
        pick_rate = pick_count / total_games if total_games else 0.0

        ban_count = 0
        if bans_df is not None and not bans_df.empty:
            ban_count = int((bans_df["champion"] == champion).sum())
        ban_rate = ban_count / total_games if total_games else 0.0

        rows[champion] = {
            "primaryRole": _ROLE_MAP.get(champion, "MID"),
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
