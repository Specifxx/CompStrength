"""Pairwise historical predictors: champion-pair synergy and champion-vs-
champion matchup history.

The main model (``train_model.py``) uses each champion's individually
blended ``strengthScore`` (see ``features.py``), summed across a team, as
its primary predictor. But some champion pairs perform meaningfully better
or worse together (or against each other in a lane/role) than their
individual ratings alone would predict -- e.g. certain engage-supports pair
unusually well with certain all-in junglers, or a particular top-lane
matchup is historically lopsided beyond what either champion's overall
strength explains.

This module computes two additional historical-data-driven residual
signals, both restricted to the same patch-restricted, decay-weighted
window used by ``features.compute_champion_features``:

Synergy (within-team, unordered pairs)
---------------------------------------
For every unordered pair of champions (A, B) that appeared on the *same*
team in the *same* game, we compute the decayed win rate of that pairing
and compare it to what the individual ``strengthScore`` values would
already predict:

    expectedLogit  = strengthScore[A] + strengthScore[B]
    residualRaw    = logit(pairWinRateRaw) - expectedLogit

``residualRaw`` is the incremental log-odds effect of this specific
pairing, beyond what's explained by their individual ratings. Small
samples are shrunk toward 0 via the same empirical-Bayes precision-weighted
blend pattern used elsewhere in this codebase, with a prior mean of 0 (no
synergy effect until proven otherwise) and a configurable pseudo-count
(``config.synergy_prior_games``).

Matchup (cross-team, same-role, ordered pairs)
------------------------------------------------
For every ordered pair (A, B) where A and B occupied the *same role* on
*opposing* teams in the same game, we compute A's team's decayed win rate
in those specific A-vs-B-same-role games, relative to what A's own
strengthScore alone would predict:

    expectedLogit  = strengthScore[A]
    residualRaw    = logit(matchupWinRateRaw) - expectedLogit

(B's opposing strength doesn't need subtracting here -- the main model's
``score_diff`` feature already separately accounts for B's overall
strength via B's own strengthScore in the opposing side's sum; this
residual only measures A's team's historical result specifically when
this matchup occurs, on top of A's own rating.) Shrunk the same way with
``config.matchup_prior_games``.

Matchup residuals are directional/asymmetric by construction -- A-vs-B need
not be the negative of B-vs-A empirically (different roles, different
underlying data) -- so no attempt is made to force symmetry.

Both tables only include pairs that actually co-occurred at least once in
the decay-weighted window (``pairGamesDecayed`` / ``matchupGamesDecayed``
> 0); pairs that never co-occurred are omitted entirely (they implicitly
have residual 0, i.e. purely additive individual strength).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from compstrength_pipeline.config import PipelineConfig
from compstrength_pipeline.features import (
    decay_weight,
    league_weight_multiplier,
    logit,
    patch_ordinal_distances,
    patch_weight_series,
)


@dataclass(frozen=True)
class PairStat:
    """One pairwise (synergy or matchup) residual entry."""

    games_decayed: float
    win_rate_raw: float
    expected_logit: float
    residual_raw: float
    residual: float  # shrunk toward 0


def _shrink_residual(residual_raw: float, games_decayed: float, prior_games: float) -> float:
    """Empirical-Bayes precision-weighted blend of ``residual_raw`` toward 0.

    ``shrunk = (0 * prior_games + residual_raw * games_decayed) / (prior_games + games_decayed)``

    Mirrors ``features.compute_blended_win_rate``'s pattern, but blending a
    log-odds residual toward a prior mean of 0 instead of blending a win
    rate toward an informative prior mean.
    """
    denom = prior_games + games_decayed
    if denom <= 0:
        return 0.0
    return (residual_raw * games_decayed) / denom


def _decayed_group_stats(
    group: pd.DataFrame,
    reference_date: pd.Timestamp,
    half_life_days: float,
    international_leagues: frozenset[str] = frozenset(),
    international_weight_multiplier: float = 1.0,
    patch_distances: dict[str, int] | None = None,
    patch_decay_base: float = 1.0,
    major_leagues: frozenset[str] = frozenset(),
    major_league_weight_multiplier: float = 1.0,
) -> tuple[float, float]:
    """Decayed (games, win_rate) for an arbitrary set of game-outcome rows.

    ``group`` must have ``date`` (tz-aware) and a binary ``win`` column
    (1 if the perspective team/pairing won that game, else 0), and may have
    ``league`` (international-event boost) and ``patch`` (patch-recency
    weighting) columns. Mirrors ``features.compute_decayed_pro_stats`` --
    including the same ``patch_decay_base ** patch_ordinal_distance``
    weighting so synergy/matchup residuals also lean on the latest patches --
    but generalized to an arbitrary "win" column rather than one row per
    champion.
    """
    if group.empty:
        return 0.0, 0.0
    days_since = (reference_date - group["date"]).dt.total_seconds() / 86400.0
    weights = days_since.map(lambda d: decay_weight(d, half_life_days))
    if "league" in group.columns:
        league_boost = group["league"].map(
            lambda lg: league_weight_multiplier(
                lg,
                international_leagues,
                international_weight_multiplier,
                major_leagues,
                major_league_weight_multiplier,
            )
        )
        weights = weights * league_boost
    if "patch" in group.columns:
        weights = weights * patch_weight_series(
            group["patch"], patch_distances, patch_decay_base
        )
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return 0.0, 0.0
    weighted_win_sum = float((weights * group["win"]).sum())
    return total_weight, weighted_win_sum / total_weight


def _build_team_game_rows(games_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (gameid, side) with the sorted list of that side's champions.

    Used as the basis for both synergy (within-side pairs) and matchup
    (cross-side same-role pairs) extraction.
    """
    rows = []
    for (gameid, side), group in games_df.groupby(["gameid", "side"]):
        # All 5 rows of a side should share one result; take the majority
        # vote rather than a single row so one NaN-coerced/mislabeled row
        # can't flip the whole side's win label.
        won = int(round(float(group["result"].mean())))
        champs_by_role = dict(zip(group["position"], group["champion"]))
        rows.append(
            {
                "gameid": gameid,
                "side": side,
                "date": group["date"].iloc[0],
                "league": group["league"].iloc[0] if "league" in group.columns else None,
                "patch": group["patch"].iloc[0] if "patch" in group.columns else None,
                "won": won,
                "champions": sorted(group["champion"].tolist()),
                "champs_by_role": champs_by_role,
            }
        )
    return pd.DataFrame(rows)


def compute_synergy_table(
    games_df: pd.DataFrame,
    champion_strength: dict[str, float],
    config: PipelineConfig,
    reference_date: pd.Timestamp | None = None,
) -> dict[str, PairStat]:
    """Compute within-team pairwise synergy residuals.

    Args:
        games_df: Cleaned, patch-restricted per-player-game table with at
            least [gameid, date, side, position, champion, result]
            columns.
        champion_strength: ``{champion: strengthScore}`` from
            ``features.compute_champion_features``.
        config: Hyperparameters (uses ``synergy_prior_games`` and
            ``patch_half_life_days``).
        reference_date: "As of" date for recency decay. Defaults to the
            max date in ``games_df``.

    Returns:
        ``{"ChampionA|ChampionB": PairStat}`` (keys sorted alphabetically,
        joined with "|"), only for pairs with ``games_decayed > 0``.
    """
    if games_df.empty:
        return {}

    if reference_date is None:
        reference_date = games_df["date"].max()

    patch_distances = patch_ordinal_distances(games_df)
    team_rows = _build_team_game_rows(games_df)

    # For each unordered champion pair, collect (date, won) rows across all
    # team-game instances where both champions appeared on that side.
    pair_rows: dict[tuple[str, str], list[dict]] = {}
    for _, row in team_rows.iterrows():
        champs = row["champions"]
        for a, b in itertools.combinations(sorted(set(champs)), 2):
            pair_rows.setdefault((a, b), []).append(
                {"date": row["date"], "win": row["won"], "league": row["league"], "patch": row["patch"]}
            )

    result: dict[str, PairStat] = {}
    for (a, b), records in pair_rows.items():
        df = pd.DataFrame(records)
        games_decayed, win_rate_raw = _decayed_group_stats(
            df,
            reference_date,
            config.patch_half_life_days,
            international_leagues=config.international_leagues,
            international_weight_multiplier=config.international_weight_multiplier,
            patch_distances=patch_distances,
            patch_decay_base=config.patch_decay_base,
            major_leagues=config.major_leagues,
            major_league_weight_multiplier=config.major_league_weight_multiplier,
        )
        if games_decayed <= 0:
            continue

        expected_logit = champion_strength.get(a, 0.0) + champion_strength.get(b, 0.0)
        residual_raw = logit(win_rate_raw) - expected_logit
        residual = _shrink_residual(residual_raw, games_decayed, config.synergy_prior_games)

        key = f"{a}|{b}"
        result[key] = PairStat(
            games_decayed=games_decayed,
            win_rate_raw=win_rate_raw,
            expected_logit=expected_logit,
            residual_raw=residual_raw,
            residual=residual,
        )

    return result


def compute_matchup_table(
    games_df: pd.DataFrame,
    champion_strength: dict[str, float],
    config: PipelineConfig,
    reference_date: pd.Timestamp | None = None,
) -> dict[str, PairStat]:
    """Compute cross-team, same-role matchup residuals.

    Args:
        games_df: Cleaned, patch-restricted per-player-game table with at
            least [gameid, date, side, position, champion, result]
            columns.
        champion_strength: ``{champion: strengthScore}`` from
            ``features.compute_champion_features``.
        config: Hyperparameters (uses ``matchup_prior_games`` and
            ``patch_half_life_days``).
        reference_date: "As of" date for recency decay. Defaults to the
            max date in ``games_df``.

    Returns:
        ``{"ChampionA>ChampionB": PairStat}`` (directional: A's team's
        result when A faced B in the same role), only for pairs with
        ``games_decayed > 0``. Both "A>B" and "B>A" may be present
        independently.
    """
    if games_df.empty:
        return {}

    if reference_date is None:
        reference_date = games_df["date"].max()

    patch_distances = patch_ordinal_distances(games_df)
    team_rows = _build_team_game_rows(games_df)

    # Pair up the two sides within each game.
    pair_records: dict[tuple[str, str], list[dict]] = {}
    for gameid, group in team_rows.groupby("gameid"):
        if len(group) != 2:
            continue
        row_a, row_b = group.iloc[0], group.iloc[1]
        for role in set(row_a["champs_by_role"]) & set(row_b["champs_by_role"]):
            champ_a = row_a["champs_by_role"][role]
            champ_b = row_b["champs_by_role"][role]
            # Directional: from champ_a's team's perspective vs champ_b, and
            # symmetrically from champ_b's team's perspective vs champ_a.
            pair_records.setdefault((champ_a, champ_b), []).append(
                {"date": row_a["date"], "win": row_a["won"], "league": row_a["league"], "patch": row_a["patch"]}
            )
            pair_records.setdefault((champ_b, champ_a), []).append(
                {"date": row_b["date"], "win": row_b["won"], "league": row_b["league"], "patch": row_b["patch"]}
            )

    result: dict[str, PairStat] = {}
    for (a, b), records in pair_records.items():
        df = pd.DataFrame(records)
        games_decayed, win_rate_raw = _decayed_group_stats(
            df,
            reference_date,
            config.patch_half_life_days,
            international_leagues=config.international_leagues,
            international_weight_multiplier=config.international_weight_multiplier,
            patch_distances=patch_distances,
            patch_decay_base=config.patch_decay_base,
            major_leagues=config.major_leagues,
            major_league_weight_multiplier=config.major_league_weight_multiplier,
        )
        if games_decayed <= 0:
            continue

        # De-confound the matchup: the observed win rate of A's team is driven
        # by BOTH A's own strength and the opposing B's strength, so subtract
        # both. residual = logit(A-team win rate vs B) - (strength[A] -
        # strength[B]) isolates the lane/matchup effect beyond raw strength,
        # which is what train_model.py adds on top of score_diff (this avoids
        # double-counting opponent strength through the matchup channel).
        expected_logit = champion_strength.get(a, 0.0) - champion_strength.get(b, 0.0)
        residual_raw = logit(win_rate_raw) - expected_logit
        residual = _shrink_residual(residual_raw, games_decayed, config.matchup_prior_games)

        key = f"{a}>{b}"
        result[key] = PairStat(
            games_decayed=games_decayed,
            win_rate_raw=win_rate_raw,
            expected_logit=expected_logit,
            residual_raw=residual_raw,
            residual=residual,
        )

    return result


def synergy_lookup(synergy_table: dict[str, PairStat]) -> dict[str, float]:
    """``{"ChampionA|ChampionB": residual}`` convenience view."""
    return {key: stat.residual for key, stat in synergy_table.items()}


def matchup_lookup(matchup_table: dict[str, PairStat]) -> dict[str, float]:
    """``{"ChampionA>ChampionB": residual}`` convenience view."""
    return {key: stat.residual for key, stat in matchup_table.items()}


def synergy_key(champ_a: str, champ_b: str) -> str:
    """Canonical unordered synergy key: two names sorted alphabetically, joined with '|'."""
    a, b = sorted((champ_a, champ_b))
    return f"{a}|{b}"


def matchup_key(champ_a: str, champ_b: str) -> str:
    """Directional matchup key: "A>B" meaning A's team's result vs B in the same role."""
    return f"{champ_a}>{champ_b}"
