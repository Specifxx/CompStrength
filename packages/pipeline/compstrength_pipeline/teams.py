"""Team-strength ratings via sequential Elo over pro games.

Why Elo and not a decayed team win rate: the dataset spans ~36 leagues of
wildly different strength (LCK next to academy/ERL tiers). A raw win rate
can't tell a 60% LCK team from a 60% LCKC team, while Elo propagates
strength through cross-league games (MSI/Worlds/EWC and promotion series),
so the rating is comparable across leagues -- exactly what's needed when a
draft can pit any two teams against each other.

Leak-freedom by construction: one chronological pass over all games, and the
feature recorded for each game is the PRE-game Elo of both sides (ratings
before that game's result is applied). A game's feature therefore only ever
depends on strictly earlier games, which makes the same per-game series safe
for both walk-forward backtesting and the deployed fit. The artifact
(``data/teams.json``) ships the POST-pass ratings -- "team strength as of
today" -- for the frontend's optional team inputs.

The model consumes ``(blue_elo - red_elo) / ELO_SCALE`` so the feature lives
on roughly the same numeric scale as the other logit-ish features; the
logistic regression learns the mapping from Elo gap to win probability
itself (we deliberately do NOT bake in Elo's own 10^(d/400) curve).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

# Classic Elo parameters. ELO_K is how much one game moves a rating;
# ELO_SCALE is the logistic width (400 = chess convention). Both were
# selected by a walk-forward sweep on real 2026 data (see repo history).
DEFAULT_ELO_K = 24.0
ELO_SCALE = 400.0
INITIAL_ELO = 1500.0
# Between-season regression toward the mean: at each calendar-year (season)
# boundary in the chronological pass, every rating keeps this fraction of its
# deviation from INITIAL_ELO. Rosters turn over between seasons, so a team's
# old-season rating is real but partially stale information. Measured on
# 15,973 real games (2026 pre-game Elo predictions): carryover 0.7 improves
# the Elo-only signal from 63.86%/0.6357 to 64.11%/0.6326 vs full carryover.
# 1.0 disables (old behavior).
DEFAULT_SEASON_CARRYOVER = 0.7


@dataclass
class TeamEloResult:
    """Output of one chronological Elo pass.

    Attributes:
        per_game: DataFrame indexed by gameid with columns
            ``blue_team, red_team, blue_elo_pre, red_elo_pre`` -- the PRE-game
            ratings used as leak-free model features.
        ratings: ``{team: final_elo}`` after the full pass ("as of today").
        games_played: ``{team: number of games in the pass}``.
        last_league: ``{team: league of the team's most recent game}`` --
            display metadata for the frontend picker.
        last_played: ``{team: ISO date of the team's most recent game}``.
    """

    per_game: pd.DataFrame
    ratings: dict[str, float] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=dict)
    last_league: dict[str, str] = field(default_factory=dict)
    last_played: dict[str, str] = field(default_factory=dict)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Classic Elo expected score of A vs B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / ELO_SCALE))


def _per_game_rows(games_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 10-row-per-game player table into one row per game with
    (date, league, blue_team, red_team, blue_win), sorted chronologically
    (date, then gameid for a deterministic same-day order)."""
    rows = []
    for gameid, group in games_df.groupby("gameid"):
        blue = group[group["side"].str.lower() == "blue"]
        red = group[group["side"].str.lower() == "red"]
        if blue.empty or red.empty:
            continue
        blue_team = blue["team"].iloc[0] if "team" in blue.columns else None
        red_team = red["team"].iloc[0] if "team" in red.columns else None
        if not isinstance(blue_team, str) or not isinstance(red_team, str):
            continue  # unknown team names can't be rated
        rows.append(
            {
                "gameid": gameid,
                "date": group["date"].max(),
                "league": str(group["league"].iloc[0]) if "league" in group.columns else "",
                "blue_team": blue_team.strip(),
                "red_team": red_team.strip(),
                # Majority vote across the side's rows, same as pairwise.py.
                "blue_win": int(round(float(blue["result"].mean()))),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["date", "gameid"]).reset_index(drop=True)


# Autocorrelation correction constant for the margin multiplier (the
# FiveThirtyEight Elo convention). Without it, a strong favourite who is
# EXPECTED to stomp gets rewarded twice -- once for winning and again for the
# blowout -- and ratings run away. Swept over [1, 10]; results were flat, so
# the conventional 2.2 is kept.
_MOV_AUTOCORR_C = 2.2


def _mov_multiplier(margin: float, winner_rating_advantage: float, scale: float) -> float:
    """Scale factor applied to K for a game won by ``margin`` (a relative gold
    advantage, see ``sources.oracles_elixir.extract_game_margins``).

    ``log1p`` keeps blowouts from dominating: the multiplier grows with the
    margin but with sharply diminishing returns, so a 40% gold lead counts
    for more than a 10% one without counting for four times as much. The
    second term damps the update when the winner was already rated well above
    its opponent, which is what stops runaway ratings.
    """
    return math.log1p(max(margin, 0.0) * scale) * (
        _MOV_AUTOCORR_C / (_MOV_AUTOCORR_C + 0.001 * winner_rating_advantage)
    )


def compute_team_elo(
    games_df: pd.DataFrame,
    k: float = DEFAULT_ELO_K,
    season_carryover: float = DEFAULT_SEASON_CARRYOVER,
    international_leagues: frozenset[str] | None = None,
    international_k_multiplier: float = 1.0,
    game_margins: dict[str, float] | None = None,
    mov_scale: float | None = None,
) -> TeamEloResult:
    """Run one chronological Elo pass over ``games_df``.

    Args:
        games_df: Cleaned per-player-game table (post ``etl.build_raw_tables``)
            with gameid/date/side/team/result columns (league optional).
        k: Elo K-factor (rating movement per game).
        season_carryover: Fraction of each rating's deviation from
            ``INITIAL_ELO`` kept across a season (calendar-year) boundary
            (see ``DEFAULT_SEASON_CARRYOVER``). ``1.0`` disables.
        international_leagues: League codes (matched case-insensitively)
            treated as international events (MSI/Worlds/EWC). Games in these
            leagues, and any game where the two teams' most-recently-seen
            leagues differ, are INTER-REGION and get ``k`` scaled by
            ``international_k_multiplier``.
        international_k_multiplier: Extra K multiplier for inter-region games.
            ``1.0`` disables (uniform K). Inter-region games are the only ones
            that calibrate Elo across regions, so they carry more information;
            measured to lift held-out international-event accuracy without
            hurting overall (see config.international_elo_k_multiplier).
        game_margins: Optional ``{gameid: relative_gold_margin}`` (see
            ``sources.oracles_elixir.extract_game_margins``). Required for
            ``mov_scale`` to have any effect; games absent from this map fall
            back to the plain binary update.
        mov_scale: Enables MARGIN-OF-VICTORY weighting. ``None`` (default)
            keeps the classic Elo where every win moves ratings equally.
            When set, K is multiplied by ``_mov_multiplier(...)`` so a
            dominant win teaches the ratings more than a narrow one, while
            the win/loss outcome itself still drives the direction. Measured
            on the walk-forward backtest (2025+2026 real pro data, 15,060
            held-out games): current-season accuracy 65.73% -> 66.16% and
            log-loss 0.6218 -> 0.6183.

            NOTE: an alternative formulation that replaced the binary outcome
            with a continuous dominance score scored *better* on the pooled
            metric (66.42%/0.6125) but was rejected -- it compressed the
            rating spread (sd 130 -> 42), which collapses the
            expected-score term toward 0.5 and so removes the
            opponent-strength correction. The ratings then degenerate toward
            "average gold margin against whoever you happened to play":
            majors held only 1 of the top 10 slots (vs 6 here), and
            cross-region accuracy DROPPED 61.9% -> 60.7%. Pooled log-loss
            hid that because inter-region games are a small slice of it.
            This K-scaling form keeps the gain and improves rating
            separation instead (sd 130 -> 143).

    Returns:
        A :class:`TeamEloResult`; see its docstring. ``per_game`` records the
        PRE-game rating of each side, so consuming it as a model feature is
        leak-free.
    """
    intl = frozenset(x.upper() for x in (international_leagues or frozenset()))
    boost = international_k_multiplier

    game_rows = _per_game_rows(games_df)
    ratings: dict[str, float] = {}
    games_played: dict[str, int] = {}
    last_league: dict[str, str] = {}
    last_played: dict[str, str] = {}
    # Each team's most-recently-seen league, used to detect a cross-region
    # matchup (the two sides come from different leagues).
    team_league: dict[str, str] = {}

    records = []
    prev_year: int | None = None
    for row in game_rows.itertuples(index=False):
        # Season boundary: regress every rating toward the mean (roster
        # turnover makes old-season ratings partially stale).
        year = pd.Timestamp(row.date).year if pd.notna(row.date) else prev_year
        if (
            season_carryover < 1.0
            and prev_year is not None
            and year is not None
            and year != prev_year
        ):
            ratings = {
                t: INITIAL_ELO + season_carryover * (r - INITIAL_ELO)
                for t, r in ratings.items()
            }
        if year is not None:
            prev_year = year

        blue, red = row.blue_team, row.red_team
        blue_elo = ratings.get(blue, INITIAL_ELO)
        red_elo = ratings.get(red, INITIAL_ELO)

        records.append(
            {
                "gameid": row.gameid,
                "blue_team": blue,
                "red_team": red,
                "blue_elo_pre": blue_elo,
                "red_elo_pre": red_elo,
            }
        )

        # Inter-region games move ratings more (they anchor cross-region Elo).
        blue_lg, red_lg = team_league.get(blue), team_league.get(red)
        inter_region = (
            boost != 1.0
            and (
                (row.league or "").upper() in intl
                or (blue_lg is not None and red_lg is not None and blue_lg != red_lg)
            )
        )
        game_k = k * boost if inter_region else k

        exp_blue = expected_score(blue_elo, red_elo)

        # Margin-of-victory: a decisive win moves ratings further than a
        # narrow one. This only ever scales the POST-game update -- the
        # feature the model consumes is the pre-game rating recorded above,
        # so it stays leak-free.
        if mov_scale is not None and game_margins:
            margin = game_margins.get(row.gameid)
            if margin is not None:
                winner_advantage = (
                    (blue_elo - red_elo) if row.blue_win == 1 else (red_elo - blue_elo)
                )
                game_k *= _mov_multiplier(float(margin), winner_advantage, mov_scale)

        delta = game_k * (row.blue_win - exp_blue)
        ratings[blue] = blue_elo + delta
        ratings[red] = red_elo - delta

        for team in (blue, red):
            games_played[team] = games_played.get(team, 0) + 1
            if row.league:
                last_league[team] = row.league
                team_league[team] = row.league
            if pd.notna(row.date):
                last_played[team] = str(pd.Timestamp(row.date).date())

    per_game = (
        pd.DataFrame(records).set_index("gameid")
        if records
        else pd.DataFrame(
            columns=["blue_team", "red_team", "blue_elo_pre", "red_elo_pre"]
        )
    )
    return TeamEloResult(
        per_game=per_game,
        ratings=ratings,
        games_played=games_played,
        last_league=last_league,
        last_played=last_played,
    )


def elo_diff_by_gameid(
    elo_result: TeamEloResult, feature_scale: float = ELO_SCALE
) -> dict[str, float]:
    """``{gameid: (blue_elo_pre - red_elo_pre) / feature_scale}`` -- the
    per-game model feature values (0 for games missing from the pass).

    ``feature_scale`` is a REGULARIZATION knob, not an Elo parameter: under
    L2, multiplying a feature by ``s`` is equivalent to dividing its
    effective penalty by ``s**2``. The synergy/matchup features leak
    in-sample and need the heavy global ``C``; the Elo feature does NOT leak
    (pre-game ratings only), so a smaller ``feature_scale`` (bigger feature)
    lets it carry real weight under the same heavy ``C``. Must match the
    ``eloScale`` shipped in ``teams.json`` so the frontend computes the
    identical feature.
    """
    if elo_result.per_game.empty:
        return {}
    diffs = (
        elo_result.per_game["blue_elo_pre"] - elo_result.per_game["red_elo_pre"]
    ) / feature_scale
    return diffs.to_dict()
