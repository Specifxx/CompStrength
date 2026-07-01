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

from dataclasses import dataclass, field

import pandas as pd

# Classic Elo parameters. ELO_K is how much one game moves a rating;
# ELO_SCALE is the logistic width (400 = chess convention). Both were
# selected by a walk-forward sweep on real 2026 data (see repo history).
DEFAULT_ELO_K = 24.0
ELO_SCALE = 400.0
INITIAL_ELO = 1500.0


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


def compute_team_elo(
    games_df: pd.DataFrame, k: float = DEFAULT_ELO_K
) -> TeamEloResult:
    """Run one chronological Elo pass over ``games_df``.

    Args:
        games_df: Cleaned per-player-game table (post ``etl.build_raw_tables``)
            with gameid/date/side/team/result columns (league optional).
        k: Elo K-factor (rating movement per game).

    Returns:
        A :class:`TeamEloResult`; see its docstring. ``per_game`` records the
        PRE-game rating of each side, so consuming it as a model feature is
        leak-free.
    """
    game_rows = _per_game_rows(games_df)
    ratings: dict[str, float] = {}
    games_played: dict[str, int] = {}
    last_league: dict[str, str] = {}
    last_played: dict[str, str] = {}

    records = []
    for row in game_rows.itertuples(index=False):
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

        exp_blue = expected_score(blue_elo, red_elo)
        delta = k * (row.blue_win - exp_blue)
        ratings[blue] = blue_elo + delta
        ratings[red] = red_elo - delta

        for team in (blue, red):
            games_played[team] = games_played.get(team, 0) + 1
            if row.league:
                last_league[team] = row.league
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
