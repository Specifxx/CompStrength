"""Player-level signals: per-player Elo and player-champion proficiency.

Why players on top of team Elo: rosters change (substitutes, transfers,
academy call-ups), and a team's rating smooths over WHO is actually playing.
A chronological per-player Elo pass tracks the five individuals, and a
player-champion proficiency table captures comfort picks (the literature's
single biggest booster for pro prediction). Measured on 5,935 held-out 2026
games (8-fold walk-forward over two seasons): team-Elo-only 63.6% ->
+player Elo 64.7% -> +proficiency 65.0% (65.2% at the swept K=32).

Leak-freedom mirrors ``teams.py``: ONE chronological pass; every per-game
value recorded is PRE-game (depends only on strictly earlier games), so the
same series is safe for walk-forward backtesting and for the deployed fit.

The artifact side (``build.py``) ships, per team, the CURRENT roster --
the five (player, position) seats seen in the team's most recent game --
with each player's Elo, overall winrate, and per-champion record, so the
frontend can reproduce these features exactly when the user selects teams.
Everything here is OPTIONAL at prediction time: no teams selected -> both
features are 0 ("unknown players"), reproducing the draft-only prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from compstrength_pipeline.teams import INITIAL_ELO, expected_score

DEFAULT_PLAYER_ELO_K = 32.0
DEFAULT_PLAYER_SEASON_CARRYOVER = 0.8
DEFAULT_PROF_SHRINK_GAMES = 8.0


@dataclass
class PlayerStatsResult:
    """Output of one chronological player pass.

    Attributes:
        player_elo_diff: {gameid: (mean blue player Elo - mean red player
            Elo) / elo_feature_scale} -- PRE-game, leak-free.
        prof_diff: {gameid: blue proficiency sum - red proficiency sum} --
            PRE-game player-champion comfort on the champions actually
            drafted that game, shrunk toward each player's own overall
            winrate (so it isolates champion-specific skill from general
            skill, which player Elo already carries).
        elo: {player: final Elo} ("as of today").
        wr: {player: [wins, games]} overall.
        champ: {(player, champion): [wins, games]}.
        rosters: {team: [(player, position), ...5]} from each team's most
            recent game.
    """

    player_elo_diff: dict[str, float] = field(default_factory=dict)
    prof_diff: dict[str, float] = field(default_factory=dict)
    elo: dict[str, float] = field(default_factory=dict)
    wr: dict[str, list] = field(default_factory=dict)
    champ: dict[tuple, list] = field(default_factory=dict)
    rosters: dict[str, list] = field(default_factory=dict)


def proficiency(
    player_wr: dict, player_champ: dict, player: str, champion: str, shrink: float
) -> float:
    """Shrunk champion-specific edge of ``player`` on ``champion``: the
    player-champion winrate shrunk (with ``shrink`` pseudo-games) toward the
    player's own overall winrate, minus that overall winrate. 0 for unseen
    players/pairs."""
    aw, ag = player_wr.get(player, (0.0, 0.0))
    base = (aw / ag) if ag > 0 else 0.5
    w, g = player_champ.get((player, champion), (0.0, 0.0))
    return (w + shrink * base) / (g + shrink) - base


def compute_player_stats(
    games_df: pd.DataFrame,
    elo_feature_scale: float,
    k: float = DEFAULT_PLAYER_ELO_K,
    season_carryover: float = DEFAULT_PLAYER_SEASON_CARRYOVER,
    prof_shrink: float = DEFAULT_PROF_SHRINK_GAMES,
) -> PlayerStatsResult:
    """One chronological pass computing all player-level pre-game features
    plus the as-of-now tables for the artifacts. See module docstring."""
    rows = []
    for gameid, group in games_df.groupby("gameid"):
        blue = group[group["side"].str.lower() == "blue"]
        red = group[group["side"].str.lower() == "red"]
        if len(blue) != 5 or len(red) != 5:
            continue
        bt = blue["team"].iloc[0] if "team" in blue.columns else None
        rt = red["team"].iloc[0] if "team" in red.columns else None
        rows.append(
            {
                "gameid": gameid,
                "date": group["date"].max(),
                "bteam": bt.strip() if isinstance(bt, str) else None,
                "rteam": rt.strip() if isinstance(rt, str) else None,
                "bseats": list(
                    zip(blue["player"].fillna(""), blue["position"], blue["champion"])
                ),
                "rseats": list(
                    zip(red["player"].fillna(""), red["position"], red["champion"])
                ),
                "blue_win": int(round(float(blue["result"].mean()))),
            }
        )
    rows.sort(key=lambda r: (r["date"], r["gameid"]))

    result = PlayerStatsResult()
    elo = result.elo
    wr = result.wr
    champ = result.champ
    prev_year: int | None = None

    for r in rows:
        year = r["date"].year if pd.notna(r["date"]) else prev_year
        if (
            season_carryover < 1.0
            and prev_year is not None
            and year is not None
            and year != prev_year
        ):
            for p in elo:
                elo[p] = INITIAL_ELO + season_carryover * (elo[p] - INITIAL_ELO)
        if year is not None:
            prev_year = year

        b_elos = [elo.get(p, INITIAL_ELO) for p, _, _ in r["bseats"]]
        r_elos = [elo.get(p, INITIAL_ELO) for p, _, _ in r["rseats"]]
        b_mean = sum(b_elos) / 5.0
        r_mean = sum(r_elos) / 5.0
        result.player_elo_diff[r["gameid"]] = (b_mean - r_mean) / elo_feature_scale

        b_prof = sum(
            proficiency(wr, champ, p, c, prof_shrink) for p, _, c in r["bseats"]
        )
        r_prof = sum(
            proficiency(wr, champ, p, c, prof_shrink) for p, _, c in r["rseats"]
        )
        result.prof_diff[r["gameid"]] = b_prof - r_prof

        # ---- post-game updates (never visible to this game's features) ----
        delta = k * (r["blue_win"] - expected_score(b_mean, r_mean))
        for p, _, _ in r["bseats"]:
            elo[p] = elo.get(p, INITIAL_ELO) + delta
        for p, _, _ in r["rseats"]:
            elo[p] = elo.get(p, INITIAL_ELO) - delta
        for seats, won in ((r["bseats"], r["blue_win"]), (r["rseats"], 1 - r["blue_win"])):
            for p, _, c in seats:
                cw = champ.setdefault((p, c), [0, 0])
                cw[0] += won
                cw[1] += 1
                pw = wr.setdefault(p, [0, 0])
                pw[0] += won
                pw[1] += 1
        for team, seats in ((r["bteam"], r["bseats"]), (r["rteam"], r["rseats"])):
            if team:
                result.rosters[team] = [(p, pos) for p, pos, _ in seats]

    return result
