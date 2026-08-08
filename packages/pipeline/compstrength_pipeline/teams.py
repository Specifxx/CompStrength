"""Team-strength ratings: our own sequential Elo, an early-game (gold-at-15)
rating, and Riot's published Global Power Rankings.

Three ratings, because they measure different things off different evidence:

- :func:`compute_team_elo` -- classic sequential Elo on WIN/LOSS, with its
  K-factor scaled by margin of victory (game length) and by whether the game
  was inter-region.
- :func:`compute_econ_ratings` -- the same Elo shape, but rating teams on the
  CONTINUOUS margin they build by 15 minutes (gold lead) instead of on one bit
  of win/loss. Converges much faster, which is why it adds real signal on top
  of the Elo pass over the very same games.
- :func:`compute_gpr_diffs` -- a join onto Riot's OWN published rating
  (``sources/lolesports_gpr.py``), which is built from inputs this pipeline
  has no access to and is therefore an independent second opinion rather than
  a re-derivation.

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

All three follow the same rule, which is what makes them safe in the
walk-forward backtest: the value recorded for a game is the rating as it
stood BEFORE that game (for GPR, the newest snapshot Riot published strictly
before it), so a game's feature only ever depends on strictly earlier
information.

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
# Between-season regression toward the mean: at each calendar-year (season)
# boundary in the chronological pass, every rating keeps this fraction of its
# deviation from INITIAL_ELO. Rosters turn over between seasons, so a team's
# old-season rating is real but partially stale information. Measured on
# 15,973 real games (2026 pre-game Elo predictions): carryover 0.7 improves
# the Elo-only signal from 63.86%/0.6357 to 64.11%/0.6326 vs full carryover.
# 1.0 disables (old behavior).
DEFAULT_SEASON_CARRYOVER = 0.7
# Bounds on the margin-of-victory K multiplier (see ``compute_team_elo``'s
# ``mov_alpha``). Keeps a single blowout from swinging a rating wildly and
# stops a marathon game from zeroing out a legitimate result.
MOV_CLAMP = (0.4, 1.8)


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


# Columns that are optional (not every source publishes them) and numeric.
# Coerced ONCE up front rather than per game group -- with ~17k groups the
# per-group coercion cost is the difference between a fast pass and a slow one.
_OPTIONAL_NUMERIC_COLUMNS = ("gamelength", "golddiffat15")


def _optional_float(series: pd.Series | None) -> float:
    """First finite value of ``series`` as a float, else NaN (missing columns
    and all-NA columns both degrade to "no data" rather than raising)."""
    if series is None:
        return float("nan")
    values = series.dropna()
    return float(values.iloc[0]) if len(values) else float("nan")


def _per_game_rows(games_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 10-row-per-game player table into one row per game with
    (date, league, blue_team, red_team, blue_win, gamelength, blue_gd15),
    sorted chronologically (date, then gameid for a deterministic same-day
    order).

    ``gamelength`` (seconds) and ``blue_gd15`` (blue's gold lead at 15
    minutes, the sum of its five players' ``golddiffat15``) are NaN when the
    source doesn't publish them; the ratings below treat NaN as "no
    refinement available" rather than as zero.
    """
    present = [c for c in _OPTIONAL_NUMERIC_COLUMNS if c in games_df.columns]
    if present:
        games_df = games_df.assign(
            **{c: pd.to_numeric(games_df[c], errors="coerce") for c in present}
        )

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
        gd15 = float("nan")
        if "golddiffat15" in blue.columns:
            values = blue["golddiffat15"]
            # All five lanes must be present: a partial sum would understate
            # the team's actual lead and quietly bias the rating.
            gd15 = float(values.sum()) if values.notna().all() else float("nan")
        rows.append(
            {
                "gameid": gameid,
                "date": group["date"].max(),
                "league": str(group["league"].iloc[0]) if "league" in group.columns else "",
                "blue_team": blue_team.strip(),
                "red_team": red_team.strip(),
                # Majority vote across the side's rows, same as pairwise.py.
                "blue_win": int(round(float(blue["result"].mean()))),
                "gamelength": _optional_float(
                    group["gamelength"] if "gamelength" in group.columns else None
                ),
                "blue_gd15": gd15,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["date", "gameid"]).reset_index(drop=True)


def compute_team_elo(
    games_df: pd.DataFrame,
    k: float = DEFAULT_ELO_K,
    season_carryover: float = DEFAULT_SEASON_CARRYOVER,
    international_leagues: frozenset[str] | None = None,
    international_k_multiplier: float = 1.0,
    mov_alpha: float = 0.0,
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
        mov_alpha: Margin-of-victory sensitivity. Plain Elo throws away HOW a
            game was won, but a 25-minute win is much stronger evidence of
            strength than a 40-minute one. Each game's K is multiplied by
            ``1 + mov_alpha * (median_length - length) / stdev_length``
            (clamped to ``MOV_CLAMP``), so decisive games move ratings more
            and grindy ones less. ``0.0`` disables (uniform K, the old
            behaviour); games with no ``gamelength`` always use plain K.

    Returns:
        A :class:`TeamEloResult`; see its docstring. ``per_game`` records the
        PRE-game rating of each side, so consuming it as a model feature is
        leak-free.
    """
    intl = frozenset(x.upper() for x in (international_leagues or frozenset()))
    boost = international_k_multiplier

    game_rows = _per_game_rows(games_df)
    # Margin-of-victory normalization constants, measured over THIS dataset
    # (game lengths drift across patches, so a hardcoded median would go
    # stale). Both are pure summary statistics of the window, not per-game
    # outcomes, so they introduce no leakage into any individual prediction.
    mov_median = mov_stdev = float("nan")
    if mov_alpha and not game_rows.empty and "gamelength" in game_rows.columns:
        lengths = pd.to_numeric(game_rows["gamelength"], errors="coerce").dropna()
        if len(lengths) > 1 and lengths.std() > 0:
            mov_median, mov_stdev = float(lengths.median()), float(lengths.std())
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

        # Margin of victory: scale K by how decisive the game was. Clamped so
        # one 20-minute stomp can't dominate a rating, and skipped entirely
        # when the source has no game length.
        if mov_alpha and mov_stdev == mov_stdev:  # NaN-safe "is a real number"
            length = getattr(row, "gamelength", float("nan"))
            if length == length:
                game_k *= float(
                    min(
                        MOV_CLAMP[1],
                        max(MOV_CLAMP[0], 1 + mov_alpha * (mov_median - length) / mov_stdev),
                    )
                )

        exp_blue = expected_score(blue_elo, red_elo)
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


@dataclass
class EconResult:
    """Output of one chronological early-game ("lane economy") rating pass.

    Attributes:
        per_game: ``{gameid: (blue_rating - red_rating) / feature_scale}``
            using PRE-game ratings only -- leak-free exactly like the Elo pass.
        ratings: ``{team: final rating}`` after the full pass, in gold units
            (0 = an exactly average team at 15 minutes).
    """

    per_game: dict[str, float] = field(default_factory=dict)
    ratings: dict[str, float] = field(default_factory=dict)


def compute_econ_ratings(
    games_df: pd.DataFrame,
    k: float,
    season_carryover: float = DEFAULT_SEASON_CARRYOVER,
    feature_scale: float = 750.0,
) -> EconResult:
    """Rate teams on the GOLD LEAD THEY BUILD BY 15 MINUTES, opponent-adjusted.

    Win/loss is one bit per game; a team's gold lead at 15 minutes is a
    continuous, much lower-variance read on how good they actually are, and
    it is settled before most of the swing (and the throws) happen. Rating on
    it therefore converges far faster than Elo on the same games -- which is
    why this measurably improves held-out prediction even though the Elo pass
    above already consumes the very same matches.

    The update is Elo-shaped but in gold units instead of probability: a
    team's rating is its expected gold lead against an average opponent, the
    prediction for a game is ``blue_rating - red_rating``, and both sides move
    by ``k * (actual - predicted)``. Subtracting the prediction is what makes
    it OPPONENT-ADJUSTED -- +2k gold against the league's worst team is
    evidence of much less than +2k against its best. Ratings regress toward
    average at each season boundary like team Elo (same roster-turnover
    logic).

    Args:
        games_df: Cleaned per-player-game table with a ``golddiffat15``
            column. Games missing it are skipped for UPDATES but still get a
            (pre-game) feature value, so a source without the column simply
            leaves every rating at 0.
        k: Learning rate on the gold residual.
        season_carryover: Fraction of each rating kept across a season
            boundary. ``1.0`` disables.
        feature_scale: Divisor turning a rating gap into the model feature
            (a regularization knob, like ``elo_feature_scale``).

    Returns:
        An :class:`EconResult`; see its docstring.
    """
    game_rows = _per_game_rows(games_df)
    if game_rows.empty or "blue_gd15" not in game_rows.columns:
        return EconResult()

    ratings: dict[str, float] = {}
    per_game: dict[str, float] = {}
    prev_year: int | None = None
    for row in game_rows.itertuples(index=False):
        year = pd.Timestamp(row.date).year if pd.notna(row.date) else prev_year
        if (
            season_carryover < 1.0
            and prev_year is not None
            and year is not None
            and year != prev_year
        ):
            ratings = {t: season_carryover * r for t, r in ratings.items()}
        if year is not None:
            prev_year = year

        blue_rating = ratings.get(row.blue_team, 0.0)
        red_rating = ratings.get(row.red_team, 0.0)
        per_game[row.gameid] = (blue_rating - red_rating) / feature_scale

        actual = row.blue_gd15
        if actual != actual:  # NaN: no gold data for this game
            continue
        delta = k * (float(actual) - (blue_rating - red_rating))
        ratings[row.blue_team] = blue_rating + delta
        ratings[row.red_team] = red_rating - delta

    return EconResult(per_game=per_game, ratings=ratings)


@dataclass
class GprResult:
    """Riot Global Power Rankings joined onto this dataset's team names.

    Attributes:
        per_game: ``{gameid: (blue_gpr - red_gpr) / feature_scale}`` using, for
            each game, the newest GPR snapshot computed STRICTLY BEFORE that
            game's date -- leak-free exactly like the pre-game Elo above. 0
            when either side has no GPR rating (GPR covers tier-1 only).
        ratings: ``{Oracle's Elixir team name: latest GPR rating}`` -- "as of
            today", for the frontend's optional team inputs.
        covered_games: How many games got a non-zero (both-sides-known) value.
        name_map: ``{GPR team name: Oracle's Elixir team name}`` actually used.
    """

    per_game: dict[str, float] = field(default_factory=dict)
    ratings: dict[str, float] = field(default_factory=dict)
    covered_games: int = 0
    name_map: dict[str, str] = field(default_factory=dict)


def compute_gpr_diffs(
    games_df: pd.DataFrame,
    history,
    feature_scale: float,
    field_name: str = "gpr",
) -> GprResult:
    """Join Riot's published GPR history onto ``games_df`` as a per-game feature.

    Riot's Global Power Rankings rate team strength using inputs this pipeline
    has no access to (in-game execution metrics, an explicit regional-strength
    score), so they are a genuinely independent second opinion next to
    :func:`compute_team_elo` rather than a re-derivation of the same win/loss
    signal -- which is what makes them worth a feature of their own.

    Args:
        games_df: Cleaned per-player-game table (post ``etl.build_raw_tables``).
        history: ``[(ts, {gpr_team: [gprScore, elo, rank]}), ...]`` from
            ``sources.lolesports_gpr.load_gpr_history``. Empty -> empty result
            (the feature is then zero everywhere and its fitted weight is
            exactly 0).
        feature_scale: Divisor turning a GPR gap into the model feature. A
            regularization knob, like ``elo_feature_scale``; must match the
            ``gprScale`` shipped in ``teams.json``.
        field_name: ``"gpr"`` (published headline score) or ``"elo"`` (GPR's
            underlying raw Elo).

    Returns:
        A :class:`GprResult`; see its docstring.
    """
    from compstrength_pipeline.sources import lolesports_gpr

    if not history or games_df.empty:
        return GprResult()

    game_rows = _per_game_rows(games_df)
    if game_rows.empty:
        return GprResult()

    oe_names = {
        str(t).strip() for t in games_df["team"].dropna().unique() if str(t).strip()
    }
    name_map = lolesports_gpr.build_team_name_map(
        {team for _, teams in history for team in teams}, oe_names
    )
    oe_to_gpr = {oe: gpr_name for gpr_name, oe in name_map.items()}

    per_game: dict[str, float] = {}
    covered = 0
    # Walk games chronologically and advance a cursor through the (already
    # sorted) snapshot list, so the whole join is a single linear pass rather
    # than a per-game search back through history.
    cursor = 0
    current: dict[str, list[float]] = {}
    index = 1 if field_name == "elo" else 0
    for row in game_rows.itertuples(index=False):
        game_date = pd.Timestamp(row.date)
        if game_date.tzinfo is None:
            game_date = game_date.tz_localize("UTC")
        while cursor < len(history) and history[cursor][0] < game_date:
            current = history[cursor][1]
            cursor += 1
        blue_key = oe_to_gpr.get(row.blue_team)
        red_key = oe_to_gpr.get(row.red_team)
        blue = current.get(blue_key) if blue_key else None
        red = current.get(red_key) if red_key else None
        if blue is None or red is None:
            per_game[row.gameid] = 0.0
            continue
        per_game[row.gameid] = (float(blue[index]) - float(red[index])) / feature_scale
        covered += 1

    latest = history[-1][1] if history else {}
    ratings = {
        oe: float(latest[gpr_name][index])
        for oe, gpr_name in oe_to_gpr.items()
        if gpr_name in latest
    }
    return GprResult(
        per_game=per_game,
        ratings=ratings,
        covered_games=covered,
        name_map=name_map,
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
