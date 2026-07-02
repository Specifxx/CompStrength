"""Fits a logistic regression predicting blue-side win probability from
five historical-data-driven predictors: the aggregate champion strength
differential, within-team pairwise synergy, cross-team same-role matchup
history, meta presence (how contested each champion is in pro drafts), and
team strength (pre-game Elo gap; see teams.py).

Features:
    ``score_diff``    = sum(blue side 5 champions' strengthScore) -
                         sum(red side 5 champions' strengthScore)
    ``synergy_diff``  = sum(synergy residual over blue's 10 unordered
                         pairs) - sum(synergy residual over red's 10
                         unordered pairs). Missing pairs contribute 0.
    ``matchup_diff``  = sum over the 5 roles of the matchup residual for
                         "{blueChampionInRole}>{redChampionInRole}" (0 if
                         that ordered pair was never observed). Already
                         directional (positive favors blue), so used as-is
                         rather than subtracted from a red-perspective
                         version.
    ``presence_diff`` = sum(blue champions' pickRate + banRate) -
                         sum(red champions' pickRate + banRate) over the
                         training window. Pro teams reveal champion
                         strength through their bans, and presence needs no
                         game *outcomes*, so it's a much lower-noise
                         strength signal than a small-sample win rate.
                         Zeroed (weight exactly 0) when
                         ``config.use_presence_feature`` is off.
    ``team_elo_diff`` = (blue team's pre-game Elo - red team's) /
                         config.elo_feature_scale. Computed by a chronological
                         Elo pass (teams.py); a game's value depends only on
                         strictly EARLIER games, so unlike synergy/matchup it
                         does not leak in-sample. 0 when teams are unknown
                         ("assume equal teams"), which is also how the site
                         behaves when the optional team inputs are blank.
    ``player_elo_diff`` = (mean of blue's 5 players' pre-game Elos - mean of
                         red's) / config.elo_feature_scale (players.py).
                         Tracks WHO is playing across roster moves. Same
                         leak-free pre-game construction as team Elo; 0 when
                         the players are unknown (no teams selected).
    ``prof_diff``     = sum over blue's 5 (player, champion) seats of the
                         player's shrunk pre-game winrate edge on that
                         champion, minus red's (players.py.proficiency) --
                         the comfort-pick signal. 0 when players are unknown.
Label: ``did blue side win`` (1/0)

We fit a 7-feature logistic regression:

    P(blue wins) = sigmoid(
        scoreDiffWeight * score_diff
        + synergyWeight * synergy_diff
        + matchupWeight * matchup_diff
        + presenceWeight * presence_diff
        + teamEloWeight * team_elo_diff
        + playerEloWeight * player_elo_diff
        + profWeight * prof_diff
        + blueSideBias
    )

using scikit-learn's ``LogisticRegression``. ``blueSideBias`` captures any
residual blue-side advantage (e.g. first pick / vision / river control)
not explained by the historical predictors above.

We use strong L2 regularization with ``C=0.001`` (far below sklearn's
default of ``C=1.0``): ``synergy_diff``/``matchup_diff`` are themselves
derived from -- and heavily correlated with -- the same underlying game
outcomes used to fit this model (most specific champion pairs/matchups only
recur a handful of times even in a large sample, so their shrunk residual is
still mostly "what this exact game's outcome was"). Left unregularized the
fit leans hard on those leaky terms (synergy/matchup weights ~0.9/0.6 vs a
~0.1 score-diff weight) and becomes badly overconfident out of sample.

This value was tuned empirically on the real Oracle's Elixir walk-forward
backtest (``backtest.py``), NOT guessed. Sweeping ``C`` over
``0.1 -> 0.01 -> 0.005 -> 0.002 -> 0.001 -> 0.0005`` on ~4,100 held-out 2026
pro games, held-out log-loss falls monotonically from ~0.77 (wildly
overconfident, worse than a coin flip) toward the ~0.69 base-rate floor:

    C=0.01   logLoss 0.720   acc 0.518
    C=0.005  logLoss 0.707   acc 0.518
    C=0.002  logLoss 0.696   acc 0.519
    C=0.001  logLoss 0.692   acc 0.531   <- chosen (first C below the
    C=0.0005 logLoss 0.690   acc 0.535      0.693 coin-flip line; retains
                                            the most champion signal)

``C=0.001`` is the knee of that curve: the largest ``C`` (hence the most
champion-driven discrimination retained) at which held-out log-loss drops
just below the coin-flip baseline and accuracy reaches the majority-class
baseline. Shrinking further (0.0005) only improves the metrics within
noise while flattening the model's response to the draft. The deeper truth
this exposes: draft alone is a genuinely weak predictor of pro outcomes, so
an honestly-calibrated model necessarily makes modest predictions near the
base rate rather than confident ones. Revisit as more real data accumulates.

Both ``blueSideBias`` and ``intercept`` keys are included per the required
output schema, but sklearn only fits one bias term. The consuming frontend
(``apps/web/lib/predict.ts``) combines them additively as
``logit = intercept + scoreDiffWeight * scoreDiff + synergyWeight *
synergyDiff + matchupWeight * matchupDiff + blueSideBias``, so to avoid
double-counting the bias we put the full fitted intercept into
``blueSideBias`` and leave ``intercept`` at ``0.0``.

If there is too little data to fit meaningfully (e.g. our small fixture,
or too few games / no variance in the label), sklearn will still return a
fit, but we print a clear warning about the small sample size and include
a caveat note in the returned metrics dict.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from compstrength_pipeline.pairwise import matchup_key, synergy_key

# Below this many training games, we consider a fit "small-sample" and add
# a warning + caveat note to the output.
SMALL_SAMPLE_THRESHOLD = 200

# Regularization strength for the 3-feature model. Empirically tuned on the
# real walk-forward backtest (see module docstring): 0.001 is the knee where
# held-out log-loss first drops below the coin-flip baseline while retaining
# the most champion-driven signal. Far below sklearn's default of 1.0 because
# the synergy/matchup features leak in-sample and must be shrunk hard.
LOGISTIC_REGRESSION_C = 0.001


@dataclass(frozen=True)
class ModelResult:
    coefficients: dict[str, float]
    metrics: dict[str, object] = field(default_factory=dict)
    training_games: int = 0


def compute_score_diff(
    champion_strength: dict[str, float],
    blue_champions: list[str],
    red_champions: list[str],
) -> float:
    """Sum of blue side champions' strengthScore minus sum of red side's.

    Champions missing from ``champion_strength`` (e.g. unrated/unknown)
    contribute 0.0.
    """
    blue_sum = sum(champion_strength.get(c, 0.0) for c in blue_champions)
    red_sum = sum(champion_strength.get(c, 0.0) for c in red_champions)
    return blue_sum - red_sum


def compute_synergy_diff(
    synergy_residuals: dict[str, float],
    blue_champions: list[str],
    red_champions: list[str],
) -> float:
    """Sum of blue side's within-team pairwise synergy residuals minus red's.

    Sums the synergy residual over all 10 unordered pairs within each
    side's 5 champions (via ``pairwise.synergy_key``); pairs absent from
    ``synergy_residuals`` (never co-occurred in the training window)
    contribute 0.
    """

    def side_sum(champions: list[str]) -> float:
        total = 0.0
        for a, b in itertools.combinations(champions, 2):
            total += synergy_residuals.get(synergy_key(a, b), 0.0)
        return total

    return side_sum(blue_champions) - side_sum(red_champions)


def compute_matchup_diff(
    matchup_residuals: dict[str, float],
    blue_champs_by_role: dict[str, str],
    red_champs_by_role: dict[str, str],
) -> float:
    """Sum over shared roles of matchup residual("{blue}>{red}") for that role.

    Already directional (positive favors blue), so used as-is rather than
    subtracted from a red-perspective version. Missing pairs (never
    observed in the training window) contribute 0. Roles present on only
    one side are skipped.
    """
    total = 0.0
    for role in set(blue_champs_by_role) & set(red_champs_by_role):
        blue_champ = blue_champs_by_role[role]
        red_champ = red_champs_by_role[role]
        total += matchup_residuals.get(matchup_key(blue_champ, red_champ), 0.0)
    return total


def compute_presence_diff(
    champion_presence: dict[str, float],
    blue_champions: list[str],
    red_champions: list[str],
) -> float:
    """Sum of blue side champions' meta presence minus red side's.

    "Presence" is pick rate + ban rate over the training window -- how
    contested a champion is in pro drafts. Pro teams reveal champion strength
    through their bans, and presence needs no game *outcomes*, so it's a much
    lower-noise strength signal than a small-sample win rate. Champions
    missing from ``champion_presence`` contribute 0.
    """
    blue_sum = sum(champion_presence.get(c, 0.0) for c in blue_champions)
    red_sum = sum(champion_presence.get(c, 0.0) for c in red_champions)
    return blue_sum - red_sum


def build_training_frame(
    games_df: pd.DataFrame,
    champion_strength: dict[str, float],
    synergy_residuals: dict[str, float] | None = None,
    matchup_residuals: dict[str, float] | None = None,
    champion_presence: dict[str, float] | None = None,
    team_elo_diffs: dict[str, float] | None = None,
    score_diff_by_game: dict[str, float] | None = None,
    player_elo_diffs: dict[str, float] | None = None,
    prof_diffs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a per-game training frame with columns [gameid, score_diff,
    synergy_diff, matchup_diff, presence_diff, team_elo_diff,
    player_elo_diff, prof_diff, blue_win].

    Args:
        games_df: Cleaned per-player-game table with exactly 10 rows per
            gameid (5 blue, 5 red), columns including gameid, side,
            position, champion, result.
        champion_strength: ``{champion: strengthScore}`` mapping, e.g.
            from ``features.compute_champion_features()["strengthScore"]``.
        synergy_residuals: ``{"ChampionA|ChampionB": residual}`` mapping,
            e.g. from ``pairwise.synergy_lookup(pairwise.compute_synergy_table(...))``.
            Defaults to an empty dict (all pairs contribute 0).
        matchup_residuals: ``{"ChampionA>ChampionB": residual}`` mapping,
            e.g. from ``pairwise.matchup_lookup(pairwise.compute_matchup_table(...))``.
            Defaults to an empty dict (all pairs contribute 0).

    Returns:
        One row per gameid with the three feature columns and the binary
        "did blue win" label.
    """
    synergy_residuals = synergy_residuals or {}
    matchup_residuals = matchup_residuals or {}
    champion_presence = champion_presence or {}
    team_elo_diffs = team_elo_diffs or {}
    score_diff_by_game = score_diff_by_game or {}
    player_elo_diffs = player_elo_diffs or {}
    prof_diffs = prof_diffs or {}

    records = []
    for gameid, group in games_df.groupby("gameid"):
        blue_rows = group[group["side"].str.lower() == "blue"]
        red_rows = group[group["side"].str.lower() == "red"]
        if blue_rows.empty or red_rows.empty:
            continue

        blue_champs = blue_rows["champion"].tolist()
        red_champs = red_rows["champion"].tolist()

        # Leave-one-game-out score diff when available (see
        # features.compute_wr_strength): the training feature must not
        # contain the game's own outcome. Prediction-time score diffs use
        # the plain full-data champion strengths (compute_score_diff).
        score_diff = score_diff_by_game.get(
            gameid,
            compute_score_diff(champion_strength, blue_champs, red_champs),
        )
        synergy_diff = compute_synergy_diff(synergy_residuals, blue_champs, red_champs)
        presence_diff = compute_presence_diff(champion_presence, blue_champs, red_champs)

        blue_by_role = dict(zip(blue_rows["position"], blue_rows["champion"]))
        red_by_role = dict(zip(red_rows["position"], red_rows["champion"]))
        matchup_diff = compute_matchup_diff(matchup_residuals, blue_by_role, red_by_role)

        blue_win = int(blue_rows["result"].iloc[0] == 1)
        records.append(
            {
                "gameid": gameid,
                "score_diff": score_diff,
                "synergy_diff": synergy_diff,
                "matchup_diff": matchup_diff,
                "presence_diff": presence_diff,
                # Pre-game Elo gap for this exact game (leak-free; see
                # teams.py). 0 when unknown -- i.e. "assume equal teams".
                "team_elo_diff": team_elo_diffs.get(gameid, 0.0),
                # Player-level pre-game features (players.py); 0 when the
                # players are unknown -- i.e. "assume equal, average players".
                "player_elo_diff": player_elo_diffs.get(gameid, 0.0),
                "prof_diff": prof_diffs.get(gameid, 0.0),
                "blue_win": blue_win,
            }
        )

    return pd.DataFrame(records)


FEATURE_COLUMNS = [
    "score_diff", "synergy_diff", "matchup_diff", "presence_diff",
    "team_elo_diff", "player_elo_diff", "prof_diff",
]


def train_model(
    games_df: pd.DataFrame,
    champion_strength: dict[str, float],
    synergy_residuals: dict[str, float] | None = None,
    matchup_residuals: dict[str, float] | None = None,
    champion_presence: dict[str, float] | None = None,
    team_elo_diffs: dict[str, float] | None = None,
    score_diff_by_game: dict[str, float] | None = None,
    player_elo_diffs: dict[str, float] | None = None,
    prof_diffs: dict[str, float] | None = None,
) -> ModelResult:
    """Fit the logistic regression model and compute evaluation metrics.

    Args:
        games_df: Cleaned per-player-game table (post ``etl.build_raw_tables``).
        champion_strength: ``{champion: strengthScore}`` mapping.
        synergy_residuals: ``{"ChampionA|ChampionB": residual}`` mapping
            (see ``pairwise.synergy_lookup``). Defaults to empty.
        matchup_residuals: ``{"ChampionA>ChampionB": residual}`` mapping
            (see ``pairwise.matchup_lookup``). Defaults to empty.
        champion_presence: ``{champion: pickRate + banRate}`` over the
            training window (see ``compute_presence_diff``). Defaults to
            empty, which zeroes the ``presence_diff`` feature so its fitted
            weight is exactly 0 (backward compatible).

    Returns:
        A :class:`ModelResult` with fitted coefficients and metrics.
        Metrics always include ``logLoss``, ``accuracy``,
        ``baselineAccuracy`` (accuracy of always predicting the majority
        class), and ``note`` (empty string unless the sample is small).
    """
    training = build_training_frame(
        games_df, champion_strength, synergy_residuals, matchup_residuals,
        champion_presence, team_elo_diffs, score_diff_by_game,
        player_elo_diffs, prof_diffs,
    )
    n = len(training)

    note = ""
    if n < SMALL_SAMPLE_THRESHOLD:
        msg = (
            f"train_model: fitting on only {n} games (< {SMALL_SAMPLE_THRESHOLD}); "
            "coefficients and metrics are likely unstable/overfit and should be "
            "treated as illustrative only until more historical data is available."
        )
        warnings.warn(msg)
        print(f"WARNING: {msg}")
        note = (
            f"Small-sample fit ({n} games): coefficients/metrics are illustrative "
            "only and will be noisy until more historical data accumulates."
        )

    if n == 0:
        # Degenerate case: nothing to fit. Return a neutral model.
        return ModelResult(
            coefficients={
                "scoreDiffWeight": 0.0,
                "synergyWeight": 0.0,
                "matchupWeight": 0.0,
                "presenceWeight": 0.0,
                "teamEloWeight": 0.0,
                "playerEloWeight": 0.0,
                "profWeight": 0.0,
                "blueSideBias": 0.0,
                "intercept": 0.0,
            },
            metrics={
                "logLoss": float("nan"),
                "accuracy": float("nan"),
                "baselineAccuracy": float("nan"),
                "note": "No training games available; model is an untrained placeholder.",
            },
            training_games=0,
        )

    X = training[FEATURE_COLUMNS].to_numpy()
    y = training["blue_win"].to_numpy()

    baseline_accuracy = max(y.mean(), 1.0 - y.mean())

    if len(np.unique(y)) < 2:
        # sklearn's LogisticRegression can't fit with a single class present.
        # Fall back to a neutral model that always predicts the observed
        # constant class, and be explicit about this in the metrics note.
        constant_prob = float(np.clip(y.mean(), 0.01, 0.99))
        coefficients = {
            "scoreDiffWeight": 0.0,
            "synergyWeight": 0.0,
            "matchupWeight": 0.0,
            "presenceWeight": 0.0,
            "teamEloWeight": 0.0,
            "playerEloWeight": 0.0,
            "profWeight": 0.0,
            "blueSideBias": float(np.log(constant_prob / (1 - constant_prob))),
            "intercept": 0.0,
        }
        preds_proba = np.full_like(y, fill_value=constant_prob, dtype=float)
        metrics = {
            "logLoss": float(log_loss(y, preds_proba, labels=[0, 1])),
            "accuracy": float(baseline_accuracy),
            "baselineAccuracy": float(baseline_accuracy),
            "note": (note + " " if note else "")
            + "Only one label class present in training data (all wins or all "
            "losses for blue side); all feature weights fixed at 0.",
        }
        return ModelResult(coefficients=coefficients, metrics=metrics, training_games=n)

    # Strong L2 (C=0.001): see module docstring for the walk-forward tuning
    # that picked this -- the synergy/matchup features leak in-sample, so
    # heavy shrinkage is what keeps held-out predictions honest.
    model = LogisticRegression(C=LOGISTIC_REGRESSION_C)
    model.fit(X, y)

    score_diff_weight = float(model.coef_[0][0])
    synergy_weight = float(model.coef_[0][1])
    matchup_weight = float(model.coef_[0][2])
    presence_weight = float(model.coef_[0][3])
    team_elo_weight = float(model.coef_[0][4])
    player_elo_weight = float(model.coef_[0][5])
    prof_weight = float(model.coef_[0][6])
    intercept = float(model.intercept_[0])

    preds_proba = model.predict_proba(X)[:, 1]
    preds = (preds_proba >= 0.5).astype(int)

    metrics = {
        "logLoss": float(log_loss(y, preds_proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, preds)),
        "baselineAccuracy": float(baseline_accuracy),
        "note": note,
    }

    # NOTE: the frontend combines these as
    #   logit = intercept + scoreDiffWeight * scoreDiff + synergyWeight *
    #           synergyDiff + matchupWeight * matchupDiff + blueSideBias
    # (see apps/web/lib/predict.ts), i.e. it *adds* all bias-like terms.
    # sklearn only fits one bias term; we put the full fitted bias into
    # `blueSideBias` (the side of the model conceptually responsible for a
    # blue-side advantage not explained by the historical predictors) and
    # leave `intercept` at 0 so the terms are not double-counted downstream.
    coefficients = {
        "scoreDiffWeight": score_diff_weight,
        "synergyWeight": synergy_weight,
        "matchupWeight": matchup_weight,
        "presenceWeight": presence_weight,
        "teamEloWeight": team_elo_weight,
        "playerEloWeight": player_elo_weight,
        "profWeight": prof_weight,
        "blueSideBias": intercept,
        "intercept": 0.0,
    }

    return ModelResult(coefficients=coefficients, metrics=metrics, training_games=n)
