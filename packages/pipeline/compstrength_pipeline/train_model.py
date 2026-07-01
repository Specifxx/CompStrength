"""Fits a logistic regression predicting blue-side win probability from
three historical-data-driven predictors: the aggregate champion strength
differential, within-team pairwise synergy, and cross-team same-role
matchup history.

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
Label: ``did blue side win`` (1/0)

We fit a 3-feature logistic regression:

    P(blue wins) = sigmoid(
        scoreDiffWeight * score_diff
        + synergyWeight * synergy_diff
        + matchupWeight * matchup_diff
        + blueSideBias
    )

using scikit-learn's ``LogisticRegression``. ``blueSideBias`` captures any
residual blue-side advantage (e.g. first pick / vision / river control)
not explained by the three historical predictors above.

We use L2 regularization with ``C=0.1`` (well below sklearn's default of
``C=1.0``): ``synergy_diff``/``matchup_diff`` are themselves derived from --
and heavily correlated with -- the same underlying game outcomes used to
fit this model (most specific champion pairs/matchups only recur a handful
of times even in a large sample, so their shrunk residual is still mostly
"what this exact game's outcome was"). Empirically (see the walk-forward
backtest in ``backtest.py``), the in-sample training accuracy barely moves
as ``C`` is tuned down from 0.5 -> 0.1 -> 0.02 -- ``scoreDiffWeight`` simply
picks up the slack -- which is a strong signal that ``score_diff`` (real,
generalizable per-champion strength) explains most of the recoverable
signal, while a large chunk of what ``synergy_diff``/``matchup_diff`` add
in-sample is leakage rather than genuine held-out predictive power.
Stronger regularization trades a bit of in-sample fit for meaningfully
better out-of-sample generalization in the backtest. This is a judgment
call, not a fully tuned value; revisit as more historical (ideally real,
not synthetic) data accumulates.

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

# Regularization strength for the 3-feature model. See module docstring for
# why this is well below sklearn's default of 1.0.
LOGISTIC_REGRESSION_C = 0.1


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


def build_training_frame(
    games_df: pd.DataFrame,
    champion_strength: dict[str, float],
    synergy_residuals: dict[str, float] | None = None,
    matchup_residuals: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a per-game training frame with columns [gameid, score_diff,
    synergy_diff, matchup_diff, blue_win].

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

    records = []
    for gameid, group in games_df.groupby("gameid"):
        blue_rows = group[group["side"].str.lower() == "blue"]
        red_rows = group[group["side"].str.lower() == "red"]
        if blue_rows.empty or red_rows.empty:
            continue

        blue_champs = blue_rows["champion"].tolist()
        red_champs = red_rows["champion"].tolist()

        score_diff = compute_score_diff(champion_strength, blue_champs, red_champs)
        synergy_diff = compute_synergy_diff(synergy_residuals, blue_champs, red_champs)

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
                "blue_win": blue_win,
            }
        )

    return pd.DataFrame(records)


FEATURE_COLUMNS = ["score_diff", "synergy_diff", "matchup_diff"]


def train_model(
    games_df: pd.DataFrame,
    champion_strength: dict[str, float],
    synergy_residuals: dict[str, float] | None = None,
    matchup_residuals: dict[str, float] | None = None,
) -> ModelResult:
    """Fit the logistic regression model and compute evaluation metrics.

    Args:
        games_df: Cleaned per-player-game table (post ``etl.build_raw_tables``).
        champion_strength: ``{champion: strengthScore}`` mapping.
        synergy_residuals: ``{"ChampionA|ChampionB": residual}`` mapping
            (see ``pairwise.synergy_lookup``). Defaults to empty.
        matchup_residuals: ``{"ChampionA>ChampionB": residual}`` mapping
            (see ``pairwise.matchup_lookup``). Defaults to empty.

    Returns:
        A :class:`ModelResult` with fitted coefficients and metrics.
        Metrics always include ``logLoss``, ``accuracy``,
        ``baselineAccuracy`` (accuracy of always predicting the majority
        class), and ``note`` (empty string unless the sample is small).
    """
    training = build_training_frame(
        games_df, champion_strength, synergy_residuals, matchup_residuals
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

    # C=0.5: see module docstring for why we use slightly stronger L2
    # regularization than sklearn's default now that we've gone from 1 to
    # 3 features on a still-small pro-game sample.
    model = LogisticRegression(C=LOGISTIC_REGRESSION_C)
    model.fit(X, y)

    score_diff_weight = float(model.coef_[0][0])
    synergy_weight = float(model.coef_[0][1])
    matchup_weight = float(model.coef_[0][2])
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
        "blueSideBias": intercept,
        "intercept": 0.0,
    }

    return ModelResult(coefficients=coefficients, metrics=metrics, training_games=n)
