"""Fits a simple logistic regression predicting blue-side win probability
from the aggregate strength-score differential between the two teams.

Feature: ``score_diff = sum(blue side 5 champions' strengthScore) -
sum(red side 5 champions' strengthScore)``
Label: ``did blue side win`` (1/0)

We fit a 1-feature logistic regression:

    P(blue wins) = sigmoid(scoreDiffWeight * score_diff + blueSideBias)

using scikit-learn's ``LogisticRegression``. ``blueSideBias`` captures any
residual blue-side advantage (e.g. first pick / vision / river control)
not explained by champion strength alone.

Both ``blueSideBias`` and ``intercept`` keys are included per the required
output schema, but since this is a single-feature model there is only one
fitted bias term. The consuming frontend (``apps/web/lib/predict.ts``)
combines them additively as
``logit = intercept + scoreDiffWeight * scoreDiff + blueSideBias``, so to
avoid double-counting the bias we put the full fitted intercept into
``blueSideBias`` and leave ``intercept`` at ``0.0``.

If there is too little data to fit meaningfully (e.g. our small fixture,
or too few games / no variance in the label), sklearn will still return a
fit, but we print a clear warning about the small sample size and include
a caveat note in the returned metrics dict.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Below this many training games, we consider a fit "small-sample" and add
# a warning + caveat note to the output.
SMALL_SAMPLE_THRESHOLD = 200


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


def build_training_frame(
    games_df: pd.DataFrame, champion_strength: dict[str, float]
) -> pd.DataFrame:
    """Build a per-game training frame with columns [gameid, score_diff, blue_win].

    Args:
        games_df: Cleaned per-player-game table with exactly 10 rows per
            gameid (5 blue, 5 red), columns including gameid, side,
            champion, result.
        champion_strength: ``{champion: strengthScore}`` mapping, e.g.
            from ``features.compute_champion_features()["strengthScore"]``.

    Returns:
        One row per gameid with the score differential feature and the
        binary "did blue win" label.
    """
    records = []
    for gameid, group in games_df.groupby("gameid"):
        blue_rows = group[group["side"].str.lower() == "blue"]
        red_rows = group[group["side"].str.lower() == "red"]
        if blue_rows.empty or red_rows.empty:
            continue

        score_diff = compute_score_diff(
            champion_strength,
            blue_rows["champion"].tolist(),
            red_rows["champion"].tolist(),
        )
        blue_win = int(blue_rows["result"].iloc[0] == 1)
        records.append({"gameid": gameid, "score_diff": score_diff, "blue_win": blue_win})

    return pd.DataFrame(records)


def train_model(
    games_df: pd.DataFrame, champion_strength: dict[str, float]
) -> ModelResult:
    """Fit the logistic regression model and compute evaluation metrics.

    Args:
        games_df: Cleaned per-player-game table (post ``etl.build_raw_tables``).
        champion_strength: ``{champion: strengthScore}`` mapping.

    Returns:
        A :class:`ModelResult` with fitted coefficients and metrics.
        Metrics always include ``logLoss``, ``accuracy``,
        ``baselineAccuracy`` (accuracy of always predicting the majority
        class), and ``note`` (empty string unless the sample is small).
    """
    training = build_training_frame(games_df, champion_strength)
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
            coefficients={"scoreDiffWeight": 0.0, "blueSideBias": 0.0, "intercept": 0.0},
            metrics={
                "logLoss": float("nan"),
                "accuracy": float("nan"),
                "baselineAccuracy": float("nan"),
                "note": "No training games available; model is an untrained placeholder.",
            },
            training_games=0,
        )

    X = training[["score_diff"]].to_numpy()
    y = training["blue_win"].to_numpy()

    baseline_accuracy = max(y.mean(), 1.0 - y.mean())

    if len(np.unique(y)) < 2:
        # sklearn's LogisticRegression can't fit with a single class present.
        # Fall back to a neutral model that always predicts the observed
        # constant class, and be explicit about this in the metrics note.
        constant_prob = float(np.clip(y.mean(), 0.01, 0.99))
        coefficients = {
            "scoreDiffWeight": 0.0,
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
            "losses for blue side); scoreDiffWeight fixed at 0.",
        }
        return ModelResult(coefficients=coefficients, metrics=metrics, training_games=n)

    model = LogisticRegression()
    model.fit(X, y)

    score_diff_weight = float(model.coef_[0][0])
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
    #   logit = intercept + scoreDiffWeight * scoreDiff + blueSideBias
    # (see apps/web/lib/predict.ts), i.e. it *adds* both bias-like terms.
    # Since this is a single-feature logistic regression, sklearn only
    # fits one bias term; we put the full fitted bias into `blueSideBias`
    # (the side of the model conceptually responsible for a blue-side
    # advantage not explained by champion strength) and leave `intercept`
    # at 0 so the two terms are not double-counted downstream.
    coefficients = {
        "scoreDiffWeight": score_diff_weight,
        "blueSideBias": intercept,
        "intercept": 0.0,
    }

    return ModelResult(coefficients=coefficients, metrics=metrics, training_games=n)
