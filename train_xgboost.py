from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    if exc.name == "xgboost":
        raise SystemExit(
            "xgboost is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    raise


RANDOM_STATE = 42
DATA_PATH = Path("high_diamond_ranked_10min.csv")
OUTPUT_ROOT = Path("analysis_outputs") / "xgboost"
TARGET_COLUMN = "blueWins"
ID_COLUMN = "gameId"

TOP3_FEATURES = ["totalGoldDiff", "totalExperienceDiff", "dragonDiff"]
REDUCED_TUNED_FEATURES = [
    "wardsDestroyedDiff",
    "firstBloodDiff",
    "dragonDiff",
    "heraldDiff",
    "towersDestroyedDiff",
    "totalGoldDiff",
    "totalExperienceDiff",
    "totalJungleMinionsKilledDiff",
]


def build_difference_features(df: pd.DataFrame, feature_set: str = "all_diff") -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "wardsPlacedDiff": df["blueWardsPlaced"] - df["redWardsPlaced"],
            "wardsDestroyedDiff": df["blueWardsDestroyed"] - df["redWardsDestroyed"],
            "firstBloodDiff": df["blueFirstBlood"] - df["redFirstBlood"],
            "killDiff": df["blueKills"] - df["redKills"],
            "assistDiff": df["blueAssists"] - df["redAssists"],
            "eliteMonstersDiff": df["blueEliteMonsters"] - df["redEliteMonsters"],
            "dragonDiff": df["blueDragons"] - df["redDragons"],
            "heraldDiff": df["blueHeralds"] - df["redHeralds"],
            "towersDestroyedDiff": df["blueTowersDestroyed"]
            - df["redTowersDestroyed"],
            "totalGoldDiff": df["blueTotalGold"] - df["redTotalGold"],
            "avgLevelDiff": df["blueAvgLevel"] - df["redAvgLevel"],
            "totalExperienceDiff": df["blueTotalExperience"]
            - df["redTotalExperience"],
            "totalMinionsKilledDiff": df["blueTotalMinionsKilled"]
            - df["redTotalMinionsKilled"],
            "totalJungleMinionsKilledDiff": df["blueTotalJungleMinionsKilled"]
            - df["redTotalJungleMinionsKilled"],
            "csPerMinDiff": df["blueCSPerMin"] - df["redCSPerMin"],
            "goldPerMinDiff": df["blueGoldPerMin"] - df["redGoldPerMin"],
        }
    )

    if feature_set == "all_diff":
        return features
    if feature_set == "top3":
        return features[TOP3_FEATURES]
    if feature_set == "reduced_tuned":
        return features[REDUCED_TUNED_FEATURES]
    raise ValueError(f"Unknown feature set: {feature_set}")


def load_data(feature_set: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    X = build_difference_features(df, feature_set)
    y = df[TARGET_COLUMN]
    game_ids = df[ID_COLUMN]
    return X, y, game_ids


def make_splits(
    X: pd.DataFrame, y: pd.Series, game_ids: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    X_train_cal, X_test, y_train_cal, y_test, ids_train_cal, ids_test = train_test_split(
        X,
        y,
        game_ids,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_cal,
        y_train_cal,
        test_size=0.25,
        stratify=y_train_cal,
        random_state=RANDOM_STATE,
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test, ids_test


def build_search() -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                    subsample=1.0,
                    colsample_bytree=1.0,
                ),
            ),
        ]
    )
    param_grid = {
        "model__learning_rate": [0.1, 0.15, 0.2],
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [1],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [0.5, 1.0, 5.0],
        "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
    }
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=5,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )


def evaluate(y_true: pd.Series, y_proba: Any) -> dict[str, float]:
    y_pred = (pd.Series(y_proba, index=y_true.index) >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def calibration_error(y_true: pd.Series, y_proba: Any, n_bins: int = 10) -> float:
    frame = pd.DataFrame({"y": y_true.to_numpy(), "p": y_proba})
    frame["bin"] = pd.cut(
        frame["p"],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
    )
    grouped = frame.groupby("bin", observed=False).agg(
        n=("y", "size"),
        pred_mean=("p", "mean"),
        true_rate=("y", "mean"),
    )
    gap = (grouped["pred_mean"] - grouped["true_rate"]).abs().fillna(0)
    return float((gap * grouped["n"]).sum() / grouped["n"].sum())


def fit_sigmoid_calibrator(
    estimator: Pipeline, X_cal: pd.DataFrame, y_cal: pd.Series
) -> CalibratedClassifierCV:
    calibrator = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv="prefit",
    )
    calibrator.fit(X_cal, y_cal)
    return calibrator


def select_probability_estimator(
    best_model: Pipeline,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    calibration: str,
) -> tuple[Any, dict[str, Any]]:
    X_cal_fit, X_cal_select, y_cal_fit, y_cal_select = train_test_split(
        X_cal,
        y_cal,
        test_size=0.50,
        stratify=y_cal,
        random_state=RANDOM_STATE,
    )

    sigmoid_for_selection = fit_sigmoid_calibrator(best_model, X_cal_fit, y_cal_fit)
    validation = {
        "none": {
            **evaluate(y_cal_select, best_model.predict_proba(X_cal_select)[:, 1]),
            "ece_10bin": calibration_error(
                y_cal_select, best_model.predict_proba(X_cal_select)[:, 1]
            ),
        },
        "sigmoid": {
            **evaluate(y_cal_select, sigmoid_for_selection.predict_proba(X_cal_select)[:, 1]),
            "ece_10bin": calibration_error(
                y_cal_select,
                sigmoid_for_selection.predict_proba(X_cal_select)[:, 1],
            ),
        },
    }

    if calibration == "auto":
        selected_method = min(
            validation,
            key=lambda method: (
                validation[method]["log_loss"],
                validation[method]["brier_score"],
            ),
        )
    else:
        selected_method = calibration

    if selected_method == "sigmoid":
        probability_estimator = fit_sigmoid_calibrator(best_model, X_cal, y_cal)
    else:
        probability_estimator = best_model

    return probability_estimator, {
        "selected_method": selected_method,
        "selection_metric": "calibration_select_log_loss",
        "calibration_validation": validation,
    }


def save_calibration_curve(
    y_true: pd.Series,
    uncalibrated_proba: Any,
    selected_proba: Any,
    path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")

    for label, proba in [
        ("Uncalibrated", uncalibrated_proba),
        ("Selected probability estimator", selected_proba),
    ]:
        prob_true, prob_pred = calibration_curve(
            y_true,
            proba,
            n_bins=10,
            strategy="uniform",
        )
        plt.plot(prob_pred, prob_true, marker="o", label=label)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost probability estimator for blueWins."
    )
    parser.add_argument(
        "--feature-set",
        choices=["all_diff", "top3", "reduced_tuned"],
        default="reduced_tuned",
        help="Feature set to train on.",
    )
    parser.add_argument(
        "--calibration",
        choices=["auto", "none", "sigmoid"],
        default="auto",
        help="Probability calibration strategy. auto selects none or sigmoid on calibration data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to analysis_outputs/xgboost/<feature-set>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or OUTPUT_ROOT / args.feature_set
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, game_ids = load_data(args.feature_set)
    X_train, X_cal, X_test, y_train, y_cal, y_test, ids_test = make_splits(
        X, y, game_ids
    )

    search = build_search()
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    probability_estimator, calibration_info = select_probability_estimator(
        best_model,
        X_cal,
        y_cal,
        args.calibration,
    )

    uncalibrated_proba = best_model.predict_proba(X_test)[:, 1]
    selected_proba = probability_estimator.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "xgboost",
        "target": TARGET_COLUMN,
        "positive_class": "blueWins=1",
        "feature_strategy": args.feature_set,
        "features": list(X_train.columns),
        "split": {
            "train_cv_rows": int(len(X_train)),
            "calibration_rows": int(len(X_cal)),
            "test_rows": int(len(X_test)),
            "train_cv_positive_rate": float(y_train.mean()),
            "calibration_positive_rate": float(y_cal.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "cv": {
            "folds": 5,
            "scoring": "neg_log_loss",
            "best_score_neg_log_loss": float(search.best_score_),
            "best_params": search.best_params_,
        },
        "calibration_selection": calibration_info,
        "test_uncalibrated": {
            **evaluate(y_test, uncalibrated_proba),
            "ece_10bin": calibration_error(y_test, uncalibrated_proba),
        },
        "test_selected_probability_estimator": {
            **evaluate(y_test, selected_proba),
            "ece_10bin": calibration_error(y_test, selected_proba),
        },
    }

    predictions = pd.DataFrame(
        {
            ID_COLUMN: ids_test.to_numpy(),
            "y_true": y_test.to_numpy(),
            "prob_blue_win": selected_proba,
            "y_pred_0_5": (selected_proba >= 0.5).astype(int),
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    save_calibration_curve(
        y_test,
        uncalibrated_proba,
        selected_proba,
        output_dir / "calibration_curve.png",
        f"XGBoost Calibration Curve ({args.feature_set})",
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(probability_estimator, output_dir / "probability_estimator.joblib")
    joblib.dump(probability_estimator, output_dir / "calibrated_model.joblib")

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
