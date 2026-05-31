# 202601_ML_term_project

# League of Legends Win Probability Estimation

This project trains probability estimators for predicting whether the blue team
wins a League of Legends ranked game using 10-minute match statistics.

The final task is binary probability estimation:

```text
target: blueWins
positive class: blueWins = 1
```

The main models are:

- Logistic Regression
- XGBoost

Both models are tuned with 5-fold cross-validation on the training split using
negative log loss, because the final output is intended to be used as a
probability estimate.

## Dataset

Dataset:

```text
League of Legends Diamond Ranked Games (10 min)
```

Source:

```text
https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min/
```

Place the CSV file in the project root:

```text
high_diamond_ranked_10min.csv
```

The script excludes `gameId` from training and uses `blueWins` as the target.

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

Tested with Python 3.12.

## Feature Sets

The scripts support three feature sets.

| Feature Set | Description |
|---|---|
| `top3` | Uses only `totalGoldDiff`, `totalExperienceDiff`, `dragonDiff` |
| `reduced_tuned` | Uses 8 selected blue-minus-red difference features |
| `all_diff` | Uses all 16 blue-minus-red difference features |

`reduced_tuned` features:

```text
wardsDestroyedDiff
firstBloodDiff
dragonDiff
heraldDiff
towersDestroyedDiff
totalGoldDiff
totalExperienceDiff
totalJungleMinionsKilledDiff
```

## Training

Logistic Regression:

```powershell
python train_logistic_regression.py --feature-set top3 --calibration auto
python train_logistic_regression.py --feature-set reduced_tuned --calibration auto
python train_logistic_regression.py --feature-set all_diff --calibration auto
```

XGBoost:

```powershell
python train_xgboost.py --feature-set top3 --calibration auto
python train_xgboost.py --feature-set reduced_tuned --calibration auto
python train_xgboost.py --feature-set all_diff --calibration auto
```

Calibration options:

| Option | Meaning |
|---|---|
| `auto` | Selects `none` or `sigmoid` using the calibration split |
| `none` | Uses the model's raw predicted probabilities |
| `sigmoid` | Applies sigmoid probability calibration |

## Output Files

Outputs are saved under:

```text
analysis_outputs/logistic_regression/<feature-set>/
analysis_outputs/xgboost/<feature-set>/
```

Each run creates:

| File | Description |
|---|---|
| `metrics.json` | Split info, best hyperparameters, test metrics |
| `test_predictions.csv` | Test-set labels and predicted probabilities |
| `calibration_curve.png` | Calibration curve plot |
| `probability_estimator.joblib` | Final probability estimator |
| `calibrated_model.joblib` | Same final estimator, kept for compatibility |

## Data Split

The data is split as follows:

```text
Train + CV: 60%
Calibration: 20%
Test: 20%
```

The training split is used for 5-fold cross-validation hyperparameter tuning.
The calibration split is used only for selecting/applying probability
calibration. The test split is used only for final evaluation.

## Latest Results

Latest test-set results from the current experiments:

| Feature Set | Model | Calibration | ROC-AUC | Log Loss | Brier | ECE | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| `top3` | Logistic Regression | sigmoid | 0.8059 | 0.5317 | 0.1797 | 0.0207 | 0.7212 |
| `reduced_tuned` | Logistic Regression | none | 0.8065 | 0.5312 | 0.1795 | 0.0245 | 0.7186 |
| `top3` | XGBoost | none | 0.8053 | 0.5344 | 0.1810 | 0.0342 | 0.7262 |
| `reduced_tuned` | XGBoost | none | 0.8057 | 0.5334 | 0.1807 | 0.0339 | 0.7267 |

For probability estimation, log loss, Brier score, and calibration error are
more important than accuracy. Logistic Regression with `reduced_tuned` features
had the best probability metrics in the current split, while Logistic
Regression with `top3` features gave similar performance with a simpler feature
set.

## Reproducibility

Random seed is fixed to:

```text
42
```

The scripts use stratified splits to preserve the class ratio across train,
calibration, and test sets.
