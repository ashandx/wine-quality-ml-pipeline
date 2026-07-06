# Wine Quality: End-to-End ML Pipeline

Binary classification of red wine quality (good `quality >= 6` vs bad `quality < 6`) from
physicochemical properties. Focused on pipeline discipline
and evaluation methodology rather than chasing a single accuracy number.

## Project Overview

This notebook builds a complete ML pipeline: full EDA (imported and adapted from the prior
dataset-selection notebook), a leakage-safe `SimpleImputer + StandardScaler + model` pipeline,
a `DummyClassifier` baseline, three models compared under `StratifiedKFold` cross-validation,
a learning curve for the best model, and a final holdout evaluation. Class balance is close
to even (53.5% good / 46.5% bad); F1 is still used as the primary metric for a cleaner read
on the harder-to-separate quality boundary, with accuracy and ROC-AUC as secondary checks.

## Setup

```bash
conda create -n wine-quality-ml-pipeline python=3.12
conda activate wine-quality-ml-pipeline
pip install -r requirements.txt
```

`data/winequality-red.csv` is already included (UCI Wine Quality dataset, red wine subset).

## Run

```bash
conda activate wine-quality-ml-pipeline
jupyter notebook notebooks/wine_quality_pipeline.ipynb
```

## Key Findings

| Model | Mean CV F1 | Std F1 |
|---|---|---|
| DummyClassifier (baseline) | 0.6969 | 0.0006 |
| Logistic Regression | 0.7605 | 0.0161 |
| Gradient Boosting | 0.7869 | 0.0138 |
| Random Forest | 0.8089 | 0.0150 |

Random Forest won cross-validation and held up on the holdout set, scoring 0.81, almost
identical to its CV mean. All three real models cleared the baseline by a wide margin.
Logistic Regression came in weakest, which tracks with the EDA: individual features
correlated only weakly with quality, so a linear boundary had less to work with than a
model that captures interactions.

The learning curve showed training F1 pinned at 1.0 across every training size while
validation F1 climbed from about 0.75 to 0.81 before flattening. That gap never closed,
pointing to Random Forest's unconstrained default tree depth rather than a shortage of
data. Regularizing the model (capping `max_depth`, raising `min_samples_leaf`) is the
most promising next step, followed by feature engineering on the sulfur dioxide and
acidity measures given how weak their individual correlations were.

## Skills Demonstrated

`binary classification` `sklearn Pipeline` `data leakage prevention` `StratifiedKFold`
`DummyClassifier baseline` `logistic regression` `random forest` `learning curves`
`F1 score` `confusion matrix` `scikit-learn` `pandas` `seaborn` `matplotlib`
