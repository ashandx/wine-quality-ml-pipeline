# Wine Quality: End-to-End ML Pipeline

Binary classification of red wine quality (good `quality >= 6` vs bad `quality < 6`) from
physicochemical properties. Week 6 Milestone 2 project — the focus is pipeline discipline
and evaluation methodology, not chasing a single accuracy number.

## Project Overview

This notebook builds a complete ML pipeline: full EDA (imported and adapted from the prior
dataset-selection notebook), a leakage-safe `SimpleImputer + StandardScaler + model` pipeline,
a `DummyClassifier` baseline, three models compared under `StratifiedKFold` cross-validation,
a learning curve for the best model, and a final holdout evaluation. Class balance is close
to even (53.5% good / 46.5% bad); F1 is still used as the primary metric for a cleaner read
on the harder-to-separate quality boundary, with accuracy and ROC-AUC as secondary checks.

## Setup

```bash
conda create -n wine-quality-ml-pipeline python=3.11
conda activate wine-quality-ml-pipeline
pip install -r requirements.txt
```

`data/winequality-red.csv` is already included (UCI Wine Quality dataset, red wine subset).

## Run

```bash
conda activate wine-quality-ml-pipeline
jupyter notebook notebooks/milestone2_wine_quality.ipynb
```

## Key Findings

_TODO — fill in once the notebook is complete: baseline F1, best model, CV F1 vs holdout F1,
what the learning curve showed._

## Skills Demonstrated

`binary classification` `sklearn Pipeline` `data leakage prevention` `StratifiedKFold`
`DummyClassifier baseline` `logistic regression` `random forest` `learning curves`
`F1 score` `confusion matrix` `scikit-learn` `pandas` `seaborn` `matplotlib`
