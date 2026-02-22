# PD model of credit scoring 

Logistic regression pipeline for Probability of Default (PD) estimation designed for IFRS 9 / internal credit risk reserving contexts. The model is based on binning rules and WoE transformations. The quality of the model is confirmed for validation.
Logistic regression pipeline for credit scoring on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset. Built as a research notebook with a clean, importable module structure for GitHub publication.

**Target:** `SeriousDlqin2yrs` — 90+ days past due within 2 years . Binary taret (0 or 1)

## Key Results

| Metric | fold 1 | fold 2 | fold 3 |
|--------|--------|--------|--------|
| Gini (train) | 0.6990 | 0.7089 | 0.7060 |
| Gini (valid) | 0.7083 | 0.6933 | 0.6992 |
| Gini (test) | 0.7024 | 0.7027 | 0.7037 |

| Metric | Value |
|--------|-------|
| Ensemble Gini (Test) | 0.7042 |
| OOF Gini (mean of 3 folds) | 0.7001 |


---

## Pipeline Overview

```
cs-training.csv
      │
      ▼
EDA  (pandas-profiling or standard stats)
      │
      ▼
Cleaning & Outlier Treatment
      │
      ▼
Feature Engineering  (5 derived features)
      │
      ▼
Stratified 70 / 30 split
      │
      ▼
┌─────────────────── 3-Fold CV ───────────────────────┐
│  WoE Binning  →  PSI filter                         │
│  RFE  →  Backward stepwise (p-value / corr / Gini)  │
│  GridSearchCV  →  best C                            │
│  sm.Logit  (warm start from sklearn coefs)          │
│  OOF predictions  +  test predictions               │
└─────────────────────────────────────────────────────┘
      │
      ▼
Ensemble  (unweighted mean across folds)
      │
      ▼
Validation  (PSI · KS · Decile table · VIF · Coef stability)
```

---

## Repository Structure

```
├── README.md             
├── config.yaml            # all hyperparameters & constants
├── run.py                 # entry point — runs full pipeline
├── requirements.txt
└── src/
    ├── preprocessing.py   # load → clean → engineer → split
    ├── binning.py         # WoE binning, monotonicity, WoE transform
    └── validation.py      # PSI, KS, decile, VIF, coef stability 
└── data/
    ├── cs-training.csv         # raw dataset (download from Kaggle)
    └── Data Dictionary.xls
```

---

## Quick Start

1) 
Use pd_model_credit_risk_v3.ipynb for understanding main logic

2)
```bash
# 1. Clone and install dependencies
git clone https://github.com/mihafx/credit-scoring.git
cd credit-scoring
pip install -r requirements.txt

# 2. Place data files in data/
#    cs-training.csv and Data Dictionary.xls

# 3. Run full pipeline
python run.py

# 4. Or with a custom config
python run.py --config config.yaml
```

---

## Configuration

All constants live in `config.yaml` — no need to edit source files:

```yaml
cv:
  n_folds: 3

binning:
  psi_threshold: 0.05   # features with PSI >= this are excluded per fold

feature_selection:
  max_features: 16      # RFE target before backward stepwise
  pvalue_threshold: 0.05

model:
  C_grid: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
  inner_cv_folds: 3
  scoring: roc_auc
```

---
## Feature Engineering — WoE Binning

All features are transformed to **Weight of Evidence (WoE)** using [`optbinning`](https://github.com/guillermo-navas-palencia/optbinning). Monotonicity is enforced by iteratively reducing bin count until **Wilson CI** intervals of adjacent bins stop overlapping.

Features are excluded per fold if:
- PSI of default rate distribution ≥ `psi_threshold` (between train and val splits)
- Optimal binning collapses everything into a single `(-inf, inf)` bin


## Feature Selection

Two-stage per fold, run on the fold's train split only:

**Stage 1 — RFE** reduces to `MAX_FEATURES` using `LogisticRegression(lbfgs)`.

**Stage 2 — Backward stepwise** iteratively removes features by:
- p-value > 0.05 (statsmodels Logit) → remove worst
- Pairwise correlation > 0.50 or name similarity > 0.65 → remove weaker by univariate Gini
- NaN p-value (numerical instability) → remove lowest univariate Gini feature

---

## Model Training

Per fold: find best regularisation strength `C` via `GridSearchCV` (inner 3-fold CV, ROC-AUC), then refit `statsmodels.Logit` without regularisation using sklearn coefficients as `start_params`:

```
GridSearchCV(C ∈ [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], scoring=roc_auc)
    │
    └──► best sklearn coefs  →  start_params
                                      │
                                      ▼
                              sm.Logit.fit(bfgs)
                                      │
                                      └──► summary() with p-values & CIs
```

This gives valid standard errors and p-values in `summary()` (not available with regularised Logit) while benefiting from a warm start for faster, more stable convergence.

---

## Ensemble

```python
oof_mean        = oof_level1.mean(axis=1)    # final train-side score
final_test_pred = test_level1.mean(axis=1)   # final test score
```

Unweighted mean across folds. A stacking meta-model was tested but rejected: OOF and test predictions have different probability ranges due to train size asymmetry across folds, causing PSI ≈ 10 and inverted decile tables.

---

## Validation Suite

| Check | What it measures | Applied to |
|-------|-----------------|------------|
| PSI on score | Score distribution shift | OOF vs Test (overall + per fold) |
| KS statistic | Max bad/good CDF separation | Test + OOF |
| Decile table | DR concentration, lift, cumulative bad capture | Test + OOF |
| VIF | Multicollinearity | Per fold → mean/std/min/max |
| Cross-fold coef stability | Sign consistency, selection frequency | All folds |

---

## Requirements

```bash
pip install -r requirements.txt
```

---



