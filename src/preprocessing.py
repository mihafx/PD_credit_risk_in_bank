# =============================================================================
# src/preprocessing.py
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV and data dictionary."""
    df = pd.read_csv(cfg["data"]["train_path"])
    df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns], inplace=True)
    data_dict = pd.read_excel(cfg["data"]["dictionary_path"], skiprows=1)
    return df, data_dict


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Fix invalid values and clip outliers.

    1. age <= 0  → NaN  (biologically impossible)
    2. RevolvingUtilizationOfUnsecuredLines → clip at 99th percentile
    3. DebtRatio → clip at 80th percentile
    4. Late payment columns >= 20 → NaN  (technical codes 96 / 98)
    """
    df = df.copy()
    p  = cfg["preprocessing"]

    df.loc[df["age"] <= p["age_min_valid"], "age"] = np.nan

    q = df["RevolvingUtilizationOfUnsecuredLines"].quantile(p["revolving_clip_quantile"])
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=q)

    q = df["DebtRatio"].quantile(p["debt_ratio_clip_quantile"])
    df["DebtRatio"] = df["DebtRatio"].clip(upper=q)

    for col in p["late_payment_cols"]:
        df.loc[df[col] >= p["late_payment_tech_threshold"], col] = np.nan

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create five derived features.

    TotalDebtLoad     : DebtRatio × RevolvingUtilization
    IncomePerDependent: MonthlyIncome / (1 + NumberOfDependents)
    LatePerCredit     : total delinquencies / max(open lines, 1)
    TotalLate         : sum of all delinquency counts
    LateIncomeRatio   : TotalLate / IncomePerDependent
    """
    df = df.copy()

    df["TotalDebtLoad"] = df["DebtRatio"] * df["RevolvingUtilizationOfUnsecuredLines"]

    df["IncomePerDependent"] = (
        df["MonthlyIncome"] / (1 + df["NumberOfDependents"].fillna(0))
    )

    df["LatePerCredit"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    ) / df["NumberOfOpenCreditLinesAndLoans"].replace(0, 1)

    df["TotalLate"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )

    df["LateIncomeRatio"] = df["TotalLate"] / df["IncomePerDependent"].replace(0, np.nan)

    return df


def split(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 70/30 train-test split."""
    target = cfg["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        stratify=y,
        random_state=cfg["data"]["random_state"],
    )

    print(
        f"Train: {len(X_train):>7,}  |  DR = {y_train.mean():.2%}\n"
        f"Test:  {len(X_test):>7,}  |  DR = {y_test.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test


def run_preprocessing(cfg: dict):
    """Full preprocessing pipeline. Returns (df, X_train, X_test, y_train, y_test)."""
    df, _ = load_data(cfg)
    df    = clean(df, cfg)
    df    = engineer_features(df)
    X_train, X_test, y_train, y_test = split(df, cfg)
    return df, X_train, X_test, y_train, y_test
