# =============================================================================
# src/validation.py — Model validation suite (ranking model)
# PSI · KS · Decile table · VIF · Cross-fold coefficient stability
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ─────────────────────────────────────────────────────────────────────────────
# PSI on model score
# ─────────────────────────────────────────────────────────────────────────────

def compute_score_psi(
    score_base: np.ndarray,
    score_test: np.ndarray,
    n_bins: int = 10,
    label: str = "",
) -> float:
    """
    Compute PSI between two score distributions.

    Breakpoints are fixed [0, 1] equally spaced; epsilon avoids log(0).
    PSI < 0.10 stable | < 0.25 monitor | >= 0.25 review.
    """
    eps = 1e-6
    bp  = np.linspace(0, 1, n_bins + 1)
    bp[0], bp[-1] = -np.inf, np.inf

    base_pct = np.histogram(score_base, bins=bp)[0] / len(score_base) + eps
    test_pct = np.histogram(score_test, bins=bp)[0] / len(score_test) + eps
    psi      = ((base_pct - test_pct) * np.log(base_pct / test_pct)).sum()

    labels = [f"{i/n_bins:.1f}–{(i+1)/n_bins:.1f}" for i in range(n_bins)]
    x      = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 0.2, base_pct, 0.4, label="Base (OOF)", alpha=0.8, color="#4C72B0")
    ax.bar(x + 0.2, test_pct, 0.4, label="Test",        alpha=0.8, color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(f"Score distribution: {label}  |  PSI = {psi:.4f}", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return psi


# ─────────────────────────────────────────────────────────────────────────────
# KS statistic
# ─────────────────────────────────────────────────────────────────────────────

def compute_ks(y_true: pd.Series, y_score: np.ndarray, label: str = "") -> float:
    """
    KS statistic with CDF separation plot.
    KS > 0.30 acceptable | > 0.40 good for credit scoring.
    """
    df = pd.DataFrame({"score": y_score, "target": np.array(y_true)})
    df = df.sort_values("score").reset_index(drop=True)

    n_bad, n_good = df["target"].sum(), len(df) - df["target"].sum()
    df["cum_bad"]  = df["target"].cumsum() / n_bad
    df["cum_good"] = (1 - df["target"]).cumsum() / n_good
    df["ks"]       = (df["cum_bad"] - df["cum_good"]).abs()

    ks_val   = df["ks"].max()
    ks_score = df.loc[df["ks"].idxmax(), "score"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["score"], df["cum_bad"],  label="Bads CDF",  color="#D62728", lw=2)
    ax.plot(df["score"], df["cum_good"], label="Goods CDF", color="#1F77B4", lw=2)
    ax.axvline(ks_score, color="gray", linestyle="--", lw=1.2,
               label=f"KS threshold = {ks_score:.3f}")
    ax.annotate(f"KS = {ks_val:.4f}", xy=(ks_score, 0.5),
                xytext=(ks_score + 0.03, 0.45), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_title(f"KS Plot — {label}", fontsize=12)
    ax.set_xlabel("Model score")
    ax.set_ylabel("Cumulative share")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.show()

    print(f"  KS = {ks_val:.4f}  |  threshold = {ks_score:.4f}")
    return ks_val


# ─────────────────────────────────────────────────────────────────────────────
# Decile table
# ─────────────────────────────────────────────────────────────────────────────

def decile_table(
    y_true: pd.Series,
    y_score: np.ndarray,
    label: str = "",
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Decile analysis: DR, lift, cumulative bad capture per score bucket.
    Decile 1 = highest score (lowest predicted risk).
    """
    df = pd.DataFrame({"score": y_score, "target": np.array(y_true)})
    df["decile"] = pd.qcut(
        df["score"].rank(method="first"),
        q=n_deciles, labels=range(n_deciles, 0, -1),
    ).astype(int)

    total_bad = df["target"].sum()
    tbl = (
        df.groupby("decile", sort=False)
        .agg(n_obs=("target", "count"), n_bad=("target", "sum"),
             score_min=("score", "min"), score_max=("score", "max"))
        .sort_index(ascending=False).reset_index()
    )
    tbl["dr"]          = tbl["n_bad"] / tbl["n_obs"]
    tbl["pct_obs"]     = tbl["n_obs"] / tbl["n_obs"].sum()
    tbl["cum_bad_pct"] = tbl["n_bad"].cumsum() / total_bad
    tbl["lift"]        = tbl["dr"] / (total_bad / len(df))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(tbl["decile"].astype(str), tbl["dr"], color="#4C72B0", alpha=0.85)
    ax.axhline(total_bad / len(df), color="red", linestyle="--", lw=1.5,
               label=f"Overall DR = {total_bad/len(df):.2%}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(f"Default Rate by Decile — {label}", fontsize=12)
    ax.set_xlabel("Decile (1 = best score)")
    ax.set_ylabel("Default Rate")
    ax.legend()
    plt.tight_layout()
    plt.show()

    fmt = {"dr": "{:.2%}", "pct_obs": "{:.2%}", "cum_bad_pct": "{:.2%}",
           "lift": "{:.2f}", "score_min": "{:.4f}", "score_max": "{:.4f}"}
    try:
        display(tbl.style.format(fmt).set_caption(f"Decile Table — {label}"))
    except NameError:
        print(tbl.to_string())
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# VIF aggregation across folds
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_vif(vif_per_fold: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate per-fold VIF into mean / std / min / max summary table.
    Also shows per-fold breakdown to spot fold-level outliers.
    VIF < 5 ok | 5–10 moderate | > 10 severe.
    """
    vif_all = pd.concat(vif_per_fold, ignore_index=True)

    vif_summary = (
        vif_all.groupby("feature")["VIF"]
        .agg(VIF_mean="mean", VIF_std="std", VIF_min="min", VIF_max="max")
        .reset_index()
        .sort_values("VIF_mean", ascending=False)
        .reset_index(drop=True)
    )

    vif_pivot = vif_all.pivot(index="feature", columns="fold", values="VIF")
    vif_pivot.columns = [f"fold_{c}" for c in vif_pivot.columns]
    vif_pivot = vif_pivot.reset_index()

    def _color(val):
        if isinstance(val, float):
            if val > 10: return "background-color: #FFCCCC"
            if val > 5:  return "background-color: #FFF3CC"
        return ""

    fold_cols = [c for c in vif_pivot.columns if c.startswith("fold_")]

    try:
        display(
            vif_summary.style
            .applymap(_color, subset=["VIF_mean"])
            .format({"VIF_mean": "{:.2f}", "VIF_std": "{:.2f}",
                     "VIF_min":  "{:.2f}", "VIF_max": "{:.2f}"})
            .set_caption("VIF — mean across folds")
        )
        display(
            vif_pivot.style
            .applymap(_color, subset=fold_cols)
            .format({c: "{:.2f}" for c in fold_cols})
            .set_caption("VIF — per fold breakdown")
        )
    except NameError:
        print(vif_summary.to_string())

    high = vif_summary[vif_summary["VIF_mean"] > 10]
    if not high.empty:
        print(f"\n[WARNING] {len(high)} feature(s) with mean VIF > 10:")
        print(high[["feature", "VIF_mean", "VIF_max"]].to_string(index=False))
    else:
        print("\nAll features VIF <= 10 — no multicollinearity detected.")

    return vif_summary


# ─────────────────────────────────────────────────────────────────────────────
# Cross-fold coefficient stability
# ─────────────────────────────────────────────────────────────────────────────

def coefficient_stability(
    coef_per_fold: list[pd.Series],
    n_folds: int,
) -> pd.DataFrame:
    """
    Align coefficients from all folds and compute stability metrics.

    Features not selected in a fold → NaN.
    Flags sign flips (red) and features absent in some folds (yellow).
    """
    coef_df = pd.concat(coef_per_fold, axis=1).T.reset_index(drop=True)
    coef_df.index = [f"fold_{i}" for i in range(n_folds)]

    stab      = coef_df.T.copy()
    stab.columns = [f"fold_{i}" for i in range(n_folds)]
    fold_cols = stab.columns.tolist()

    stab["coef_mean"] = stab[fold_cols].mean(axis=1)
    stab["coef_std"]  = stab[fold_cols].std(axis=1)
    stab["coef_min"]  = stab[fold_cols].min(axis=1)
    stab["coef_max"]  = stab[fold_cols].max(axis=1)
    stab["n_folds_selected"] = stab[fold_cols].notna().sum(axis=1)

    def _sign_stable(row):
        vals = row[fold_cols].dropna()
        return len(vals) < 2 or (vals > 0).all() or (vals < 0).all()

    stab["sign_stable"] = stab.apply(_sign_stable, axis=1)
    stab = stab.sort_values("coef_mean").reset_index().rename(columns={"index": "feature"})

    colors = ["#4C72B0" if s else "#D62728" for s in stab["sign_stable"]]
    y_pos  = np.arange(len(stab))

    fig, ax = plt.subplots(figsize=(10, max(4, len(stab) * 0.55)))
    ax.barh(y_pos, stab["coef_mean"],
            xerr=[stab["coef_mean"] - stab["coef_min"],
                  stab["coef_max"]  - stab["coef_mean"]],
            align="center", color=colors, alpha=0.8,
            error_kw={"ecolor": "#333333", "capsize": 4, "lw": 1.5})
    ax.axvline(0, color="black", linestyle="--", lw=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stab["feature"].tolist(), fontsize=9)
    ax.set_xlabel("Coefficient (mean across folds, error bars = min/max)")
    ax.set_title(
        f"Cross-Fold Coefficient Stability  ({n_folds} folds)\n"
        "Blue = sign stable  |  Red = sign flip",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

    fmt = {c: "{:.4f}" for c in ["coef_mean", "coef_std", "coef_min", "coef_max"] + fold_cols}

    try:
        def _c(v): return "" if v else "background-color: #FFCCCC"
        def _s(v): return "background-color: #FFF3CC" if v < n_folds else ""
        display(
            stab.style
            .applymap(_c, subset=["sign_stable"])
            .applymap(_s, subset=["n_folds_selected"])
            .format(fmt, na_rep="—")
            .format({"n_folds_selected": "{:.0f}"})
            .set_caption(f"Coefficient Stability Across {n_folds} Folds")
        )
    except NameError:
        print(stab.to_string())

    unstable = stab[~stab["sign_stable"]]
    if not unstable.empty:
        print(f"[WARNING] {len(unstable)} feature(s) with sign flip across folds")
    else:
        print(f"All coefficients sign-stable across {n_folds} folds.")

    return stab


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    y_train: pd.Series,
    y_test: pd.Series,
    oof_mean: np.ndarray,
    final_test_pred: np.ndarray,
    oof_level1: np.ndarray,
    test_level1: np.ndarray,
    fold_val_indices: list,
    vif_per_fold: list,
    coef_per_fold: list,
    cfg: dict,
) -> None:
    """
    Run full validation suite for a ranking model:
      1. PSI (overall + per fold)
      2. KS (test + OOF)
      3. Decile table (test + OOF)
      4. VIF aggregated across folds
      5. Cross-fold coefficient stability
    """
    n_folds   = cfg["cv"]["n_folds"]
    n_bins    = cfg["validation"]["n_score_bins"]
    n_deciles = cfg["validation"]["n_deciles"]

    print("\n" + "=" * 60 + "\n  VALIDATION REPORT\n" + "=" * 60)

    print("\n─── 1 · PSI on Score ───────────────────────────────────────")
    psi = compute_score_psi(oof_mean, final_test_pred,
                            n_bins=n_bins, label="Train (OOF) vs Test")
    print(f"PSI overall: {psi:.4f}")
    for i, val_idx in enumerate(fold_val_indices):
        psi_f = compute_score_psi(oof_level1[val_idx, i], test_level1[:, i],
                                  n_bins=n_bins, label=f"Fold {i+1} OOF vs Test")
        print(f"  Fold {i+1} PSI: {psi_f:.4f}")

    print("\n─── 2 · KS Statistic ───────────────────────────────────────")
    compute_ks(y_test,  final_test_pred, label="Test (ensemble)")
    compute_ks(y_train, oof_mean,        label="Train OOF (mean)")

    print("\n─── 3 · Decile Table ───────────────────────────────────────")
    decile_table(y_test,  final_test_pred, label="Test (ensemble)",  n_deciles=n_deciles)
    decile_table(y_train, oof_mean,        label="Train OOF (mean)", n_deciles=n_deciles)

    print("\n─── 4 · VIF ────────────────────────────────────────────────")
    aggregate_vif(vif_per_fold)

    print("\n─── 5 · Cross-Fold Coefficient Stability ───────────────────")
    coefficient_stability(coef_per_fold, n_folds)
