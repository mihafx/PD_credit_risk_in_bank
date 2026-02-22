# =============================================================================
# run.py — Credit Scoring Pipeline Entry Point
# =============================================================================
#
# Usage:
#   python run.py                        # uses default config.yaml
#   python run.py --config config.yaml
#
# For interactive use, import stages directly from src/:
#   from src.preprocessing import run_preprocessing
#   from src.binning import binning_rule, binning_apply
#   from src.validation import run_validation
# =============================================================================

import argparse
import difflib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.preprocessing import run_preprocessing
from src.binning import binning_rule, binning_apply
from src.validation import run_validation

warnings.filterwarnings("ignore")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_cv(df, X_train, X_test, y_train, y_test, cfg) -> dict:
    """
    Stratified K-Fold CV pipeline.

    Per fold:
      1. WoE binning + PSI filter
      2. RFE + backward stepwise feature selection
      3. GridSearchCV for best regularisation C
      4. sm.Logit with warm start from sklearn coefficients
      5. OOF + test predictions, VIF, coefficient storage
    """
    TARGET       = cfg["data"]["target"]
    N_FOLDS      = cfg["cv"]["n_folds"]
    RANDOM_STATE = cfg["cv"]["random_state"]
    PSI_THR      = cfg["binning"]["psi_threshold"]
    MAX_FEAT     = cfg["feature_selection"]["max_features"]
    MAX_ITER     = cfg["feature_selection"]["max_iter"]
    PVAL_THR     = cfg["feature_selection"]["pvalue_threshold"]
    CORR_THR     = cfg["feature_selection"]["correlation_threshold"]
    NAME_THR     = cfg["feature_selection"]["name_similarity_threshold"]
    TREND_MAP    = cfg["binning"]["trend_map"]
    CAT_OVR      = set(cfg["binning"]["categorical_override"])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    X_train_m   = X_train.copy()
    oof_level1  = np.zeros((len(X_train_m), N_FOLDS))
    test_level1 = np.zeros((len(X_test), N_FOLDS))

    df_appl             = pd.DataFrame(index=["gini_train", "gini_valid", "gini_test", "gini_all"])
    table_binning       = pd.DataFrame()
    log_exclude         = []
    vif_per_fold        = []
    coef_per_fold       = []
    fold_val_indices    = []
    best_params_records = []

    obj_list = [c for c in X_train_m.columns if X_train_m[c].dtype == object]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_m, y_train)):
        print(f"\n{'='*60}")
        print(f"  Fold {fold + 1} / {N_FOLDS}  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"{'='*60}")

        X_test_f = X_test.copy()
        df_all_f = df.copy()

        X_tr  = X_train_m.iloc[train_idx].copy()
        X_val = X_train_m.iloc[val_idx].copy()
        y_tr  = y_train.iloc[train_idx].copy()
        y_val = y_train.iloc[val_idx].copy()

        # ── Step 1: Binning + PSI filter ─────────────────────────────────────
        valid_cols = []

        for name, trend in TREND_MAP.items():
            print(f"  Binning: {name}")
            bb, spl, par_type = binning_rule(
                z=X_tr.copy(), y=y_tr.copy(),
                name=name, target=TARGET,
                obj_list=obj_list,
                categorical_override=CAT_OVR,
                chet_1=cfg["binning"]["max_n_prebins"],
                trend=trend,
            )
            bb["name"] = name
            table_binning = pd.concat([table_binning, bb], ignore_index=True)

            for df_part in (X_tr, X_val, X_test_f, df_all_f):
                df_part[name] = df_part[name].apply(
                    lambda x, _par=par_type, _spl=spl, _bb=bb: binning_apply(x, _par, _spl, _bb)
                )

            dist_tr  = (pd.concat([X_tr,  y_tr],  axis=1)
                        .groupby(name).agg(dr=(TARGET, lambda x: x.sum() / len(x))))
            dist_val = (pd.concat([X_val, y_val], axis=1)
                        .groupby(name).agg(dr=(TARGET, lambda x: x.sum() / len(x))))

            has_single = any(b == "(-inf, inf)" for b in bb["Bin"])
            psi = ((dist_tr["dr"] - dist_val["dr"])
                   * np.log(dist_tr["dr"] / dist_val["dr"])).sum()

            if not has_single and psi < PSI_THR:
                valid_cols.append(name)
            elif psi >= PSI_THR:
                print(f"    [SKIP] PSI={psi:.4f} >= {PSI_THR}: {name}")
            else:
                print(f"    [SKIP] Could not bin: {name}")

        print(f"\n  Binning done — {len(valid_cols)} features passed")

        # ── Step 2: Scale for feature selection ──────────────────────────────
        scaler = StandardScaler().fit(X_tr[valid_cols])
        X_v    = pd.DataFrame(scaler.transform(X_tr[valid_cols]), columns=valid_cols)
        X_v["intersept"] = 1
        Y_v    = y_tr.reset_index(drop=True)

        # ── Step 3: RFE + backward stepwise ──────────────────────────────────
        num_f, cols_remove, iterr = MAX_FEAT, [], 0
        cols1 = list(X_v.columns)

        while len(cols1) > 0 and num_f > 3:
            if len(cols1) <= 4 or len(cols1) < num_f:
                num_f -= 1
                cols_remove = []

            cols1 = [c for c in X_v.columns if c not in cols_remove]

            rfe = RFE(LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE,
                                         max_iter=1500),
                      n_features_to_select=num_f, verbose=0)
            rfe.fit(X_v[cols1], Y_v)
            cols1 = list(X_v[cols1].columns[rfe.support_])
            if "intersept" not in cols1:
                cols1.append("intersept")

            clf_sm = sm.Logit(Y_v.astype(float), X_v[cols1].astype(float)).fit(
                method="bfgs", disp=False)
            p    = pd.Series(clf_sm.pvalues.values, index=cols1)
            pmax = p.max()
            worst = p.idxmax()

            # Check correlated / name-similar pairs
            drop_c, g1, g2, i2, k2 = None, 0, 0, None, None
            for a in [c for c in cols1 if c != "intersept"]:
                for b in [c for c in cols1 if c not in {a, "intersept"}]:
                    sim  = difflib.SequenceMatcher(None, a, b).ratio() >= NAME_THR
                    corr = abs(X_v[[a, b]].corr().iloc[1, 0]) > CORR_THR
                    if sim or corr:
                        for col, flag in ((a, True), (b, False)):
                            lm = sm.Logit(Y_v.astype(float),
                                          X_v[col].astype(float)).fit(method="bfgs", disp=False)
                            g  = 2 * roc_auc_score(Y_v, lm.predict(X_v[col])) - 1
                            if flag: g1, i2 = g, a
                            else:    g2, k2 = g, b
                        drop_c = k2 if g1 >= g2 else i2
                        break
                if drop_c:
                    break

            if pmax > PVAL_THR:
                cols_remove.append(worst)
                log_exclude.append(f"fold={fold} n={num_f} drop={worst} p={pmax:.4f}")
            elif drop_c:
                cols_remove.append(drop_c)
                log_exclude.append(f"fold={fold} n={num_f} drop={drop_c} corr/sim")
            elif pd.isna(pmax):
                ginis = [2*roc_auc_score(Y_v, sm.Logit(Y_v.astype(float),
                          X_v[c].astype(float)).fit(method="bfgs",disp=False).predict(X_v[c]))-1
                         if c != "intersept" else 1.0 for c in cols1]
                weakest = cols1[int(np.argmin(ginis))]
                cols_remove.append(weakest)
                log_exclude.append(f"fold={fold} n={num_f} drop={weakest} NaN p-value")
            else:
                break

            iterr += 1
            if iterr >= MAX_ITER:
                print("  [WARNING] Max iterations reached.")
                break

        cols1 = [c for c in set(cols1) if c != "intersept"]
        print(f"  Selected ({len(cols1)}): {cols1}")

        # ── Step 4: Final scaling on selected features ────────────────────────
        scaler   = StandardScaler().fit(X_tr[cols1])
        X_tr     = pd.DataFrame(scaler.transform(X_tr[cols1]),     columns=cols1)
        X_val    = pd.DataFrame(scaler.transform(X_val[cols1]),    columns=cols1)
        X_test_f = pd.DataFrame(scaler.transform(X_test_f[cols1]), columns=cols1)
        df_all_f = pd.DataFrame(scaler.transform(df_all_f[cols1]), columns=cols1)

        for dp in (X_tr, X_val, X_test_f, df_all_f):
            dp["intersept"] = 1
        cols1 = X_tr.columns

        # ── Step 5: GridSearchCV → warm start → sm.Logit ─────────────────────
        no_int = [c for c in cols1 if c != "intersept"]
        gs = GridSearchCV(
            LogisticRegression(penalty="l2", solver=cfg["model"]["solver"],
                               max_iter=cfg["model"]["max_iter"],
                               random_state=RANDOM_STATE),
            param_grid={"C": cfg["model"]["C_grid"]},
            scoring=cfg["model"]["scoring"],
            cv=StratifiedKFold(n_splits=cfg["model"]["inner_cv_folds"],
                               shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1, refit=True,
        )
        gs.fit(X_tr[no_int].astype(float), y_tr.reset_index(drop=True).astype(int))

        best_C = gs.best_params_["C"]
        print(f"  Best C={best_C}  |  inner CV AUC={gs.best_score_:.4f}")
        best_params_records.append({"fold": fold, "best_C": best_C, "cv_auc": gs.best_score_})

        # sklearn coefs → start_params for sm.Logit  [features..., intercept]
        start_params = np.append(gs.best_estimator_.coef_[0],
                                 gs.best_estimator_.intercept_[0])

        clf = sm.Logit(y_tr.reset_index(drop=True).astype(float),
                       X_tr[cols1].astype(float)).fit(
            method="bfgs", start_params=start_params, disp=False)
        print(clf.summary())

        # ── Step 6: Predictions & Gini ────────────────────────────────────────
        y_pr_tr  = clf.predict(X_tr[cols1])
        y_pr_val = clf.predict(X_val[cols1])
        y_pr_te  = clf.predict(X_test_f[cols1])
        y_pr_all = clf.predict(df_all_f[cols1])

        g_tr  = 2 * roc_auc_score(y_tr,      y_pr_tr)  - 1
        g_val = 2 * roc_auc_score(y_val,      y_pr_val) - 1
        g_te  = 2 * roc_auc_score(y_test,     y_pr_te)  - 1
        g_all = 2 * roc_auc_score(df[TARGET], y_pr_all) - 1

        print(f"  Train={g_tr:.4f} | Valid={g_val:.4f} | Test={g_te:.4f} | All={g_all:.4f}")
        df_appl[f"model_{fold}"] = [g_tr, g_val, g_te, g_all]

        # ── Step 7: Store artefacts ───────────────────────────────────────────
        oof_level1[val_idx, fold] = y_pr_val
        test_level1[:, fold]      = y_pr_te
        fold_val_indices.append(val_idx)

        vif_cols = [c for c in X_tr.columns if c != "intersept"]
        vif_per_fold.append(pd.DataFrame({
            "feature": vif_cols,
            "VIF":  [variance_inflation_factor(X_tr[vif_cols].values, i)
                     for i in range(len(vif_cols))],
            "fold": fold,
        }))
        coef_per_fold.append(
            pd.Series(clf.params.values, index=X_tr[cols1].columns, name=f"fold_{fold}")
        )

    # ── Ensemble ──────────────────────────────────────────────────────────────
    oof_mean        = oof_level1.mean(axis=1)
    final_test_pred = test_level1.mean(axis=1)
    print(f"\nOOF Gini:  {2 * roc_auc_score(y_train, oof_mean) - 1:.4f}")
    print(f"Test Gini: {2 * roc_auc_score(y_test, final_test_pred) - 1:.4f}")

    return dict(
        df_appl=df_appl,
        oof_level1=oof_level1, test_level1=test_level1,
        oof_mean=oof_mean, final_test_pred=final_test_pred,
        fold_val_indices=fold_val_indices,
        vif_per_fold=vif_per_fold, coef_per_fold=coef_per_fold,
        log_exclude=log_exclude, best_params_records=best_params_records,
        table_binning=table_binning,
    )


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    global df  # needed inside run_cv for df_all predictions

    df, X_train, X_test, y_train, y_test = run_preprocessing(cfg)
    results = run_cv(df, X_train, X_test, y_train, y_test, cfg)

    print("\nGini summary:")
    print(results["df_appl"].to_string())

    print("\nFeature exclusion log:")
    for entry in results["log_exclude"]:
        print(" ", entry)

    run_validation(
        y_train=y_train, y_test=y_test,
        oof_mean=results["oof_mean"],
        final_test_pred=results["final_test_pred"],
        oof_level1=results["oof_level1"],
        test_level1=results["test_level1"],
        fold_val_indices=results["fold_val_indices"],
        vif_per_fold=results["vif_per_fold"],
        coef_per_fold=results["coef_per_fold"],
        cfg=cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Scoring Pipeline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
