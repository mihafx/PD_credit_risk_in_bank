# =============================================================================
# src/binning.py — WoE binning: fit, visualise, transform
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import statsmodels.api as sm
from optbinning import OptimalBinning


def calc_gini_ks(event: np.ndarray, nonevent: np.ndarray) -> tuple[float, float]:
    """
    Compute Gini and KS from bin-level event / non-event counts.

    Gini: area between Lorenz curve and diagonal.
    KS:   max vertical separation of cumulative bad / good distributions.
    """
    te, tne = event.sum(), nonevent.sum()
    ner = nonevent / (event + nonevent)
    idx = np.argsort(ner)
    ev, ne = event[idx], nonevent[idx]

    s     = np.zeros(len(ev))
    s[1:] = 2.0 * ne[:-1].cumsum()
    gini  = 1.0 - np.dot(ev, ne + s) / (te * tne)
    ks    = np.abs((event / te).cumsum() - (nonevent / tne).cumsum()).max()
    return gini, ks


def draw_binning(table: pd.DataFrame, variable: str) -> None:
    """
    Stacked bar chart (events / non-events) + event rate line.
    Missing bin is hatched.
    """
    t = table[(table["Count"] != 0) & (table["Count (%)"] != 1)].copy()
    gini_val, ks_val = calc_gini_ks(t["Event"].values, t["Non-event"].values)

    bins, x_pos = t["Bin"].values, np.arange(len(t))
    fig, ax1 = plt.subplots()

    bars_ev  = ax1.bar(x_pos, t["Event"].values,     color="tab:red",  label="Event")
    bars_nev = ax1.bar(x_pos, t["Non-event"].values, color="tab:blue", label="Non-event",
                       bottom=t["Event"].values)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bins, rotation=45, ha="right")
    ax1.set_ylabel("Bin count", fontsize=12)
    ax1.set_title(f"{variable}  |  Gini={gini_val:.4f}  |  KS={ks_val:.4f}")

    ax2 = ax1.twinx()
    ax2.plot(x_pos, t["Event rate"].values, "o-", color="black", label="Event rate")
    ax2.set_ylabel("Event rate", fontsize=13)
    ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

    if "Missing" in bins:
        pos = np.where(bins == "Missing")[0][0]
        bars_ev[pos].set_hatch("\\")
        bars_nev[pos].set_hatch("\\")
        handles = [bars_nev[0], bars_ev[0],
                   mpatches.Patch(hatch="\\", alpha=0.1, label="Bin missing")]
        labels  = ["Non-event", "Event", "Bin missing"]
    else:
        handles = [bars_nev[0], bars_ev[0]]
        labels  = ["Non-event", "Event"]

    plt.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(1.51, 1.1), ncol=2, fontsize=12)
    plt.tight_layout()
    plt.show()


def binning_func(
    df: pd.DataFrame,
    x: str,
    y: str,
    dtype: str,
    solver: str,
    user_splits: list,
    user_splits_fixed: list,
    rule: bool,
    max_n_prebins: int,
    trend: str,
    flag_missing: bool,
    missing_place: str,
    itog: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Fit OptimalBinning and return binning table with WoE, Wilson CIs, Gini, KS.

    rule=True  : user-defined splits
    rule=False : automatic optimal binning (monotonicity + p-value constraints)
    flag_missing=True : merge missing into first or last bin (missing_place)
    itog=True  : production table; False = exploratory (all bins kept)
    """
    X, Y = df[x], df[y]

    if rule:
        optb = OptimalBinning(name=x, dtype=dtype, solver=solver,
                              user_splits=user_splits,
                              user_splits_fixed=user_splits_fixed)
    else:
        optb = OptimalBinning(name=x, dtype=dtype, solver=solver,
                              max_n_prebins=max_n_prebins,
                              min_prebin_size=0.05,
                              max_pvalue=0.05,
                              monotonic_trend=trend)
    optb.fit(X, Y)

    raw   = optb.binning_table.build()[["Bin", "Event", "Non-event", "Count"]]
    spl   = optb.splits
    z_val = raw.values
    rows  = []

    if itog and flag_missing:
        missing_idx = len(z_val) - 2
        for i in range(len(z_val) - 3):
            merge = (
                (missing_place == "missing_first" and i == 0)
                or (missing_place == "missing_last" and i == len(z_val) - 4)
            )
            if merge:
                rows.append({"Bin":  str(z_val[i, 0]) + " & missing",
                             "Bad":  z_val[i, 1] + z_val[missing_idx, 1],
                             "Good": z_val[i, 2] + z_val[missing_idx, 2],
                             "All":  z_val[i, 3] + z_val[missing_idx, 3]})
            else:
                rows.append({"Bin": str(z_val[i, 0]), "Bad": z_val[i, 1],
                             "Good": z_val[i, 2], "All": z_val[i, 3]})
    elif itog:
        for i in range(len(z_val) - 3):
            rows.append({"Bin": str(z_val[i, 0]), "Bad": z_val[i, 1],
                         "Good": z_val[i, 2], "All": z_val[i, 3]})
    else:
        for row in z_val:
            if str(row[0]) not in {"Special", "Totals"}:
                rows.append({"Bin": str(row[0]), "Bad": row[1],
                             "Good": row[2], "All": row[3]})

    z = pd.DataFrame(rows)
    z["% DR"] = z["Bad"] / z["All"] * 100
    z["WoE"]  = np.log(
        (z["Good"] / z["Good"].sum()) / (z["Bad"] / z["Bad"].sum())
    )

    ci_low, ci_up = sm.stats.proportion_confint(
        z["Bad"], z["All"], alpha=0.10, method="wilson"
    )
    z["Wilson Lower %"] = ci_low * 100
    z["Wilson Upper %"] = ci_up  * 100

    bb = z.rename(columns={"All": "Count", "Good": "Non-event", "Bad": "Event"})
    bb["Event rate"] = bb["% DR"] / 100
    bb["Count (%)"]  = bb["Count"] / bb["Count"].sum()

    t = bb[(bb["Count"] != 0) & (bb["Count (%)"] != 1)]
    g, k   = calc_gini_ks(t["Event"].values, t["Non-event"].values)
    bb["gini"] = g
    bb["ks"]   = k

    return bb[["Bin", "Event", "Non-event", "Count", "% DR", "WoE",
               "Wilson Lower %", "Wilson Upper %", "gini", "ks"]], spl


def binning_rule(
    z: pd.DataFrame,
    y: pd.Series,
    name: str,
    target: str,
    obj_list: list,
    categorical_override: set,
    chet_1: int = 20,
    trend: str = "auto",
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """
    Fit a single feature with iterative Wilson-CI monotonicity enforcement.

    Reduces bin count until adjacent Wilson CIs no longer overlap.
    Features in categorical_override skip the monotonicity check.

    Returns (binning_table, split_points, par_type)
    """
    par_type = "categorical" if name in obj_list else "numerical"
    missing_placement = (
        "missing_first" if (par_type == "numerical" and trend == "descending")
        else "missing_last"
    )

    z = z.copy()
    z[target] = y
    user_splits, user_splits_fixed = [], []
    monotone = False

    while not monotone:
        is_fixed_cat = name in categorical_override
        if is_fixed_cat:
            par_type          = "categorical"
            user_splits       = [["female"], ["male"]]
            user_splits_fixed = [True, True]
            bb, spl = binning_func(z, name, target, par_type, "cp",
                                   user_splits, user_splits_fixed,
                                   True, chet_1, "auto", False, "missing_last", True)
        else:
            try:
                bb, spl = binning_func(z, name, target, par_type, "cp",
                                       user_splits, user_splits_fixed,
                                       False, chet_1, trend, True,
                                       missing_placement, True)
            except Exception:
                bb, spl = binning_func(z, name, target, par_type, "cp",
                                       user_splits, user_splits_fixed,
                                       False, chet_1, trend, False,
                                       missing_placement, False)

        non_missing = (
            bb[bb["Bin"] != "Missing"][["Bin", "Wilson Lower %", "Wilson Upper %"]]
            .sort_values("Wilson Lower %")
            .reset_index(drop=True)
        )
        chet_1   = len(non_missing)
        monotone = True

        for i in range(1, len(non_missing)):
            if name in categorical_override:
                break
            if non_missing["Wilson Lower %"][i] < non_missing["Wilson Upper %"][i - 1]:
                monotone = False
                break

        chet_1 -= 1
        if chet_1 <= 1:
            monotone = True

    return bb, spl, par_type


def binning_apply(x, par_type: str, spl: np.ndarray, bb: pd.DataFrame) -> float:
    """
    Map a single raw value to its WoE using fitted split points and binning table.

    bb must be reset_index(drop=True) before calling.
    """
    bb = bb.reset_index(drop=True)

    if par_type == "categorical":
        is_missing = pd.isna(x) or str(x) == "nan"
        for u in range(len(bb)):
            bv = bb["Bin"][u]
            if not is_missing and ("'" + str(x) + "'" in bv):
                return bb["WoE"][u]
            if is_missing and (bv == "Missing" or "& missing" in bv):
                return bb["WoE"][u]
        x = np.nan
        for u in range(len(bb)):
            bv = bb["Bin"][u]
            if pd.isna(x) and (bv == "Missing" or "& missing" in bv):
                return bb["WoE"][u]

    elif par_type == "numerical":
        for m, threshold in enumerate(spl):
            if x < threshold:
                return bb["WoE"][m]
        is_missing = pd.isna(x) or str(x) == "nan"
        for m in range(len(bb)):
            bv = bb["Bin"][m]
            if not is_missing and ", inf)" in bv:
                return bb["WoE"][m]
            if is_missing and (bv == "Missing" or "& missing" in bv):
                return bb["WoE"][m]
