"""Train Headwater's model, test it honestly, and make next year's forecast.

The test: train only on earlier years, then predict a year the model never saw
(default: the latest year). Compare against the PERSISTENCE BASELINE, which simply
flags the ZIP codes with the most shutoffs last year. Headwater has to at least
match that baseline, and its extra value is catching ZIPs that are newly heating up.

Usage:  python scripts/backtest.py                 (tests on the latest year)
        python scripts/backtest.py --test-year 2024
        python scripts/backtest.py --top-share 0.10 (flag the top 10% of ZIPs; default)
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from common import BACKTEST_CSV, FORECAST_CSV, OUT, PANEL_CSV, RESULTS_JSON, RESULTS_MD

NUMERIC = ["customers", "households", "population", "median_income", "poverty_rate",
           "pct_people_of_color", "cost_burden", "n_utilities"]
READABLE = {
    "log_shutoffs_now": "Shutoffs this year",
    "log_shutoffs_prev": "Shutoffs in the previous reported year",
    "change_since_prev": "Change since previous reported year",
    "customers": "Residential water customers",
    "households": "Number of households",
    "population": "Population",
    "median_income": "Median household income",
    "poverty_rate": "Poverty rate",
    "pct_people_of_color": "Share of residents who are people of color",
    "cost_burden": "Water bill as share of income",
    "n_utilities": "Number of utilities serving the ZIP",
}


def make_rows(panel, years):
    """One row per ZIP per year t, with features from year t (and the year before)
    and, when it exists, the target: shutoffs in year t+1."""
    rows = []
    by_year = {y: g.set_index("zip") for y, g in panel.groupby("year")}
    for i, t in enumerate(years):
        now = by_year[t]
        # only use "last year" when it really is the year before (2019 -> 2022 is a 3-year gap)
        prev = by_year[years[i - 1]] if i > 0 and years[i - 1] == t - 1 else None
        nxt = by_year.get(t + 1)  # only consecutive years count as a "next year"
        df = now.copy()
        df["year_t"] = t
        df["log_shutoffs_now"] = np.log1p(df["shutoffs"])
        if prev is not None:
            p = prev["shutoffs"].reindex(df.index)
            df["log_shutoffs_prev"] = np.log1p(p)
            df["change_since_prev"] = df["log_shutoffs_now"] - df["log_shutoffs_prev"]
        else:
            df["log_shutoffs_prev"] = np.nan
            df["change_since_prev"] = np.nan
        if nxt is not None:
            df["shutoffs_next"] = nxt["shutoffs"].reindex(df.index)
        else:
            df["shutoffs_next"] = np.nan
        rows.append(df.reset_index())
    return pd.concat(rows, ignore_index=True)


def feature_matrix(df, utilities):
    cols = ["log_shutoffs_now", "log_shutoffs_prev", "change_since_prev"] + \
           [c for c in NUMERIC if c in df.columns]
    X = df[cols].astype(float).copy()
    for u in utilities:  # the utility's own collection policy matters a lot
        X[f"utility: {u}"] = (df["utility"] == u).astype(float)
    return X


def usable_columns(X):
    """Skip columns that are blank or identical everywhere (e.g. cost burden before
    bills are filled in); the model can't learn from them."""
    return [c for c in X.columns if X[c].nunique(dropna=True) > 1]


def relative_change(df):
    """How much a ZIP's shutoffs changed from year t to t+1 (log scale), minus the
    average change across all ZIPs that year. Removing the statewide trend matters:
    2022 -> 2023 is the rebound after the shutoff moratorium, when almost every ZIP
    jumped. We want the model to learn which ZIPs rise FASTER THAN THE REST."""
    change = np.log1p(df["shutoffs_next"]) - np.log1p(df["shutoffs"])
    return change - change.groupby(df["target_year"]).transform("mean")


def fit(X, target):
    """The model starts from 'same as this year' and only moves a ZIP up or down
    relative to the others when the data says so."""
    model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                          min_samples_leaf=10, l2_regularization=1.0,
                                          random_state=0)
    model.fit(X, target)
    return model


def predict_next(model, X):
    """Predicted shutoffs next year, assuming statewide shutoffs stay at this year's level.
    (Rankings, which drive outreach, don't depend on that assumption.)"""
    return np.expm1(X["log_shutoffs_now"] + model.predict(X)).clip(lower=0)


def explain(model, X, X_train):
    """For each ZIP, how much each factor moved its prediction, compared with a typical
    value of that factor (the training median). Effects are on a log scale; the app
    turns them into percentages. The four utility columns are treated as one factor,
    compared with the average of all utilities."""
    base = model.predict(X)
    med = X_train.median()
    out = []
    ucols = [c for c in X.columns if c.startswith("utility: ")]
    for c in [c for c in X.columns if c not in ucols]:
        Xa = X.copy()
        Xa[c] = med[c]
        out.append(pd.DataFrame({"zip": X.index, "factor": READABLE.get(c, c),
                                 "effect": base - model.predict(Xa)}))
    if ucols:
        alts = []
        for u in ucols:
            Xa = X.copy()
            Xa[ucols] = 0.0
            Xa[u] = 1.0
            alts.append(model.predict(Xa))
        out.append(pd.DataFrame({"zip": X.index, "factor": "Water utility",
                                 "effect": base - np.mean(alts, axis=0)}))
    return pd.concat(out, ignore_index=True)


def top_k(series, k):
    return set(series.sort_values(ascending=False).index[:k])


def evaluate(test, k):
    actual = test["shutoffs_next"]
    total = actual.sum()
    model_flags = top_k(test["predicted_next"], k)
    persist_flags = top_k(test["shutoffs"], k)
    best_flags = top_k(actual, k)
    newly_hot = best_flags - persist_flags
    cap = lambda flags: float(actual.loc[list(flags)].sum() / total) if total else float("nan")
    return {
        "k": k,
        "capture_headwater": cap(model_flags),
        "capture_persistence": cap(persist_flags),
        "capture_best_possible": cap(best_flags),
        "newly_hot_zips": len(newly_hot),
        "newly_hot_caught_by_headwater": len(newly_hot & model_flags),
        "rank_corr_headwater": float(test["predicted_next"].corr(actual, method="spearman")),
        "rank_corr_persistence": float(test["shutoffs"].corr(actual, method="spearman")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-year", type=int, default=None)
    ap.add_argument("--top-share", type=float, default=0.10)
    ap.add_argument("--min-train-year", type=int, default=None,
                    help="Ignore training years before this one, e.g. 2024 to skip the 2023 post-moratorium rebound")
    ap.add_argument("--no-race", action="store_true",
                    help="Leave race out of the model's inputs (it is still shown in the app)")
    args = ap.parse_args()
    if args.no_race and "pct_people_of_color" in NUMERIC:
        NUMERIC.remove("pct_people_of_color")

    panel = pd.read_csv(PANEL_CSV, dtype={"zip": str})
    synthetic = bool(panel["synthetic"].any())
    years = sorted(panel["year"].unique().tolist())
    rows = make_rows(panel, years)
    labeled = rows.dropna(subset=["shutoffs_next"]).copy()
    labeled["target_year"] = labeled["year_t"] + 1
    target_years = sorted(labeled["target_year"].unique().tolist())
    if len(target_years) < 2:
        raise SystemExit("Need at least two pairs of back-to-back years (e.g. 2022->2023 and "
                         "2023->2024) to train on one and test on the other.")
    test_year = args.test_year or target_years[-1]
    if test_year not in target_years:
        raise SystemExit(f"Can't test on {test_year}. Testable years: {target_years}")

    if args.min_train_year:
        labeled = labeled[(labeled["target_year"] >= args.min_train_year) |
                          (labeled["target_year"] == test_year)]
    train = labeled[labeled["target_year"] < test_year].set_index("zip", drop=False)
    if train.empty:
        raise SystemExit("No training years left before the test year. Lower --min-train-year.")
    test = labeled[labeled["target_year"] == test_year].set_index("zip", drop=False)
    utilities = sorted(panel["utility"].unique().tolist())
    X_train = feature_matrix(train, utilities)
    cols = usable_columns(X_train)
    skipped = [c for c in X_train.columns if c not in cols]
    if skipped:
        print(f"Skipping columns with no usable variation: {skipped}")
    model = fit(X_train[cols], relative_change(train))
    X_test = feature_matrix(test, utilities)[cols]
    test["predicted_next"] = predict_next(model, X_test)

    k = max(5, int(round(args.top_share * len(test))))
    main_result = evaluate(test, k)
    by_k = [evaluate(test, max(3, int(round(s * len(test))))) for s in (0.05, 0.10, 0.20)]

    imp = permutation_importance(model, X_test,
                                 relative_change(test),
                                 n_repeats=20, random_state=0, scoring="neg_mean_absolute_error")
    importance = sorted(
        [(READABLE.get(c, c), float(m)) for c, m in zip(X_test.columns, imp.importances_mean)],
        key=lambda x: -x[1])

    test["flag_headwater"] = test.index.isin(top_k(test["predicted_next"], k))
    test["flag_persistence"] = test.index.isin(top_k(test["shutoffs"], k))
    test["newly_hot"] = test.index.isin(top_k(test["shutoffs_next"], k) - top_k(test["shutoffs"], k))
    OUT.mkdir(exist_ok=True)
    test[[c for c in ["zip", "city"] if c in test.columns] + ["utility", "year_t", "shutoffs", "shutoffs_next", "predicted_next",
          "flag_headwater", "flag_persistence", "newly_hot"]] \
        .rename(columns={"year_t": "from_year", "shutoffs": "shutoffs_from_year",
                         "shutoffs_next": "shutoffs_test_year"}) \
        .to_csv(BACKTEST_CSV, index=False)

    # Forecast the year after the latest one, training on everything we have.
    X_all = feature_matrix(labeled, utilities)
    cols_all = usable_columns(X_all)
    full = fit(X_all[cols_all], relative_change(labeled))
    latest = rows[rows["year_t"] == years[-1]].set_index("zip", drop=False).copy()
    X_latest = feature_matrix(latest, utilities)[cols_all]
    latest["predicted_next"] = predict_next(full, X_latest)
    explain(full, X_latest, X_all[cols_all]).to_csv(OUT / "explanations.csv", index=False)
    kf = max(5, int(round(args.top_share * len(latest))))
    latest["flag_headwater"] = latest.index.isin(top_k(latest["predicted_next"], kf))
    latest["hot_last_year"] = latest.index.isin(top_k(latest["shutoffs"], kf))
    latest["status"] = np.select(
        [latest["flag_headwater"] & ~latest["hot_last_year"],
         latest["flag_headwater"] & latest["hot_last_year"]],
        ["Emerging: newly flagged", "Persistent hot spot"], default="Not flagged")
    if "customers" in latest.columns:
        latest["predicted_per_1k_customers"] = 1000 * latest["predicted_next"] / latest["customers"]
    if "households" in latest.columns:
        latest["predicted_per_1k_households"] = 1000 * latest["predicted_next"] / latest["households"]
    keep = ["zip", "city", "utility", "shutoffs", "predicted_next", "status", "flag_headwater",
            "hot_last_year", "predicted_per_1k_customers", "predicted_per_1k_households",
            "customers", "households", "population", "median_income",
            "poverty_rate", "pct_people_of_color", "cost_burden"]
    latest[[c for c in keep if c in latest.columns]] \
        .rename(columns={"shutoffs": f"shutoffs_{years[-1]}"}) \
        .sort_values("predicted_next", ascending=False).to_csv(FORECAST_CSV, index=False)

    results = {"synthetic": synthetic, "train_years": sorted(train["target_year"].unique().tolist()),
               "test_year": test_year, "n_test_zips": len(test), "forecast_year": years[-1] + 1,
               "main": main_result, "by_k": by_k, "importance": importance[:8],
               "race_used": "pct_people_of_color" in cols_all, "flag_count": kf,
               "top_share": args.top_share}
    RESULTS_JSON.write_text(json.dumps(results, indent=2))

    m = main_result
    lines = [
        "# Headwater backtest results",
        "",
        "**STAND-IN DATA. Do not report these numbers.**" if synthetic else "",
        f"Trained on years predicting {results['train_years']}; tested on {test_year} "
        f"({len(test)} ZIP codes), a year the model never saw.",
        "",
        f"Flagging the top {k} ZIP codes ({args.top_share:.0%}):",
        f"- Headwater's flagged ZIPs held **{m['capture_headwater']:.1%}** of {test_year}'s shutoffs.",
        f"- The persistence baseline (last year's worst ZIPs) held **{m['capture_persistence']:.1%}**.",
        f"- The best possible choice of {k} ZIPs would hold {m['capture_best_possible']:.1%}.",
        f"- {m['newly_hot_zips']} ZIPs became hot spots in {test_year} without being one the year before. "
        f"Headwater flagged **{m['newly_hot_caught_by_headwater']}** of them; the baseline, by design, flagged 0.",
        f"- Rank correlation with actual shutoffs: Headwater {m['rank_corr_headwater']:.2f}, "
        f"baseline {m['rank_corr_persistence']:.2f}.",
        "",
        "| Share of ZIPs flagged | Headwater capture | Baseline capture | New hot spots caught |",
        "|---|---|---|---|",
    ]
    for r in by_k:
        lines.append(f"| top {r['k']} | {r['capture_headwater']:.1%} | {r['capture_persistence']:.1%} | "
                     f"{r['newly_hot_caught_by_headwater']} of {r['newly_hot_zips']} |")
    lines += ["", "What the model relies on most (on the test year):"]
    lines += [f"- {name}: {val:.3f}" for name, val in importance[:6]]
    RESULTS_MD.write_text("\n".join(l for l in lines if l is not None))
    print("\n".join(lines))
    print(f"\nSaved {BACKTEST_CSV}, {FORECAST_CSV}, {RESULTS_MD}")


if __name__ == "__main__":
    main()
