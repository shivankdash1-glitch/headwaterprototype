"""Combine shutoff counts, Census data and water bills into one table:
one row per ZIP code per year.

Input you create:  data/shutoffs.csv  with columns  zip, year, utility, shutoffs
                   (one row per ZIP x year x utility; see README step 3)
Inputs made by scripts:  data/acs_zcta.csv (fetch_acs.py)
Input you fill in:  data/utility_bills.csv (optional but needed for cost burden)

Usage:  python scripts/build_panel.py
"""
import sys

import pandas as pd

from common import ACS_CSV, BILLS_CSV, OUT, PANEL_CSV, SHUTOFFS_CSV, clean_zip

REQUIRED = ["zip", "year", "utility", "shutoffs"]


def fail(msg):
    sys.exit(f"\nPROBLEM: {msg}\n")


def main():
    if not SHUTOFFS_CSV.exists():
        fail(f"{SHUTOFFS_CSV} not found. Make it from the NRDC spreadsheet (README step 3), "
             "or run scripts/make_fake_data.py to test with stand-in data.")
    raw = pd.read_csv(SHUTOFFS_CSV, dtype={"zip": str})
    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        fail(f"shutoffs.csv is missing column(s) {missing}. It needs exactly: {REQUIRED}")

    raw["zip"] = clean_zip(raw["zip"])
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["shutoffs"] = pd.to_numeric(raw["shutoffs"], errors="coerce")
    raw["utility"] = raw["utility"].astype(str).str.strip()
    bad = raw[raw[["year", "shutoffs"]].isna().any(axis=1)]
    if len(bad):
        print(f"Warning: dropping {len(bad)} rows with a blank or non-numeric year/shutoffs value")
        raw = raw.drop(bad.index)
    bad_zip = ~raw["zip"].str.match(r"^0[78]\d{3}$")
    if bad_zip.any():
        print(f"Warning: dropping {bad_zip.sum()} rows whose ZIP isn't a NJ ZIP "
              f"(examples: {raw.loc[bad_zip, 'zip'].unique()[:5].tolist()})")
        raw = raw[~bad_zip]
    synthetic = "source" in raw.columns and raw["source"].astype(str).str.upper().eq("SYNTHETIC").any()

    # A ZIP can be served by more than one utility: add their shutoffs together,
    # and label the ZIP with the utility responsible for the most shutoffs.
    totals = raw.groupby(["zip", "year"], as_index=False)["shutoffs"].sum()
    by_util = raw.groupby(["zip", "year", "utility"], as_index=False)["shutoffs"].sum()
    main_util = (by_util.sort_values("shutoffs", ascending=False)
                 .drop_duplicates(["zip", "year"])[["zip", "year", "utility"]])
    n_util = by_util.groupby(["zip", "year"], as_index=False)["utility"].nunique() \
        .rename(columns={"utility": "n_utilities"})
    panel = totals.merge(main_util, on=["zip", "year"]).merge(n_util, on=["zip", "year"])

    # Residential customers (NRDC file). Only trust the per-customer rate when every
    # utility reporting in that ZIP-year also gave a customer count (Veolia gives none).
    if "customers" in raw.columns:
        raw["customers"] = pd.to_numeric(raw["customers"], errors="coerce")
        cust = raw.groupby(["zip", "year"]).agg(
            customers=("customers", lambda s: s.sum(min_count=1)),
            customers_complete=("customers", lambda s: s.notna().all())).reset_index()
        panel = panel.merge(cust, on=["zip", "year"], how="left")
        panel["shutoffs_per_1k_customers"] = (1000 * panel["shutoffs"] / panel["customers"]) \
            .where(panel["customers_complete"])
        panel.loc[~panel["customers_complete"].astype(bool), "customers"] = pd.NA

    demog_csv = ACS_CSV.parent / "nrdc_demographics.csv"
    if ACS_CSV.exists():
        acs = pd.read_csv(ACS_CSV, dtype={"zip": str})
        acs["zip"] = clean_zip(acs["zip"])
        if "synthetic" in acs.columns and acs["synthetic"].any():
            synthetic = True
        acs = acs.drop(columns=[c for c in ["synthetic"] if c in acs.columns])
        panel = panel.merge(acs, on="zip", how="left")
        no_acs = panel["households"].isna().mean()
        if no_acs > 0.1:
            print(f"Warning: {no_acs:.0%} of rows have no Census match. Check ZIP formatting.")
    elif demog_csv.exists():
        d = pd.read_csv(demog_csv, dtype={"zip": str})
        d["zip"] = clean_zip(d["zip"])
        panel = panel.merge(d.drop(columns=["city"], errors="ignore"), on="zip", how="left")
        print("Using income and race data from the NRDC file. "
              "(Optional: run scripts/fetch_acs.py to add households and poverty rate.)")
    else:
        print("Warning: no demographic data. Run convert_nrdc.py or fetch_acs.py.")

    if demog_csv.exists():  # city names, for display only
        d = pd.read_csv(demog_csv, dtype={"zip": str})
        if "city" in d.columns:
            d["zip"] = clean_zip(d["zip"])
            panel = panel.merge(d[["zip", "city"]], on="zip", how="left")

    if BILLS_CSV.exists():
        bills = pd.read_csv(BILLS_CSV)
        bills["utility"] = bills["utility"].astype(str).str.strip()
        bills["typical_annual_bill"] = pd.to_numeric(bills["typical_annual_bill"], errors="coerce")
        panel = panel.merge(bills[["utility", "typical_annual_bill"]], on="utility", how="left")
        unmatched = set(panel["utility"]) - set(bills.dropna(subset=["typical_annual_bill"])["utility"])
        if unmatched:
            print(f"Note: no bill amount yet for {sorted(unmatched)} -> cost burden left blank for those. "
                  "Fill in data/utility_bills.csv (utility names must match shutoffs.csv exactly).")
        if "median_income" in panel.columns:
            panel["cost_burden"] = panel["typical_annual_bill"] / panel["median_income"]

    if "households" in panel.columns:
        panel["shutoffs_per_1k_households"] = 1000 * panel["shutoffs"] / panel["households"]

    panel["synthetic"] = synthetic
    OUT.mkdir(exist_ok=True)
    panel = panel.sort_values(["zip", "year"])
    panel.to_csv(PANEL_CSV, index=False)

    print(f"\nSaved {PANEL_CSV}: {panel['zip'].nunique()} ZIP codes, years {sorted(panel['year'].unique().tolist())}")
    print("\nZIP codes and total shutoffs per year:")
    print(panel.groupby("year").agg(zips=("zip", "nunique"), shutoffs=("shutoffs", "sum")).to_string())
    print("\nZIP codes per utility (latest year):")
    print(panel[panel["year"] == panel["year"].max()]["utility"].value_counts().to_string())
    if synthetic:
        print("\n*** This is STAND-IN DATA. Do not report these numbers. ***")


if __name__ == "__main__":
    main()
