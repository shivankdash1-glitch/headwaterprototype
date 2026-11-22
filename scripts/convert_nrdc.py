"""Turn the NRDC file downloaded by fetch_nrdc.py into data/shutoffs.csv
(one row per ZIP x year x utility) and data/nrdc_demographics.csv.

The NRDC file has one row per ZIP per year, with separate columns per utility:
  tdisc_<UTIL>    = residential disconnections (shutoffs)
  rescust_<UTIL>  = residential customers (the same snapshot for every year)
for UTIL = AQUA, MWC (Middlesex), NJAW (NJ American Water), VEO (Veolia).

A blank tdisc cell means no report for that utility/ZIP/year (zeros are written
in explicitly), so blanks are skipped, not counted as zero.

Usage:  python scripts/convert_nrdc.py                       (uses data/raw/*.csv)
        python scripts/convert_nrdc.py path/to/nrdc_file.csv
"""
import sys
from pathlib import Path

import pandas as pd

from common import DATA, SHUTOFFS_CSV, clean_zip

UTILITIES = {"AQUA": "Aqua New Jersey", "MWC": "Middlesex Water",
             "NJAW": "New Jersey American Water", "VEO": "Veolia New Jersey"}
DEMOG_CSV = DATA / "nrdc_demographics.csv"


def find_input():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for f in sorted((DATA / "raw").glob("*.csv")):
        cols = pd.read_csv(f, nrows=0).columns
        if any(c.startswith("tdisc_") for c in cols):
            return f
    sys.exit("No NRDC file found. Run fetch_nrdc.py first, or pass the file path.")


def main():
    src = find_input()
    print("Reading", src)
    df = pd.read_csv(src, dtype=str)
    zip_col = "zipcode" if "zipcode" in df.columns else "ZIP_CODE"
    df["zip"] = clean_zip(df[zip_col])
    df = df[df["year"].str.strip().str.lower() != "all"].copy()   # "All" = sum of years; skip
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    rows = []
    for code, name in UTILITIES.items():
        t = pd.to_numeric(df.get(f"tdisc_{code}"), errors="coerce")
        c = pd.to_numeric(df.get(f"rescust_{code}"), errors="coerce")
        part = pd.DataFrame({"zip": df["zip"], "year": df["year"], "utility": name,
                             "shutoffs": t, "customers": c.where(c > 0)})
        rows.append(part[part["shutoffs"].notna()])
    long = pd.concat(rows, ignore_index=True).sort_values(["zip", "year", "utility"])
    long["source"] = "NRDC"
    long.to_csv(SHUTOFFS_CSV, index=False)

    # Consistency check against NRDC's own all-utility totals
    all_tot = pd.to_numeric(df["tdisc_all"], errors="coerce").groupby(df["year"]).sum()
    ours = long.groupby("year")["shutoffs"].sum()
    print(f"\nSaved {SHUTOFFS_CSV}: {len(long)} rows, {long['zip'].nunique()} ZIP codes")
    print("\nShutoffs per year (ours vs NRDC's tdisc_all total):")
    for y in ours.index:
        flag = "" if abs(ours[y] - all_tot.get(y, 0)) < 1 else "   <-- MISMATCH, check"
        print(f"  {y}: {int(ours[y]):>7,}  vs {int(all_tot.get(y, 0)):>7,}{flag}")
    print("\nZIP codes reporting, by utility and year:")
    print(long.pivot_table(index="utility", columns="year", values="zip", aggfunc="nunique").fillna(0)
          .astype(int).to_string())
    no_cust = long["customers"].isna().mean()
    print(f"\nRows without a customer count: {no_cust:.0%} (Veolia reports none; "
          "per-customer rates are left blank there)")

    # NRDC's file already includes Census demographics for each ZIP
    d = df.drop_duplicates("zip")
    demog = pd.DataFrame({
        "zip": d["zip"],
        "city": d.get("city", pd.Series(pd.NA, index=d.index)).str.strip(),
        "population": pd.to_numeric(d.get("TotalPopul"), errors="coerce"),
        "median_income": pd.to_numeric(d.get("MHI_1"), errors="coerce"),
        "pct_people_of_color": pd.to_numeric(d.get("P_ppl_of_color"), errors="coerce") / 100,
    })
    for c in ["population", "median_income"]:
        demog.loc[demog[c] < 0, c] = pd.NA   # Census "no data" codes
    demog.to_csv(DEMOG_CSV, index=False)
    print(f"Saved {DEMOG_CSV} (population, median income, share people of color)")


if __name__ == "__main__":
    main()
