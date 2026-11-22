"""Download Census ACS 5-year data for every ZIP code area (ZCTA). Run once.

Pulls: households, median household income, poverty rate, share of residents
who are people of color. No API key is needed for one request like this.

Usage:  python scripts/fetch_acs.py            (default: 2022 ACS 5-year)
        python scripts/fetch_acs.py 2023
"""
import sys

import pandas as pd
import requests

from common import ACS_CSV

VARS = {
    "B11001_001E": "households",
    "B19013_001E": "median_income",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "below_poverty",
    "B03002_001E": "pop_total",
    "B03002_003E": "pop_white_nonhispanic",
}

if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2022"
    url = (f"https://api.census.gov/data/{year}/acs/acs5?get={','.join(VARS)}"
           "&for=zip%20code%20tabulation%20area:*")
    print("Requesting", url)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(columns={"zip code tabulation area": "zip", **VARS})
    for c in VARS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = pd.NA   # Census uses -666666666 for "no data"
    df["poverty_rate"] = df["below_poverty"] / df["poverty_universe"]
    df["pct_people_of_color"] = 1 - df["pop_white_nonhispanic"] / df["pop_total"]
    # keep New Jersey ZIPs only (they start with 07 or 08)
    df = df[df["zip"].str[:2].isin(["07", "08"])]
    out = df[["zip", "households", "median_income", "poverty_rate", "pct_people_of_color"]]
    out.to_csv(ACS_CSV, index=False)
    print(f"Saved {len(out)} NJ ZIP areas to {ACS_CSV}")
