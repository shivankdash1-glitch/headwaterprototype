"""Make STAND-IN data so you can run the whole pipeline before the real
spreadsheet is ready. Every row is labeled SYNTHETIC and the app shows a warning.
Never present results from this data.

It will not overwrite a real data/shutoffs.csv.

Usage:  python scripts/make_fake_data.py
"""
import json

import numpy as np
import pandas as pd

from common import ACS_CSV, GEOJSON, SHUTOFFS_CSV

UTILITIES = ["New Jersey American Water", "Aqua New Jersey", "Middlesex Water", "Veolia New Jersey"]

if __name__ == "__main__":
    if SHUTOFFS_CSV.exists():
        existing = pd.read_csv(SHUTOFFS_CSV, dtype=str)
        if "source" not in existing.columns or not existing["source"].str.upper().eq("SYNTHETIC").all():
            raise SystemExit(f"{SHUTOFFS_CSV} holds real data. Not overwriting it.")

    rng = np.random.default_rng(7)
    if GEOJSON.exists():
        zips = sorted(f["properties"]["ZCTA5CE10"] for f in json.loads(GEOJSON.read_text())["features"])
    else:
        zips = [f"0{n}" for n in range(7001, 8999)]
    zips = sorted(rng.choice(zips, size=346, replace=False).tolist())
    n = len(zips)

    util = rng.choice(UTILITIES, size=n, p=[0.6, 0.15, 0.1, 0.15])
    households = rng.lognormal(8.6, 0.7, n).round()
    poverty = np.clip(rng.beta(2, 12, n), 0.01, 0.6)
    poc = np.clip(poverty * 2.2 + rng.normal(0.15, 0.15, n), 0.02, 0.98)
    income = np.exp(11.6 - 2.2 * poverty + rng.normal(0, 0.25, n)).round(-2)

    policy = {"New Jersey American Water": 1.0, "Aqua New Jersey": 0.7,
              "Middlesex Water": 0.5, "Veolia New Jersey": 0.8}
    base_rate = 0.004 * np.exp(6 * poverty) * np.array([policy[u] for u in util])
    momentum = rng.normal(0, 0.15, n)
    momentum[rng.choice(n, 25, replace=False)] += rng.uniform(0.5, 1.2, 25)  # ZIPs heating up

    rows = []
    level = base_rate * households
    for year, factor in [(2019, 1.0), (2022, 0.45), (2023, 1.0), (2024, 1.0)]:
        if year >= 2023:
            level = level * np.exp(momentum + rng.normal(0, 0.2, n))
            if year == 2024:  # one utility tightens collections
                level = level * np.where(util == "New Jersey American Water", 1.6, 1.0)
        counts = rng.poisson(level * factor)
        rows += [{"zip": z, "year": year, "utility": u, "shutoffs": int(c), "source": "SYNTHETIC"}
                 for z, u, c in zip(zips, util, counts)]
    pd.DataFrame(rows).to_csv(SHUTOFFS_CSV, index=False)
    print(f"Wrote STAND-IN shutoffs for {n} ZIPs to {SHUTOFFS_CSV}")

    if not ACS_CSV.exists():
        pd.DataFrame({"zip": zips, "households": households, "median_income": income,
                      "poverty_rate": poverty, "pct_people_of_color": poc, "synthetic": True}) \
            .to_csv(ACS_CSV, index=False)
        print(f"Wrote STAND-IN Census data to {ACS_CSV} (run fetch_acs.py to replace it with real data)")
