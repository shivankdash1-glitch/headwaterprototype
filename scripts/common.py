"""Shared paths and helpers for the Headwater prototype."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SHUTOFFS_CSV = DATA / "shutoffs.csv"        # you create this from the NRDC spreadsheet
ACS_CSV = DATA / "acs_zcta.csv"             # made by fetch_acs.py
BILLS_CSV = DATA / "utility_bills.csv"      # you fill this in from each utility's tariff
PROGRAMS_CSV = DATA / "programs.csv"        # assistance programs for the questionnaire
GEOJSON = DATA / "nj_zips.geojson"          # made by fetch_geo.py
PANEL_CSV = OUT / "panel.csv"               # made by build_panel.py
BACKTEST_CSV = OUT / "backtest_predictions.csv"
FORECAST_CSV = OUT / "forecast.csv"
RESULTS_MD = OUT / "results.md"
RESULTS_JSON = OUT / "results.json"


def clean_zip(series: pd.Series) -> pd.Series:
    """Excel drops the leading zero from NJ ZIPs (07302 -> 7302). Put it back."""
    s = series.astype(str).str.strip().str.split(".").str[0].str.split("-").str[0]
    return s.str.zfill(5)
