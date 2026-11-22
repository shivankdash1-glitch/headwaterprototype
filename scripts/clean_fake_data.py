"""Delete the stand-in data and everything made from it, so nothing fake
ends up in your demo. Only deletes files that are marked SYNTHETIC.

Usage:  python scripts/clean_fake_data.py
"""
import pandas as pd

from common import ACS_CSV, OUT, SHUTOFFS_CSV

removed = []
if SHUTOFFS_CSV.exists():
    s = pd.read_csv(SHUTOFFS_CSV, dtype=str)
    if "source" in s.columns and s["source"].str.upper().eq("SYNTHETIC").all():
        SHUTOFFS_CSV.unlink()
        removed.append(SHUTOFFS_CSV.name)
    else:
        print(f"{SHUTOFFS_CSV.name} holds real data; leaving it.")
if ACS_CSV.exists():
    a = pd.read_csv(ACS_CSV, dtype=str)
    if "synthetic" in a.columns:
        ACS_CSV.unlink()
        removed.append(ACS_CSV.name)
if OUT.exists():
    for f in OUT.iterdir():
        f.unlink()
        removed.append(f"outputs/{f.name}")
print("Removed:", ", ".join(removed) if removed else "nothing")
