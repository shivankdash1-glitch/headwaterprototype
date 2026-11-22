"""Download New Jersey ZIP code boundaries (for the map). Run once.

Usage:  python scripts/fetch_geo.py
"""
import requests

from common import GEOJSON

URL = ("https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/"
       "master/nj_new_jersey_zip_codes_geo.min.json")

if __name__ == "__main__":
    GEOJSON.parent.mkdir(exist_ok=True)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    GEOJSON.write_bytes(r.content)
    print(f"Saved {GEOJSON} ({len(r.content) / 1e6:.1f} MB)")
