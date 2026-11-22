"""Download every table behind NRDC's NJ water-shutoff dashboard as CSV files.

The dashboard has no export button, but its table is loaded from a public
ArcGIS map service. Give this script any URL from that service (copied from
your browser's Network tab, see README) and it saves every layer to data/raw/.

Usage:
    python scripts/fetch_nrdc.py "https://services....arcgis.com/.../FeatureServer/0/query?..."
"""
import re
import sys
import time

import pandas as pd
import requests

from common import DATA

RAW = DATA / "raw"


def get_json(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, params={**(params or {}), "f": "json"}, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                msg = data["error"].get("message", data["error"])
                if "token" in str(msg).lower():
                    sys.exit("This service needs a login token. Copy the FULL request URL "
                             "from the Network tab (including any token=...) and try again.")
                raise RuntimeError(msg)
            return data
        except (requests.RequestException, ValueError) as e:
            if attempt == 2:
                raise
            print(f"  retrying after error: {e}")
            time.sleep(2)


def service_root(url):
    m = re.search(r"(https?://.+?/(?:FeatureServer|MapServer))", url, re.IGNORECASE)
    if not m:
        sys.exit("That URL doesn't contain /FeatureServer or /MapServer. "
                 "Copy a request URL that does (see README step 3b).")
    token = re.search(r"[?&]token=([^&]+)", url)
    return m.group(1), (token.group(1) if token else None)


def download_layer(root, layer_id, token):
    extra = {"token": token} if token else {}
    info = get_json(f"{root}/{layer_id}", extra)
    page = int(info.get("maxRecordCount") or 1000)
    oid = info.get("objectIdField") or next(
        (f["name"] for f in info.get("fields", []) if f.get("type") == "esriFieldTypeOID"), None)
    base = {"where": "1=1", "outFields": "*", "returnGeometry": "false", **extra}
    if oid:
        base["orderByFields"] = oid
    url = f"{root}/{layer_id}/query"

    rows, offset = [], 0
    paging = info.get("advancedQueryCapabilities", {}).get("supportsPagination", True)
    if paging:
        while True:
            data = get_json(url, {**base, "resultOffset": offset, "resultRecordCount": page})
            feats = data.get("features", [])
            rows += [f["attributes"] for f in feats]
            offset += len(feats)
            if not feats or (len(feats) < page and not data.get("exceededTransferLimit")):
                break
    else:  # older services: fetch all IDs, then fetch records in chunks
        ids = get_json(url, {**base, "returnIdsOnly": "true"}).get("objectIds") or []
        for i in range(0, len(ids), page):
            chunk = ",".join(map(str, ids[i:i + page]))
            data = get_json(url, {**base, "where": "", "objectIds": chunk})
            rows += [f["attributes"] for f in data.get("features", [])]
    return info.get("name", f"layer{layer_id}"), rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root, token = service_root(sys.argv[1])
    print("Service:", root)
    svc = get_json(root, {"token": token} if token else {})
    items = [(x["id"], x.get("name", "")) for x in svc.get("layers", []) + svc.get("tables", [])]
    if not items:
        sys.exit("No layers or tables found at that address.")
    RAW.mkdir(parents=True, exist_ok=True)
    for layer_id, name in items:
        print(f"\nDownloading layer {layer_id}: {name} ...")
        name, rows = download_layer(root, layer_id, token)
        df = pd.DataFrame(rows).astype("string")  # keep ZIPs like 07666 as text
        safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:60] or f"layer{layer_id}"
        out = RAW / f"layer{layer_id}_{safe}.csv"
        df.to_csv(out, index=False)
        print(f"  {len(df)} rows, {len(df.columns)} columns -> {out}")
        print("  columns:", ", ".join(df.columns))
    print(f"\nDone. Upload the files in {RAW} (or just paste the column lists above) "
          "so the converter to shutoffs.csv can be written for the real column names.")
