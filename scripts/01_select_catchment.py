# =============================================================================
# Step 01 - Choose a catchment for PDM flood forecasting and the rating-curve review
# Criteria: England (EA data), UK Benchmark Network, 50-500 km2, responsive
# (BFI < 0.55), hourly level available, >= 95% hourly flow 2006-2016, few QC flags
# Missing hourly files for candidate catchments are downloaded from the EIDC
# (token read from the EIDC_TOKEN environment variable or typed at the prompt)
# =============================================================================
import os
import re
import csv
import getpass
from urllib.parse import urljoin, unquote

import numpy as np
import pandas as pd
import requests

from config import DATA_ROOT as DESKTOP, WORK

PERIOD = ("2006-01-01", "2016-12-31")       # both hourly rainfall products available
NEEDED = ["topographic", "hydrologic", "humaninfluence", "hydrometry"]
EIDC_HOURLY = ("https://catalogue.ceh.ac.uk/datastore/eidchub/9a46d428-958f-4ac1-86eb-94eee70c0955/"
               "Catchment_Timeseries/hydro-meteorological/hourly/")

# --- Step 0: find the attribute files and the hourly folder ------------------
def find_one(pattern):
    """First match for a file name under the data folder (skips our outputs)."""
    hits = [p for p in DESKTOP.rglob(pattern) if WORK.name not in p.parts]
    return hits[0] if hits else None

attr_files = {n: find_one(f"camels_gb_v2_{n}_attributes.csv") for n in NEEDED}
hourly_dir = next((p for p in DESKTOP.rglob("hourly")
                   if p.is_dir() and "hydro-meteorological" in p.parts), None)
missing = [n for n, p in attr_files.items() if p is None] + ([] if hourly_dir else ["hourly folder"])
if missing:
    raise FileNotFoundError(f"Missing: {missing}")
print(f"Hourly folder: {hourly_dir}  ({len(list(hourly_dir.glob('*.csv')))} of 671 files on this PC)")

def read_attr(name):
    """Read one attribute table; rows with stray commas are repaired, not dropped."""
    path = attr_files[name]
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except pd.errors.ParserError:
        with open(path, encoding="latin-1", newline="") as fh:
            rows = list(csv.reader(fh))
        n = len(rows[0])
        fixed = [r[:n - 1] + [",".join(r[n - 1:])] if len(r) > n else r for r in rows[1:]]
        df = pd.DataFrame(fixed, columns=rows[0]).replace({"": np.nan, "NaN": np.nan})
        for c in df.columns:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
    df["gauge_id"] = df["gauge_id"].astype(str)
    return df.set_index("gauge_id")

# --- Step 1: catchment attributes --------------------------------------------
topo, hydro = read_attr("topographic"), read_attr("hydrologic")
human, metry = read_attr("humaninfluence"), read_attr("hydrometry")
cat = pd.concat([
    topo[["gauge_name", "area", "elev_mean"]],
    hydro[["baseflow_index", "q_mean"]],
    human[["benchmark_catch"]],
    metry[["station_type", "max_gauging_flow", "max_hourly_flow",
           "station_quality_hourlyflow_issues", "quncert_meta"]],
], axis=1)
cat["extrap_ratio"] = cat["max_hourly_flow"] / cat["max_gauging_flow"]

# --- Step 2: England only, via the NRFA hydrometric area (first digits of ID) -
# England: areas 22-54 and 68-76. Wales 55-67 and Scotland 1-21, 77+ are excluded.
# Area 54 includes some Welsh headwaters (e.g. 54025 Dulas) - checked by hand.
area_no = cat.index.astype(int) // 1000
cat["england"] = ((area_no >= 22) & (area_no <= 54)) | ((area_no >= 68) & (area_no <= 76))

pool = cat[(cat["benchmark_catch"] == "Y") & cat["england"]
           & cat["area"].between(50, 500) & (cat["baseflow_index"] < 0.55)
           & cat["station_quality_hourlyflow_issues"].isna()].copy()
print(f"\nBenchmark, England, 50-500 km2, BFI < 0.55, no hourly issues: {len(pool)}")

# --- Step 3a: download hourly files that are missing for the candidates -----
def local_hourly():
    return {re.search(r"timeseries_(\d+)_", p.name).group(1): p for p in hourly_dir.glob("*.csv")}

hourly_files = local_hourly()
need = [g for g in pool.index if g not in hourly_files]
print(f"Candidate hourly files missing on this PC: {len(need)}")
if need:
    token = os.environ.get("EIDC_TOKEN") or getpass.getpass("Paste your EIDC Personal Access Token: ")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    page = s.get(EIDC_HOURLY, timeout=60)
    page.raise_for_status()
    remote = {}                                          # gauge ID -> exact file name on the EIDC
    for href in re.findall(r'href="([^"#?]+\.csv)"', page.text):
        name = unquote(urljoin(EIDC_HOURLY, href).rsplit("/", 1)[-1])
        m = re.search(r"timeseries_(\d+)_", name)
        if m:
            remote[m.group(1)] = name
    for i, gid in enumerate(need, 1):
        if gid not in remote:
            print(f"  {gid}: not found on the EIDC - skipped")
            continue
        r = s.get(EIDC_HOURLY + remote[gid], timeout=300)
        r.raise_for_status()
        (hourly_dir / remote[gid]).write_bytes(r.content)
        print(f"  [{i}/{len(need)}] downloaded {remote[gid]} ({len(r.content)/1e6:.1f} MB)")
    hourly_files = local_hourly()

# --- Step 3b: hourly data quality 2006-2016 for each candidate --------------
SUSPECT_PEAK = {1, 2, 3, 4, 5, 7, 8}       # high-flow QC codes: unrealistic / unsupported events
rows = []
for gid in pool.index:
    if gid not in hourly_files:
        continue
    h = pd.read_csv(hourly_files[gid], parse_dates=["date"]).set_index("date").loc[PERIOD[0]:PERIOD[1]]
    flag = h["discharge_flag"].dropna().astype(int).astype(str).str.zfill(3)
    rows.append({
        "gauge_id": gid,
        "flow_%": 100 * h["discharge_vol"].notna().mean(),
        "level_%": 100 * h["level"].notna().mean() if "level" in h else 0.0,
        "basic_qc_flag_%": 100 * (flag.str[1] != "0").mean(),          # spikes, drops, truncation...
        "suspect_peaks": int(flag.str[2].astype(int).isin(SUSPECT_PEAK).sum()),
        "validated_peaks": int(flag.str[2].isin(["6", "9"]).sum()),
        "rain_cehgear1hr_%": 100 * h["precipitation_cehgear"].notna().mean(),
        "rain_gradgb_%": 100 * h["precipitation_gradgb"].notna().mean(),
    })
qc = pd.DataFrame(rows).set_index("gauge_id")
cand = pool.join(qc, how="inner")

# --- Step 4: keep good-quality candidates and rank them ----------------------
good = cand[(cand["flow_%"] >= 95) & (cand["level_%"] >= 90)].copy()
good = good.sort_values(["suspect_peaks", "basic_qc_flag_%", "extrap_ratio"])
cols = ["gauge_name", "area", "baseflow_index", "station_type", "flow_%", "level_%",
        "basic_qc_flag_%", "suspect_peaks", "validated_peaks", "extrap_ratio", "quncert_meta"]
print(f"\nCandidates with >= 95% hourly flow and >= 90% level, 2006-2016: {len(good)}\n")
print(good[cols].round(2).to_string())

cand.to_csv(WORK / "pdm_catchment_candidates.csv")
print(f"\nSaved: {WORK / 'pdm_catchment_candidates.csv'}")
