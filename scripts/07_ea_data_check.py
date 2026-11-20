# =============================================================================
# Step 07 - EA operational data check for Kildwick (EA Hydrology + Flood-Monitoring APIs)
# 1. Find the station and its 15-min level and flow measures (Hydrology API)
# 2. Download 15-min level and flow, 2005 to the latest available (cached)
# 3. Overlap check: EA archive vs CAMELS-GB hourly data, 2005-2016
# 4. Re-rating check: post-2016 EA flows vs the 2005-2016 rating from step 06
# 5. Published stage scale (typical range, records) - Flood-Monitoring API
# Data: Environment Agency, Open Government Licence (no key needed)
# =============================================================================
import time

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from config import WORK, FIG

EA_DIR = WORK / "ea_data"
EA_DIR.mkdir(exist_ok=True)
HYD = "https://environment.data.gov.uk/hydrology"
FMON = "https://environment.data.gov.uk/flood-monitoring"
HDR = {"Accept": "application/json"}
START_YEAR = 2005
SPLIT = pd.Timestamp("2016-10-01")      # end of the CAMELS-GB period we modelled
TITLE = "27035 Aire at Kildwick - EA operational data check"


def get_json(url, params=None, tries=3):
    """GET with retries; returns parsed JSON or raises with the URL shown."""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=HDR, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if k == tries - 1:
                raise RuntimeError(f"Request failed: {url} {params}\n{e}")
            time.sleep(3 * (k + 1))


def is_aire(s):
    text = (str(s.get("riverName", "")) + str(s.get("label", ""))).lower()
    return "aire" in text


# --- 1. Find the Kildwick station and its measures ---------------------------
# Only needed for the first download; skipped when both caches already exist.
MEAS = {}


def find_measures():
    stations = get_json(f"{HYD}/id/stations", {"search": "Kildwick"})["items"]
    print(f"Hydrology API stations matching 'Kildwick': {len(stations)}")
    for s in stations:
        print(f"  {s.get('label')} | river: {s.get('riverName')} "
              f"| wiskiID: {s.get('wiskiID')} | RLOIid: {s.get('RLOIid')}")
    stn = next(s for s in stations if is_aire(s))
    ids = [m["@id"] if isinstance(m, dict) else m for m in stn.get("measures", [])]
    if not ids:                          # some responses need the measures endpoint
        items = get_json(f"{HYD}/id/measures", {"station": stn["@id"]})["items"]
        ids = [m["@id"] for m in items]
    for param in ("level", "flow"):      # 15-minute instantaneous measures
        hits = [m for m in ids if f"-{param}-i-900" in m]
        if not hits:
            raise KeyError(f"No 15-min {param} measure: {ids}")
        MEAS[param] = hits[0]


# --- 2. Download 15-min readings year by year (cached as CSV) ---------------
def download(param):
    cache = EA_DIR / f"kildwick_{param}_15min.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["dateTime"])
        return df.set_index("dateTime")["value"]
    if not MEAS:
        find_measures()
    parts = []
    for yr in range(START_YEAR, pd.Timestamp.today().year + 1):
        q = {"mineq-date": f"{yr}-01-01", "max-date": f"{yr + 1}-01-01",
             "_limit": 50000}
        items = get_json(f"{MEAS[param]}/readings", q)["items"]
        if items:
            parts.append(pd.DataFrame(items)[["dateTime", "value", "quality"]])
        print(f"  {param} {yr}: {len(items):,} readings")
    df = pd.concat(parts, ignore_index=True)
    df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True).dt.tz_localize(None)
    print(f"  {param} quality flags:", df["quality"].value_counts().to_dict())
    df = df[~df["quality"].isin(["Suspect", "Missing"])].dropna(subset=["value"])
    df.to_csv(cache, index=False)
    return df.set_index("dateTime")["value"]


ea = pd.concat({"ea_level": download("level"), "ea_flow": download("flow")},
               axis=1, sort=True)
ea = ea[~ea.index.duplicated()]
print(f"EA record: {ea.index.min():%Y-%m-%d} to {ea.index.max():%Y-%m-%d}, "
      f"{ea.dropna().shape[0]:,} paired 15-min readings "
      "(Suspect/Missing removed)")

# --- 3. Overlap check against CAMELS-GB (hourly means, 2005-2016) ----------
cam = pd.read_csv(WORK / "kildwick_27035_hourly.csv", parse_dates=["date"])
cam = cam.set_index("date")
ea_h = ea.resample("1h").mean()

# Test hour offsets: datasets may label an hour by its start or its end
best_shift, best_r = 0, -1
for shift in (-1, 0, 1):
    j = cam[["q"]].join(ea_h.shift(shift, freq="h"), how="inner").dropna()
    r = np.corrcoef(j["q"], j["ea_flow"])[0, 1]
    print(f"  shift {shift:+d} h: r(flow) = {r:.5f}")
    if r > best_r:
        best_shift, best_r = shift, r

both = cam[["q", "level"]].join(ea_h.shift(best_shift, freq="h"), how="inner")
both = both.dropna()
print(f"\nOverlap with CAMELS-GB (best alignment {best_shift:+d} h): "
      f"{len(both):,} hours")
for c_cam, c_ea in [("q", "ea_flow"), ("level", "ea_level")]:
    r = np.corrcoef(both[c_cam], both[c_ea])[0, 1]
    bias = 100 * (both[c_ea].mean() / both[c_cam].mean() - 1)
    p99 = np.percentile(np.abs(both[c_ea] - both[c_cam]), 99)
    print(f"  {c_ea:>8}: r = {r:.4f}, mean difference = {bias:+.2f}%, "
          f"99th pct absolute difference = {p99:.3f}")

# --- 4. Re-rating check: EA flow vs the 2005-2016 rating from step 06 ---------
seg = pd.read_csv(WORK / "rating_segments_27035.csv")
breaks = [-np.inf] + seg["level_to_m"].iloc[:-1].tolist() + [np.inf]
BK_UPPER = seg["level_to_m"].iloc[1]


def rating_0516(h):
    """2005-2016 rating recovered in step 06: stage (m) -> flow (m3/s)."""
    h = np.asarray(h, dtype=float)
    q = np.full_like(h, np.nan)
    for (_, s), lo, hi in zip(seg.iterrows(), breaks[:-1], breaks[1:]):
        m = (h > lo) & (h <= hi)
        q[m] = s["a"] * np.clip(h[m] - s["h0_m"], 0, None) ** s["b"]
    return q


ea_pairs = ea.dropna().copy()
ea_pairs = ea_pairs[ea_pairs["ea_level"] > 0.3]          # skip low-stage noise
ea_pairs["ratio"] = ea_pairs["ea_flow"] / rating_0516(ea_pairs["ea_level"].values)
ea_pairs["wy"] = ea_pairs.index.year + (ea_pairs.index.month >= 10)  # water year

bands = [(0.9, 1.1), (1.9, 2.1), (2.6, 2.8), (3.2, 3.4), (3.8, 4.6)]
cols = {}
for lo, hi in bands:
    sel = ea_pairs[(ea_pairs["ea_level"] > lo) & (ea_pairs["ea_level"] <= hi)]
    cols[f"{lo}-{hi} m"] = sel.groupby("wy")["ratio"].median()
tab = pd.DataFrame(cols)
print("\nEA flow / 2005-2016 rating, median by water year and stage band")
print("(1.00 = same rating; NaN = stage not reached):")
print(tab.round(3).to_string())

wy_all = ea.index.year + (ea.index.month >= 10)
amax = pd.DataFrame({"max_stage_m": ea.groupby(wy_all)["ea_level"].max(),
                     "max_flow_m3s": ea.groupby(wy_all)["ea_flow"].max()})
print("\nAnnual maximum stage and flow by water year:")
print(amax.round(2).to_string())

# --- 5. Published stage scale from the Flood-Monitoring API ----------------
try:
    fm = get_json(f"{FMON}/id/stations", {"search": "Kildwick"})["items"]
    fm_stn = next(s for s in fm if is_aire(s))
    ref = fm_stn["stationReference"]
    scale = get_json(f"{FMON}/id/stations/{ref}/stageScale")["items"]
    pub = {k: scale.get(k)
           for k in ("typicalRangeLow", "typicalRangeHigh", "scaleMax")}
    for k in ("highestRecent", "maxOnRecord", "minOnRecord"):
        v = scale.get(k)
        if isinstance(v, dict):
            pub[k] = f"{v.get('value')} m on {str(v.get('dateTime'))[:10]}"
    print(f"\nFlood-Monitoring station {ref} published stage scale:")
    for k, v in pub.items():
        print(f"  {k:<17} {v}")
except Exception as e:
    print("\nStage scale not retrieved:", e)

# --- 6. Figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
ax.scatter(both["q"], both["ea_flow"], s=2, alpha=0.3, color="#1f2a44")
lim = [0, both[["q", "ea_flow"]].max().max() * 1.05]
ax.plot(lim, lim, color="#c8553d", lw=1)
ax.set_xlabel("CAMELS-GB hourly flow (m³/s)")
ax.set_ylabel("EA hourly mean flow (m³/s)")
ax.set_title("(a) EA archive vs CAMELS-GB, 2005-2016", fontsize=10)

ax = axes[1]
pre = ea_pairs[ea_pairs.index < SPLIT]
post = ea_pairs[ea_pairs.index >= SPLIT]
ax.scatter(pre["ea_level"], pre["ratio"], s=2, alpha=0.2,
           color="#7a7a7a", label="Before Oct 2016")
ax.scatter(post["ea_level"], post["ratio"], s=2, alpha=0.3,
           color="#c8553d", label="Oct 2016 onwards")
ax.axhline(1, color="black", lw=0.8)
ax.axvline(BK_UPPER, color="#3b6fb6", lw=0.8, ls="--")
ax.set_ylim(0.7, 1.3)
ax.set_xlabel("Stage (m)")
ax.set_ylabel("EA flow / 2005-2016 rating")
ax.set_title("(b) Has the rating changed since 2016?", fontsize=10)
ax.legend(frameon=False, fontsize=8, markerscale=4)

ax = axes[2]
bar_col = ["#c8553d" if y >= 2017 else "#3b6fb6" for y in amax.index]
ax.bar(amax.index, amax["max_stage_m"], color=bar_col)
ax.axhline(3.27, color="grey", lw=0.8, ls="--",
           label="Stage of highest gauging (1995)")
ax.axhline(BK_UPPER, color="#e0a458", lw=0.8, ls="--",
           label="Out-of-bank breakpoint")
ax.set_xlabel("Water year")
ax.set_ylabel("Annual maximum stage (m)")
ax.set_title("(c) Annual maximum stage, EA 15-min record", fontsize=10)
ax.legend(frameon=False, fontsize=8)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle(TITLE, fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "ea_data_check.png", dpi=200)

# --- 7. Save -------------------------------------------------------------------
tab.to_csv(WORK / "ea_rating_ratio_by_year.csv")
amax.to_csv(WORK / "ea_annual_max.csv")
print(f"\nSaved EA comparison tables to {WORK}")
