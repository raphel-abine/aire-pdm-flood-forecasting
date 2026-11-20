# =============================================================================
# Step 02 - Build the hourly dataset for 27035 Aire at Kildwick Bridge
# Hourly flow, level, rainfall (CEH-GEAR1hr, GRaD-GB) + daily PE spread to hourly
# Split-sample design (Klemes 1986): warm-up 2005-10 to 2006-09,
# calibration 2006-10 to 2011-09, validation 2011-10 to 2016-09
# Flood events: peaks over threshold (POT), independent if >= 5 days apart
# Missing daily file is downloaded from the EIDC (token via EIDC_TOKEN or prompt)
# =============================================================================
import os
import re
import getpass
from urllib.parse import urljoin, unquote

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from config import DATA_ROOT as DESKTOP, WORK, FIG, GAUGE, AREA_KM2

START, END = "2005-10-01", "2016-09-30 23:00"
WARM = ("2005-10-01", "2006-09-30 23:00")
CAL = ("2006-10-01", "2011-09-30 23:00")
VAL = ("2011-10-01", "2016-09-30 23:00")
EIDC_TS = ("https://catalogue.ceh.ac.uk/datastore/eidchub/9a46d428-958f-4ac1-86eb-94eee70c0955/"
           "Catchment_Timeseries/hydro-meteorological/")

# --- 1. Locate the hourly and daily files for this gauge ---------------------
files = [p for p in DESKTOP.rglob(f"*{GAUGE}*.csv")
         if WORK.name not in p.parts and re.search(rf"_{GAUGE}_", p.name)]
hourly_f = next((p for p in files if "hourly" in p.name.lower()), None)
daily_f = next((p for p in files if "hourly" not in p.name.lower()
                and "timeseries" in p.name.lower()), None)
if hourly_f is None:
    raise FileNotFoundError(f"Hourly file for {GAUGE} not found - run 01_select_catchment.py first")

# --- 1b. Download the daily file from the EIDC if it is missing --------------
if daily_f is None:
    print("Daily file missing - downloading from the EIDC")
    token = os.environ.get("EIDC_TOKEN") or getpass.getpass("Paste your EIDC Personal Access Token: ")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    daily_url = EIDC_TS + "daily/"
    page = s.get(daily_url, timeout=60)
    page.raise_for_status()
    names = [unquote(urljoin(daily_url, h).rsplit("/", 1)[-1])
             for h in re.findall(r'href="([^"#?]+\.csv)"', page.text)]
    match = [n for n in names if f"_{GAUGE}_" in n]
    if not match:
        raise FileNotFoundError(f"No daily file for {GAUGE} on the EIDC page")
    target_dir = hourly_f.parent.parent / "daily"
    target_dir.mkdir(exist_ok=True)
    r = s.get(daily_url + match[0], timeout=300)
    r.raise_for_status()
    daily_f = target_dir / match[0]
    daily_f.write_bytes(r.content)
print("Hourly:", hourly_f.name, "\nDaily: ", daily_f.name)

# --- 2. Load hourly data -------------------------------------------------------
h = pd.read_csv(hourly_f, parse_dates=["date"]).set_index("date").loc[START:END]

# --- 3. Daily PE (Hydro-PE) spread evenly over 24 h ---------------------------
# PE has little effect on hourly flood response; uniform disaggregation keeps
# the daily total exact and is a common simplification for hourly PDM runs.
d = pd.read_csv(daily_f, parse_dates=["date"]).set_index("date")
pe_cols = [c for c in d.columns if re.search(r"pet|pe_|^pe$|evap", c, re.I)]
if not pe_cols:
    raise KeyError(f"No PE column found in the daily file: {list(d.columns)}")
pe_col = next((c for c in pe_cols if "hydro" in c.lower()), pe_cols[0])   # prefer Hydro-PE
pe_hourly = (d[pe_col] / 24.0).reindex(h.index, method="ffill")           # mm/h
print(f"Using daily PE column: {pe_col}")

# --- 4. Assemble the model dataset -------------------------------------------
data = pd.DataFrame({
    "rain_gear": h["precipitation_cehgear"],      # mm/h, CEH-GEAR1hr (gauge-based)
    "rain_grad": h["precipitation_gradgb"],       # mm/h, GRaD-GB (radar-gauge)
    "pe": pe_hourly,                              # mm/h
    "q": h["discharge_vol"],                      # m3/s
    "level": h["level"],                          # m above gauge datum
    "flag": h["discharge_flag"],
})
data["period"] = np.select(
    [data.index <= WARM[1], data.index <= CAL[1]], ["warmup", "calibration"], "validation")

# --- 5. Data completeness by period -------------------------------------------
print("\nCompleteness (%) by period:")
print((data.drop(columns=["flag", "period"]).notna().groupby(data["period"]).mean() * 100).round(2))

# --- 6. Water balance check across the full window ----------------------------
q_mm = data["q"] * 3.6 / AREA_KM2                 # m3/s -> mm/h over the catchment
yrs = (data.index[-1] - data.index[0]).days / 365.25
print(f"\nMean annual rain (GEAR1hr): {data['rain_gear'].sum() / yrs:7.0f} mm")
print(f"Mean annual rain (GRaD-GB): {data['rain_grad'].sum() / yrs:7.0f} mm")
print(f"Mean annual PE:             {data['pe'].sum() / yrs:7.0f} mm")
print(f"Mean annual runoff:         {q_mm.sum() / yrs:7.0f} mm")
print(f"Runoff coefficient (GEAR):  {q_mm.sum() / data['rain_gear'].sum():7.2f}")

# --- 7. Independent flood events (POT) ----------------------------------------
# Threshold at the 99th percentile hourly flow; peaks at least 120 h apart
thr = data["q"].quantile(0.99)
idx, _ = find_peaks(data["q"].fillna(0).values, height=thr, distance=120)
events = data.iloc[idx][["q", "level", "period"]].copy()
events = events.sort_values("q", ascending=False)
print(f"\nPOT threshold: {thr:.1f} m3/s -> {len(events)} independent events "
      f"({(events.period == 'calibration').sum()} calibration, "
      f"{(events.period == 'validation').sum()} validation)")
print(events.head(10).round(2).to_string())

# --- 8. Overview figure: rainfall on top, flow below, periods shaded -------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2.5]})
daily_rain = data["rain_gear"].resample("D").sum()
ax1.bar(daily_rain.index, daily_rain.values, width=1.0, color="#3b6fb6")
ax1.invert_yaxis()                                 # hyetograph hangs from the top
ax1.set_ylabel("Rain (mm/day)")
ax2.plot(data.index, data["q"], lw=0.6, color="#1f2a44")
ax2.scatter(events.index, events["q"], s=14, color="#c8553d", zorder=3, label="POT event peaks")
ax2.axhline(thr, ls="--", lw=0.8, color="grey", label=f"99th percentile ({thr:.0f} m³/s)")
ymax = data["q"].max() * 1.12
ax2.set_ylim(0, ymax)
for (a, b), name, shade in [(WARM, "Warm-up", "#eeeeee"),
                            (CAL, "Calibration", "#e3ecf7"),
                            (VAL, "Validation", "#f7ece3")]:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    for ax in (ax1, ax2):
        ax.axvspan(a, b, color=shade, zorder=0)
    ax2.text(a + (b - a) / 2, ymax * 0.95, name, ha="center", fontsize=9, color="#555555")
ax2.set_ylabel("Flow (m³/s)")
ax2.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=8, frameon=False)
fig.suptitle("27035 Aire at Kildwick Bridge - hourly record and split-sample periods", fontsize=11)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "overview_27035.png", dpi=200)

# --- 9. Save outputs -----------------------------------------------------------
data.to_csv(WORK / "kildwick_27035_hourly.csv")
events.to_csv(WORK / "kildwick_27035_events.csv")
print(f"\nSaved dataset, events and overview figure to {WORK}")
