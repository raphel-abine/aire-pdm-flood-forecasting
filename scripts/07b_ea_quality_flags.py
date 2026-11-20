# =============================================================================
# Step 07b - Check: how does the EA flag flows above the gauged range?
# Downloads flow and level for three large floods with ALL quality flags kept
# (the step 07 cache removes 'Suspect' readings) and reports the peak by flag
# =============================================================================
import pandas as pd
import requests

from config import WORK

HYD = "https://environment.data.gov.uk/hydrology/id/measures/"
SID = "351f05c7-7a89-46c0-9d8a-dc56c5f7b42d"        # EA station F1503 Kildwick
MEAS = {"flow": f"{HYD}{SID}-flow-i-900-m3s-qualified",
        "level": f"{HYD}{SID}-level-i-900-m-qualified"}
EVENTS = {"Boxing Day 2015": ("2015-12-25", "2015-12-28"),
          "Sep 2012": ("2012-09-24", "2012-09-28"),
          "Feb 2020 (Storm Ciara)": ("2020-02-08", "2020-02-11")}

rows = []
for name, (d1, d2) in EVENTS.items():
    frames = {}
    for p, url in MEAS.items():
        r = requests.get(f"{url}/readings",
                         params={"mineq-date": d1, "max-date": d2, "_limit": 2000},
                         headers={"Accept": "application/json"}, timeout=120)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["items"])[["dateTime", "value", "quality"]]
        frames[p] = df.set_index("dateTime").add_suffix(f"_{p}")
    j = frames["level"].join(frames["flow"])
    print(f"\n=== {name} ===")
    print("Flow quality flags:", j["quality_flow"].value_counts(dropna=False).to_dict())
    top = j.sort_values("value_level", ascending=False).head(3)
    print(top[["value_level", "quality_level", "value_flow", "quality_flow"]].to_string())
    print("Max flow by quality:", j.groupby("quality_flow")["value_flow"].max().round(1).to_dict())
    sus = j[j["quality_flow"] == "Suspect"]
    if len(sus):
        print(f"Flow flagged Suspect from stage {sus['value_level'].min():.2f} m "
              f"({sus['value_flow'].min():.1f} m3/s) upwards")
        rows.append({"event": name, "peak_stage_m": j["value_level"].max(),
                     "peak_flow_suspect_m3s": sus["value_flow"].max(),
                     "max_good_flow_m3s": j.loc[j["quality_flow"] == "Good", "value_flow"].max(),
                     "suspect_from_stage_m": sus["value_level"].min()})

pd.DataFrame(rows).to_csv(WORK / "ea_quality_flags_floods.csv", index=False)
print(f"\nSaved: {WORK / 'ea_quality_flags_floods.csv'}")
