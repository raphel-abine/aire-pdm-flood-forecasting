# =============================================================================
# Step 03 - Verify the PDM (pdm.py) before calibration
# Test 1: water balance closes over the full 2005-2016 record
# Test 2: a single rain pulse from empty stores - all water accounted for
# Test 3: stress test (5x rain, 3x PE, fast stores) - no NaNs, no negative storage
# Test 4: run time (numba) and a first-guess hydrograph for Boxing Day 2015
# =============================================================================
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import WORK, FIG, AREA_KM2
import pdm

print("numba available:", pdm.HAS_NUMBA)
data = pd.read_csv(WORK / "kildwick_27035_hourly.csv", parse_dates=["date"]).set_index("date")
P, E, Qobs = data["rain_gear"].values, data["pe"].values, data["q"].values

# First-guess parameters (NOT calibrated) - only to exercise the code
guess = dict(cmax=150, b=0.5, be=2, kg=1500, st=0.0, bg=1.0, k1=5, k2=5, kb=3e-7, td=1)

# --- Test 1: full-record water balance ----------------------------------------
pdm.simulate(guess, P[:48], E[:48], q0=0.1)          # first call compiles (numba)
t0 = time.perf_counter()
res = pdm.simulate(guess, P, E, q0=pdm.to_mmh(Qobs[0], AREA_KM2))
run_s = time.perf_counter() - t0
wb = pdm.water_balance(res)
print(f"\nTest 1 - full record ({len(P):,} hours), one run = {run_s:.3f} s")
for k, v in wb.items():
    print(f"  {k:>10}: {v:12.6f} mm")
assert abs(wb["closure"]) < 1e-6, "Water balance does not close"

# --- Test 2: single 50 mm pulse from empty stores, 3000 days of drainage ------
Pp = np.zeros(24 * 3000); Pp[:5] = 10.0
res2 = pdm.simulate(guess, Pp, np.zeros_like(Pp), states=dict(S=0, S1=0, S2=0, Sg=0))
wb2 = pdm.water_balance(res2)
print(f"\nTest 2 - 50 mm pulse: flow out {wb2['flow']:.2f} mm, still stored "
      f"{wb2['d_storage']:.2f} mm (slow cubic tail), closure {wb2['closure']:.1e} mm")
assert abs(wb2["closure"]) < 1e-6

# --- Test 3: stress test ------------------------------------------------------
harsh = dict(guess, cmax=20, b=2, kg=50, bg=2, k1=0.3, k2=0.2, kb=1e-3)
res3 = pdm.simulate(harsh, 5 * P, 3 * E, q0=pdm.to_mmh(Qobs[0], AREA_KM2))
wb3 = pdm.water_balance(res3)
ok = (not np.isnan(res3["q"]).any()) and res3["S"].min() >= 0 and res3["Sg"].min() >= 0
print(f"\nTest 3 - stress test: closure {wb3['closure']:.1e} mm, "
      f"no NaNs / no negative stores: {ok}")
assert ok and abs(wb3["closure"]) < 1e-6
print("\nAll checks passed - the model conserves mass.")

# --- Test 4: first-guess hydrograph, Boxing Day 2015 (uncalibrated) ----------
qsim = pd.Series(pdm.to_m3s(res["q"], AREA_KM2), index=data.index)
win = slice("2015-12-20", "2016-01-02")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2.5]})
ax1.bar(data.loc[win].index, data.loc[win, "rain_gear"], width=1 / 24, color="#3b6fb6")
ax1.invert_yaxis()
ax1.set_ylabel("Rain (mm/h)")
ax2.plot(data.loc[win].index, data.loc[win, "q"], color="#1f2a44", lw=1.4, label="Observed")
ax2.plot(qsim.loc[win].index, qsim.loc[win], color="#c8553d", lw=1.2, ls="--",
         label="PDM first guess (uncalibrated)")
ax2.set_ylabel("Flow (m³/s)")
ax2.legend(frameon=False, fontsize=8)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("27035 Aire at Kildwick - code check only, parameters not calibrated", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "pdm_code_check.png", dpi=150)
