# =============================================================================
# Step 04 - Calibrate the PDM on 27035 Aire at Kildwick Bridge
# Split-sample test (Klemes 1986): calibrate Oct 2006 - Sep 2011,
# validate Oct 2011 - Sep 2016 (continuous run, warm-up from Oct 2005)
# Objective: NSE on hourly flow (same measure as the benchmark project)
# Optimiser: differential evolution (Storn & Price 1997), 3 independent seeds
#            to check the optimum is stable (not a local minimum)
# Fixed (standard PDM choices, Moore 2007): be = 2 (quadratic evaporation),
#            bg = 1 (linear drainage) - reduces parameter interaction
# Run time: about 4-5 minutes (3 seeds x ~75 s with numba)
# =============================================================================
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from config import WORK, FIG, AREA_KM2
import pdm

SEEDS = [1, 2, 3]

# --- 1. Data ---------------------------------------------------------------------
data = pd.read_csv(WORK / "kildwick_27035_hourly.csv", parse_dates=["date"]).set_index("date")
events = pd.read_csv(WORK / "kildwick_27035_events.csv", parse_dates=["date"])
P, E, Q = data["rain_gear"].values, data["pe"].values, data["q"].values
period = data["period"].values
q0 = pdm.to_mmh(Q[0], AREA_KM2)                           # initial baseflow from observed flow

end_cal = data.index.get_loc(pd.Timestamp("2011-09-30 23:00")) + 1   # run only to end of calibration
cal_mask = (period == "calibration")[:end_cal]
Q_cal = Q[:end_cal][cal_mask]
den_cal = ((Q_cal - Q_cal.mean()) ** 2).sum()

# --- 2. Parameter space ---------------------------------------------------------
# kg and kb span orders of magnitude, so they are searched in log10 space.
# The two routing constants are sorted so k1 (fast) <= k2 (slow); this removes the
# k1/k2 swap symmetry so different seeds return directly comparable parameters.
FIXED = dict(be=2.0, bg=1.0)
SEARCH = [("cmax", 20, 600),       # maximum store capacity (mm)
          ("b", 0.05, 2.0),        # Pareto shape (spatial variability of capacity)
          ("log10_kg", 0, 6),      # drainage time constant (h)
          ("st", 0.0, 0.99),       # drainage threshold as fraction of Smax
          ("k_a", 0.5, 72),        # routing time constant A (h)
          ("k_b", 0.5, 72),        # routing time constant B (h)
          ("log10_kb", -12, -2),   # cubic baseflow store constant
          ("td", 0, 6)]            # time delay (h, integer)
bounds = [(lo, hi) for _, lo, hi in SEARCH]
integrality = [name == "td" for name, _, _ in SEARCH]

def unpack(x):
    """Optimiser vector -> PDM parameter dict."""
    p = dict(cmax=x[0], b=x[1], kg=10 ** x[2], st=x[3],
             k1=min(x[4], x[5]), k2=max(x[4], x[5]),
             kb=10 ** x[6], td=int(round(x[7])))
    p.update(FIXED)
    return {k: float(v) for k, v in p.items()}

def objective(x):
    """1 - NSE over the calibration period (minimised)."""
    r = pdm.simulate(unpack(x), P[:end_cal], E[:end_cal], q0=q0)
    sim = pdm.to_m3s(r["q"], AREA_KM2)[cal_mask]
    return ((sim - Q_cal) ** 2).sum() / den_cal

# --- 3. Calibrate with three independent seeds ---------------------------------
objective(np.array([lo for _, lo, _ in SEARCH]))          # compile numba before timing
runs = []
for seed in SEEDS:
    t0 = time.perf_counter()
    r = differential_evolution(objective, bounds, integrality=integrality, popsize=12,
                               maxiter=150, tol=1e-6, seed=seed, init="sobol", polish=False)
    runs.append({"seed": seed, "nse_cal": 1 - r.fun, **unpack(r.x)})
    print(f"Seed {seed}: calibration NSE = {1 - r.fun:.4f}  "
          f"({r.nfev:,} model runs, {time.perf_counter() - t0:.0f} s)")
runs = pd.DataFrame(runs).set_index("seed")
print("\nParameters from each seed (should agree closely):")
print(runs.T.to_string(float_format=lambda v: f"{v:.4g}"))

best = runs["nse_cal"].idxmax()
params = runs.loc[best].drop("nse_cal").to_dict()
params["td"] = int(params["td"])

# Flag any parameter sitting on a search bound (a sign the bounds are too tight)
x_best = [params["cmax"], params["b"], np.log10(params["kg"]), params["st"],
          params["k1"], params["k2"], np.log10(params["kb"]), params["td"]]
on_bound = [n for (n, lo, hi), v in zip(SEARCH, x_best)
            if n not in ("k_a", "k_b") and (abs(v - lo) < 1e-3 * (hi - lo) or abs(v - hi) < 1e-3 * (hi - lo))]
print("\nParameters on a search bound:", on_bound if on_bound else "none")

# --- 4. Full continuous run: calibration + validation -------------------------
res = pdm.simulate(params, P, E, q0=q0)
sim = pd.Series(pdm.to_m3s(res["q"], AREA_KM2), index=data.index, name="q_sim")

def nse(o, s):
    return 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()

def kge(o, s):
    """Kling-Gupta efficiency (Gupta et al. 2009)."""
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)

# Event metrics: simulated peak within +/- 24 h of each observed POT peak
ev_rows = []
for _, e in events.iterrows():
    if e["period"] == "warmup":
        continue
    w = sim[e["date"] - pd.Timedelta("24h"): e["date"] + pd.Timedelta("24h")]
    ev_rows.append({"date": e["date"], "period": e["period"], "q_obs": e["q"],
                    "q_sim": w.max(),
                    "peak_error_%": 100 * (w.max() - e["q"]) / e["q"],
                    "timing_error_h": (w.idxmax() - e["date"]) / pd.Timedelta("1h")})
ev = pd.DataFrame(ev_rows)

rows = []
for per in ["calibration", "validation"]:
    m = period == per
    o, s = Q[m], sim.values[m]
    e = ev[ev["period"] == per]
    rows.append({"period": per, "NSE": nse(o, s), "KGE": kge(o, s),
                 "bias_%": 100 * (s.mean() - o.mean()) / o.mean(),
                 "baseflow_share": res["qb"][m].sum() / res["q"][m].sum(),
                 "n_events": len(e),
                 "median_abs_peak_error_%": e["peak_error_%"].abs().median(),
                 "median_abs_timing_error_h": e["timing_error_h"].abs().median()})
perf = pd.DataFrame(rows).set_index("period")
print("\nPerformance (continuous hourly simulation):")
print(perf.round(3).to_string())
print("\nFive largest events:")
print(ev.sort_values("q_obs", ascending=False).head(5).to_string(index=False, float_format="%.1f"))

# --- 5. Figure: four largest events -------------------------------------------
top = ev.sort_values("q_obs", ascending=False).head(4).sort_values("date")
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, (_, e) in zip(axes.ravel(), top.iterrows()):
    win = slice(e["date"] - pd.Timedelta("4D"), e["date"] + pd.Timedelta("5D"))
    ax.plot(data.loc[win].index, data.loc[win, "q"], color="#1f2a44", lw=1.5, label="Observed")
    ax.plot(sim.loc[win].index, sim.loc[win], color="#c8553d", lw=1.3, ls="--", label="PDM")
    axr = ax.twinx()                                   # rainfall hangs from the top
    axr.bar(data.loc[win].index, data.loc[win, "rain_gear"], width=1 / 24, color="#3b6fb6", alpha=0.5)
    axr.set_ylim(data.loc[win, "rain_gear"].max() * 4, 0)
    axr.set_ylabel("Rain (mm/h)", fontsize=8)
    ax.set_title(f"{e['date']:%d %b %Y} ({e['period']}) - obs {e['q_obs']:.0f}, "
                 f"PDM {e['q_sim']:.0f} m³/s", fontsize=9)
    ax.set_ylabel("Flow (m³/s)")
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.spines[["top"]].set_visible(False)
axes[0, 0].legend(frameon=False, fontsize=8, loc="center left")
fig.suptitle("27035 Aire at Kildwick - calibrated PDM, four largest events 2006-2016", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "pdm_calibrated_events.png", dpi=200)

# --- 6. Save --------------------------------------------------------------------
with open(WORK / "pdm_params_27035.json", "w") as fh:
    json.dump({"gauge": "27035", "objective": "NSE", "calibration": "2006-10-01/2011-09-30",
               "params": params}, fh, indent=2)
runs.to_csv(WORK / "pdm_calibration_seeds.csv")
perf.to_csv(WORK / "pdm_performance.csv")
ev.to_csv(WORK / "pdm_event_metrics.csv", index=False)
sim.to_frame().join(data[["q", "rain_gear"]]).to_csv(WORK / "pdm_simulation_27035.csv")
print(f"\nSaved parameters, metrics and simulation to {WORK}")
