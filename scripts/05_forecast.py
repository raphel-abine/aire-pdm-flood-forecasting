# =============================================================================
# Step 05 - PDM flood forecasts at 1, 2, 3, 6, 12 and 24 h lead times (hindcast)
# Forecast origin = every hour of the validation period (Oct 2011 - Sep 2016)
# Methods compared at each lead time:
#   Persistence          : forecast = flow observed at the origin (benchmark)
#   PDM open loop        : PDM with observed future rain ("perfect rainfall forecast")
#   PDM + updating       : open loop corrected by error prediction (Moore 2007):
#                          AR model of log-flow errors, fitted on calibration only
#   PDM + updating, no rain : same, but zero rainfall after the origin
#                          (worst-case rainfall forecast - a realistic lower bound)
# Skill: persistence index PI (Kitanidis & Bras 1980) - PI > 0 beats persistence
#        + peak error for forecasts issued L hours before each validation flood peak
# =============================================================================
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import WORK, FIG, AREA_KM2
import pdm

LEADS = [1, 2, 3, 6, 12, 24]
LMAX = max(LEADS)
AR_ORDERS = [1, 2, 3, 4]                    # candidate orders for the error model

# --- 1. Data, calibrated parameters and the continuous simulation -------------
data = pd.read_csv(WORK / "kildwick_27035_hourly.csv", parse_dates=["date"]).set_index("date")
events = pd.read_csv(WORK / "kildwick_27035_events.csv", parse_dates=["date"])
params = json.load(open(WORK / "pdm_params_27035.json"))["params"]
P, E, Q = data["rain_gear"].values, data["pe"].values, data["q"].values
period = data["period"].values
n = len(Q)

res = pdm.simulate(params, P, E, q0=pdm.to_mmh(Q[0], AREA_KM2))
sim = pdm.to_m3s(res["q"], AREA_KM2)
err = np.log(Q) - np.log(sim)               # multiplicative (log) error - suits flood flows
THR = np.quantile(Q, 0.99)                  # high-flow threshold (same as POT threshold)

def origins_for(name):
    """All forecast origins in a period with a full 24 h of observations ahead."""
    o = np.where(period == name)[0]
    return o[(o + LMAX < n) & (o >= LMAX)]

def pi_score(obs, fc, obs0):
    """Persistence index: 1 - SSE(forecast) / SSE(persistence)."""
    return 1 - ((obs - fc) ** 2).sum() / ((obs - obs0) ** 2).sum()

# --- 2. Error-prediction model: AR(p) on log errors, fitted on calibration ----
def fit_ar(p, idx):
    X = np.column_stack([err[idx - k] for k in range(1, p + 1)])
    phi, *_ = np.linalg.lstsq(X, err[idx], rcond=None)
    return phi

def ar_forecast(phi, orig, L):
    """Iterate the AR recursion L steps ahead from each origin (vectorised)."""
    hist = [err[orig - k] for k in range(len(phi))]          # hist[0] = error at origin
    for _ in range(L):
        nxt = sum(phi[k] * hist[k] for k in range(len(phi)))
        hist = [nxt] + hist[:-1]
    return hist[0]

# Choose the AR order by mean forecast skill over all leads - calibration period only
cal_orig = origins_for("calibration")
order_skill = {}
for p in AR_ORDERS:
    phi = fit_ar(p, cal_orig)
    order_skill[p] = np.mean([pi_score(Q[cal_orig + L],
                                       sim[cal_orig + L] * np.exp(ar_forecast(phi, cal_orig, L)),
                                       Q[cal_orig]) for L in LEADS])
p_best = max(order_skill, key=order_skill.get)
phi = fit_ar(p_best, cal_orig)
print("Mean calibration PI by AR order:", {k: round(v, 3) for k, v in order_skill.items()})
print(f"Selected AR({p_best}), coefficients: {np.round(phi, 4)}")

# --- 3. Zero-rainfall forecasts: restart the PDM from its states at each origin
# Rain that fell before the origin but is still within the time delay td is kept;
# everything after the origin is set to zero.
val_orig = origins_for("validation")
p0 = dict(params, td=0)                     # delay handled by shifting the input below
td = int(params["td"])
steps = np.arange(1, LMAX + 1)
zero_rain = np.zeros((len(val_orig), LMAX))
for i, t0 in enumerate(val_orig):
    st = {k: res[k][t0] for k in ("S", "S1", "S2", "Sg")}
    src = t0 + steps - td
    rain_in = np.where(src <= t0, P[src], 0.0)
    r = pdm.simulate(p0, rain_in, E[t0 + 1: t0 + LMAX + 1], states=st)
    zero_rain[i] = pdm.to_m3s(r["q"], AREA_KM2)

# Check: restarting with the observed rain must reproduce the continuous run exactly
t0 = val_orig[len(val_orig) // 2]
st = {k: res[k][t0] for k in ("S", "S1", "S2", "Sg")}
chk = pdm.simulate(p0, P[t0 + steps - td], E[t0 + 1: t0 + LMAX + 1], states=st)
print(f"Restart check (max difference, m3/s): {np.abs(pdm.to_m3s(chk['q'], AREA_KM2) - sim[t0 + 1: t0 + LMAX + 1]).max():.1e}")

# --- 4. Forecasts and skill at each lead time --------------------------------
fc = {}                                      # (method, L) -> forecast array over val_orig
rows = []
for L in LEADS:
    obs, obs0 = Q[val_orig + L], Q[val_orig]
    corr = np.exp(ar_forecast(phi, val_orig, L))
    fc[("Persistence", L)] = obs0
    fc[("PDM open loop", L)] = sim[val_orig + L]
    fc[("PDM + updating", L)] = sim[val_orig + L] * corr
    fc[("PDM + updating, no rain", L)] = zero_rain[:, L - 1] * corr
    hi = obs >= THR
    for m in ["Persistence", "PDM open loop", "PDM + updating", "PDM + updating, no rain"]:
        f = fc[(m, L)]
        rows.append({"lead_h": L, "method": m,
                     "PI_all": pi_score(obs, f, obs0),
                     "PI_high_flow": pi_score(obs[hi], f[hi], obs0[hi]),
                     "NSE": 1 - ((obs - f) ** 2).sum() / ((obs - obs.mean()) ** 2).sum(),
                     "RMSE_high_flow": np.sqrt(((obs[hi] - f[hi]) ** 2).mean())})
skill = pd.DataFrame(rows)
print("\nPersistence index, all hours (validation):")
print(skill.pivot(index="lead_h", columns="method", values="PI_all").round(3).to_string())
print("\nPersistence index, high flows >= %.0f m3/s (validation):" % THR)
print(skill.pivot(index="lead_h", columns="method", values="PI_high_flow").round(3).to_string())

# --- 5. Peak forecasts: issued L hours before each validation flood peak ----
pos = {t: i for i, t in enumerate(val_orig)}
val_events = events[events["period"] == "validation"]
prow = []
for _, e in val_events.iterrows():
    tp = data.index.get_loc(e["date"])
    for L in LEADS:
        if tp - L not in pos:
            continue
        i = pos[tp - L]
        for m in ["Persistence", "PDM open loop", "PDM + updating", "PDM + updating, no rain"]:
            prow.append({"event": e["date"], "lead_h": L, "method": m,
                         "peak_error_%": 100 * (fc[(m, L)][i] - e["q"]) / e["q"]})
peaks = pd.DataFrame(prow)
peak_tab = (peaks.assign(abs_err=peaks["peak_error_%"].abs())
            .pivot_table(index="lead_h", columns="method", values="abs_err", aggfunc="median"))
print(f"\nMedian absolute error in the forecast flood peak (%), {val_events.shape[0]} validation events:")
print(peak_tab.round(1).to_string())

# --- 6. Figure ----------------------------------------------------------------
COL = {"Persistence": "#7a7a7a", "PDM open loop": "#3b6fb6",
       "PDM + updating": "#c8553d", "PDM + updating, no rain": "#e0a458"}
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
for j, (col, title) in enumerate([("PI_all", "(a) Skill vs persistence - all hours"),
                                  ("PI_high_flow", f"(b) Skill vs persistence - flows >= {THR:.0f} m³/s")]):
    ax = fig.add_subplot(gs[0, j])
    for m, c in COL.items():
        s = skill[skill["method"] == m]
        ax.plot(s["lead_h"], s[col], marker="o", ms=4, lw=1.8, color=c, label=m)
    # Values below the axis floor (open loop at short leads) are written in, not hidden
    off = skill[(skill[col] < -1)]
    if len(off):
        txt = "Off scale (PDM open loop): " + ", ".join(
            f"{L} h = {v:.1f}" for L, v in zip(off["lead_h"], off[col]))
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=7.5, color=COL["PDM open loop"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(LEADS)
    ax.set_xticklabels(LEADS)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Persistence index")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if j == 0:
        ax.legend(frameon=False, fontsize=8, loc="lower right")

# (c) Boxing Day 2015: 24 h forecasts issued every 6 h before the peak
ax = fig.add_subplot(gs[1, :])
peak_t = pd.Timestamp("2015-12-26 17:00")
win = slice(peak_t - pd.Timedelta("36h"), peak_t + pd.Timedelta("30h"))
ax.plot(data.loc[win].index, data.loc[win, "q"], color="#1f2a44", lw=2.2, label="Observed", zorder=5)
for k, h_before in enumerate([24, 18, 12, 6]):
    t0 = data.index.get_loc(peak_t) - h_before
    i = pos[t0]
    times = data.index[t0 + 1: t0 + LMAX + 1]
    corr = np.exp(np.array([ar_forecast(phi, np.array([t0]), L)[0] for L in steps]))
    ax.plot(times, sim[t0 + 1: t0 + LMAX + 1] * corr, color=COL["PDM + updating"], lw=1.3,
            alpha=0.45 + 0.15 * k, label="PDM + updating" if k == 0 else None)
    ax.plot(times, zero_rain[i] * corr, color=COL["PDM + updating, no rain"], lw=1.3, ls="--",
            alpha=0.45 + 0.15 * k, label="PDM + updating, no rain" if k == 0 else None)
    ax.scatter(data.index[t0], Q[t0], color="black", s=18, zorder=6)
    ax.annotate(f"T-{h_before}h", (data.index[t0], Q[t0]), textcoords="offset points",
                xytext=(-6, 8), fontsize=8)
ax.set_ylabel("Flow (m³/s)")
ax.set_title("(c) Boxing Day 2015 - 24 h forecasts issued 24, 18, 12 and 6 h before the peak "
             "(dots = forecast origins)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("27035 Aire at Kildwick - PDM forecast skill, validation Oct 2011 - Sep 2016", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "forecast_skill.png", dpi=200)

# --- 7. Save ------------------------------------------------------------------
skill.to_csv(WORK / "pdm_forecast_skill.csv", index=False)
peaks.to_csv(WORK / "pdm_forecast_peak_errors.csv", index=False)
json.dump({"ar_order": p_best, "phi": phi.tolist(),
           "selection_mean_PI_calibration": order_skill},
          open(WORK / "pdm_error_model.json", "w"), indent=2)
print(f"\nSaved skill tables, peak errors and error model to {WORK}")
