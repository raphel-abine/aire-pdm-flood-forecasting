# =============================================================================
# Step 06 - Rating-curve review, 27035 Aire at Kildwick Bridge (velocity-area station)
# 1. Recover the rating applied to the record from the hourly level-flow pairs
#    as a multi-segment power law Q = a (h - h0)^b (ISO 18320 practice),
#    breakpoints chosen by least squares in log space
# 2. Link the rating breakpoints to the "shelf" seen on flood rising limbs
# 3. Extrapolation: how much of the flood record lies above the highest gauging,
#    and how much the Boxing Day 2015 peak changes under alternative extrapolations
# 4. Convert PDM forecasts to levels and verify threshold crossings
#    (POD / FAR / CSI) at each lead time
# =============================================================================
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from config import DATA_ROOT as DESKTOP, WORK, FIG, GAUGE, AREA_KM2
import pdm

LEADS = [1, 2, 3, 6, 12, 24]
LMAX = max(LEADS)

data = pd.read_csv(WORK / "kildwick_27035_hourly.csv", parse_dates=["date"]).set_index("date")
events = pd.read_csv(WORK / "kildwick_27035_events.csv", parse_dates=["date"])

# --- 0. Station metadata from CAMELS-GB hydrometry attributes ---------------
hm_path = next(p for p in DESKTOP.rglob("camels_gb_v2_hydrometry_attributes.csv")
               if WORK.name not in p.parts)
try:
    hm = pd.read_csv(hm_path, encoding="latin-1", dtype={"gauge_id": str})
except pd.errors.ParserError:
    # A few rows contain stray commas in the last (free-text) field:
    # merge the extra fields back into that column instead of dropping rows
    import csv
    with open(hm_path, encoding="latin-1", newline="") as fh:
        rows = list(csv.reader(fh))
    ncol = len(rows[0])
    fixed = [r[:ncol - 1] + [",".join(r[ncol - 1:])] if len(r) > ncol else r for r in rows[1:]]
    hm = pd.DataFrame(fixed, columns=rows[0]).replace({"": np.nan, "NaN": np.nan})
    for c in hm.columns:
        if c != "gauge_id":
            conv = pd.to_numeric(hm[c], errors="coerce")
            if conv.notna().sum() == hm[c].notna().sum():
                hm[c] = conv
st = hm.set_index("gauge_id").loc[GAUGE]
Q_GAUGED = float(st["max_gauging_flow"])
print(f"Highest gauged flow: {Q_GAUGED:.2f} m3/s")
unc_cols = [c for c in hm.columns if "uncert" in c.lower() or "gauging" in c.lower()]
print("Gauging / uncertainty attributes:")
print(st[unc_cols].to_string())

# --- 1. Recover the applied rating as a 3-segment power law ------------------
pairs = data.dropna(subset=["q", "level"])
u = pairs.groupby(pairs["level"].round(3))["q"].median()          # one flow per mm of stage
h_u, q_u = u.index.values, u.values

def fit_segment(h, q):
    """Fit log Q = log a + b log(h - h0); returns (a, h0, b) and SSE in log space."""
    f = lambda x, la, h0, b: la + b * np.log(np.clip(x - h0, 1e-6, None))
    try:
        p, _ = curve_fit(f, h, np.log(q), p0=[np.log(q.mean()), h.min() - 0.2, 1.8],
                         bounds=([-10, -2, 0.5], [10, h.min() - 1e-3, 4]), maxfev=5000)
    except RuntimeError:
        return None, np.inf
    return (np.exp(p[0]), p[1], p[2]), ((np.log(q) - f(h, *p)) ** 2).sum()

grid = np.arange(0.60, 3.25, 0.05)                 # candidate breakpoints (m)
best = (np.inf, None, None, None)
for i, b1 in enumerate(grid):
    for b2 in grid[i + 2:]:
        sse, segs = 0.0, []
        for lo, hi in [(-np.inf, b1), (b1, b2), (b2, np.inf)]:
            m = (h_u > lo) & (h_u <= hi)
            if m.sum() < 20:
                sse = np.inf
                break
            p, s = fit_segment(h_u[m], q_u[m])
            sse += s
            segs.append(p)
        if sse < best[0]:
            best = (sse, b1, b2, segs)
_, BK1, BK2, SEGS = best
BREAKS = [-np.inf, BK1, BK2, np.inf]

def rating(h):
    """Recovered rating: level (m) -> flow (m3/s)."""
    h = np.asarray(h, dtype=float)
    q = np.full_like(h, np.nan)
    for (a, h0, b), lo, hi in zip(SEGS, BREAKS[:-1], BREAKS[1:]):
        m = (h > lo) & (h <= hi)
        q[m] = a * np.clip(h[m] - h0, 0, None) ** b
    return q

Q_BREAK = rating(np.array([BK1, BK2]))
def inverse_rating(q):
    """Flow (m3/s) -> level (m), segment chosen by flow range."""
    q = np.asarray(q, dtype=float)
    h = np.full_like(q, np.nan)
    qlims = [-np.inf, Q_BREAK[0], Q_BREAK[1], np.inf]
    for (a, h0, b), lo, hi in zip(SEGS, qlims[:-1], qlims[1:]):
        m = (q > lo) & (q <= hi)
        h[m] = h0 + (np.clip(q[m], 0, None) / a) ** (1 / b)
    return h

rec_err = 100 * (rating(pairs["level"].values) / pairs["q"].values - 1)
seg_tab = pd.DataFrame([{"segment": i + 1,
                         "level_from_m": max(lo, pairs["level"].min()),
                         "level_to_m": min(hi, pairs["level"].max()),
                         "a": a, "h0_m": h0, "b": b}
                        for i, ((a, h0, b), lo, hi) in enumerate(zip(SEGS, BREAKS[:-1], BREAKS[1:]))])
print("\nRecovered rating, Q = a (h - h0)^b:")
print(seg_tab.round(3).to_string(index=False))
print(f"Recovery error vs published flows: median {np.median(np.abs(rec_err)):.2f}%, "
      f"95th percentile {np.percentile(np.abs(rec_err), 95):.2f}%")
# A single rating across 2005-2016 means flow at a given stage should not drift by year
band = pairs[(pairs["level"] > 1.95) & (pairs["level"] <= 2.05)]
by_year = band.groupby(band.index.year)["q"].median()
print(f"Flow at 2.0 m stage by year: {by_year.min():.1f}-{by_year.max():.1f} m3/s "
      "(narrow range = one rating applied throughout)")

# --- 2. The rising-limb shelf vs the upper rating breakpoint -----------------
H_GAUGED = float(inverse_rating(np.array([Q_GAUGED]))[0])
shelf = []
for _, e in events.sort_values("q", ascending=False).head(6).iterrows():
    w = data.loc[e["date"] - pd.Timedelta("36h"): e["date"] - pd.Timedelta("3h"), "level"]
    flat = w[(w.diff().abs() < 0.03) & (w > 1.5)]          # rising limb hours with < 3 cm/h change
    shelf.append({"event": e["date"], "peak_level_m": e["level"], "peak_q": e["q"],
                  "shelf_hours": len(flat),
                  "shelf_level_m": flat.median() if len(flat) else np.nan})
shelf = pd.DataFrame(shelf)
print(f"\nUpper rating breakpoint: {BK2:.2f} m ({Q_BREAK[1]:.0f} m3/s)")
print("Rising-limb shelves in the six largest floods:")
print(shelf.to_string(index=False, float_format="%.2f"))

# --- 3. Extrapolation above the highest gauging --------------------------------
above = pairs["q"] > Q_GAUGED
print(f"\nHighest gauging {Q_GAUGED:.1f} m3/s is reached at {H_GAUGED:.2f} m stage")
print(f"Hours above the gauged range 2005-2016: {above.sum()}  "
      f"(floods: {', '.join(events.loc[events['q'] > Q_GAUGED, 'date'].dt.strftime('%b %Y'))})")

# Alternative extrapolations an independent reviewer would test:
#  A. as applied - top segment extended above the gauged range
#  B. one power law fitted to the whole gauged range (no floodplain segment)
#  C. the in-bank (middle) segment extended - lower bound if out-of-bank flow is
#     not gauged at this section (bypass / storage)
gauged = (h_u <= H_GAUGED)
pB, _ = fit_segment(h_u[gauged], q_u[gauged])
alt = {"A: as applied (top segment extended)": rating,
       "B: single power law, gauged range": lambda h: pB[0] * np.clip(np.asarray(h) - pB[1], 0, None) ** pB[2],
       "C: in-bank segment extended": lambda h: SEGS[1][0] * np.clip(np.asarray(h) - SEGS[1][1], 0, None) ** SEGS[1][2]}
big = events[events["q"] > Q_GAUGED].sort_values("q", ascending=False)
ext = pd.DataFrame({name: [float(f(np.array([lv]))[0]) for lv in big["level"]] for name, f in alt.items()},
                   index=big["date"].dt.strftime("%d %b %Y"))
ext.insert(0, "level_m", big["level"].values)
print("\nPeak flow (m3/s) of floods above the gauged range under each extrapolation:")
print(ext.round(1).to_string())
spread = 100 * (ext.iloc[:, 1:].min(axis=1) / ext.iloc[:, 1] - 1)
print("Lowest alternative vs applied rating (%):", spread.round(1).to_dict())

# --- 4. Level forecasts and threshold verification ----------------------------
# Thresholds: EA "typical range" upper level for Kildwick (1.60 m, published on the
# EA flood-information service) and the upper rating breakpoint (out-of-bank, inferred)
THRESH = {"EA typical-range high (1.60 m)": 1.60, f"Out-of-bank breakpoint ({BK2:.2f} m)": BK2}

params = json.load(open(WORK / "pdm_params_27035.json"))["params"]
errm = json.load(open(WORK / "pdm_error_model.json"))
phi = np.array(errm["phi"])
P, E, Q, H = data["rain_gear"].values, data["pe"].values, data["q"].values, data["level"].values
period, n = data["period"].values, len(Q)
res = pdm.simulate(params, P, E, q0=pdm.to_mmh(Q[0], AREA_KM2))
sim = pdm.to_m3s(res["q"], AREA_KM2)
err = np.log(Q) - np.log(sim)

orig = np.where(period == "validation")[0]
orig = orig[(orig + LMAX < n) & (orig >= LMAX)]

def ar_forecast(L):
    hist = [err[orig - k] for k in range(len(phi))]
    for _ in range(L):
        hist = [sum(phi[k] * hist[k] for k in range(len(phi)))] + hist[:-1]
    return hist[0]

def contingency(obs_hit, fc_hit):
    a = (obs_hit & fc_hit).sum(); b = (~obs_hit & fc_hit).sum(); c = (obs_hit & ~fc_hit).sum()
    return {"POD": a / (a + c) if a + c else np.nan, "FAR": b / (a + b) if a + b else np.nan,
            "CSI": a / (a + b + c) if a + b + c else np.nan, "obs_hours": int(a + c)}

ver = []
for L in LEADS:
    lv_obs = H[orig + L]
    lv_fc = {"Persistence": H[orig],
             "PDM + updating": inverse_rating(sim[orig + L] * np.exp(ar_forecast(L)))}
    for tname, thr in THRESH.items():
        for m, lv in lv_fc.items():
            ver.append({"lead_h": L, "threshold": tname, "method": m,
                        **contingency(lv_obs >= thr, lv >= thr)})
ver = pd.DataFrame(ver)
print("\nThreshold verification, validation period (hourly exceedance):")
print(ver.pivot_table(index="lead_h", columns=["threshold", "method"], values="CSI").round(2).to_string())

# --- 5. Figure -------------------------------------------------------------------
fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 2)

ax = fig.add_subplot(gs[0, 0])                                   # (a) rating curve
hh = np.linspace(0.2, 4.4, 400)
ax.scatter(q_u, h_u, s=4, color="#1f2a44", alpha=0.5, label="Published hourly level-flow pairs")
for (name, f), c, ls in zip(alt.items(), ["#c8553d", "#3b6fb6", "#e0a458"], ["-", "--", ":"]):
    m = hh >= (0.2 if name.startswith("A") else H_GAUGED)
    ax.plot(f(hh[m]), hh[m], color=c, lw=1.6, ls=ls, label=name)
ax.axhline(H_GAUGED, color="grey", lw=0.8, ls="--")
ax.text(2, H_GAUGED + 0.05, f"Highest gauging {Q_GAUGED:.0f} m³/s", fontsize=8, color="grey")
for bk in (BK1, BK2):
    ax.axhline(bk, color="#999999", lw=0.6, ls=":")
ax.text(2, BK2 - 0.18, f"Breakpoint {BK2:.2f} m", fontsize=8, color="#777777")
ax.set_xlabel("Flow (m³/s)"); ax.set_ylabel("Stage (m)")
ax.set_title("(a) Recovered rating and extrapolation options", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[0, 1])                                   # (b) shelves in stage
for _, e in events.sort_values("q", ascending=False).head(4).iterrows():
    w = data.loc[e["date"] - pd.Timedelta("30h"): e["date"] + pd.Timedelta("6h"), "level"]
    ax.plot((w.index - e["date"]) / pd.Timedelta("1h"), w.values, lw=1.5,
            label=f"{e['date']:%b %Y} (peak {e['q']:.0f} m³/s)")
ax.axhline(BK2, color="#c8553d", lw=1, ls="--", label=f"Upper rating breakpoint {BK2:.2f} m")
ax.axhline(H_GAUGED, color="grey", lw=0.8, ls="--", label="Stage of highest gauging")
ax.set_xlabel("Hours relative to peak"); ax.set_ylabel("Stage (m)")
ax.set_title("(b) Rising limbs stall near the upper breakpoint", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[1, 0])                                   # (c) extrapolation impact
x = np.arange(len(ext))
for k, (col, c) in enumerate(zip(ext.columns[1:], ["#c8553d", "#3b6fb6", "#e0a458"])):
    ax.bar(x + (k - 1) * 0.27, ext[col], width=0.27, color=c, label=col.split(":")[0])
ax.axhline(Q_GAUGED, color="grey", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels(ext.index, fontsize=8)
ax.set_ylabel("Peak flow (m³/s)")
ax.set_title("(c) Floods above the gauged range - peak by extrapolation option", fontsize=10)
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[1, 1])                                   # (d) threshold CSI
for (tname, m), g in ver.groupby(["threshold", "method"]):
    ax.plot(g["lead_h"], g["CSI"], marker="o", ms=4, lw=1.6,
            ls="-" if m.startswith("PDM") else ":",
            color="#c8553d" if tname.startswith("Out") else "#3b6fb6",
            label=f"{m} - {tname}")
ax.set_xscale("log"); ax.set_xticks(LEADS); ax.set_xticklabels(LEADS)
ax.set_ylim(0, 1); ax.set_xlabel("Lead time (h)"); ax.set_ylabel("Critical success index")
ax.set_title("(d) Forecasting threshold exceedance (level)", fontsize=10)
ax.legend(frameon=False, fontsize=7, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("27035 Aire at Kildwick Bridge - rating-curve review", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "rating_review.png", dpi=200)

# --- 6. Save --------------------------------------------------------------------
seg_tab.to_csv(WORK / "rating_segments_27035.csv", index=False)
shelf.to_csv(WORK / "rating_shelves_27035.csv", index=False)
ext.to_csv(WORK / "rating_extrapolation_27035.csv")
ver.to_csv(WORK / "level_threshold_verification.csv", index=False)
print(f"\nSaved rating tables and verification to {WORK}")
