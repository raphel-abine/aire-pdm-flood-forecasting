"""
pdm.py - Probability Distributed Model (PDM) for hourly flood forecasting
=========================================================================
Formulation after Moore (2007), "The PDM rainfall-runoff model",
Hydrology and Earth System Sciences 11(1), 483-499.

Configuration (the standard UK flood-forecasting set-up described by Moore 2007):
  * Soil moisture: store capacities follow a Pareto distribution
        F(c) = 1 - (1 - c/cmax)^b ,   cmin = 0,  Smax = cmax / (b + 1)
    Direct runoff is generated from the saturated fraction; the integrated
    (analytical) form is used, so runoff and storage change balance exactly.
  * Actual evaporation: Ea/Ep = 1 - ((Smax - S)/Smax)^be
  * Drainage to groundwater: d = (S - St)^bg / kg  for S > St
  * Fast (surface) pathway: two linear reservoirs in series (k1, k2)
  * Slow (baseflow) pathway: cubic store, qb = kb * Sg^3
  * Pure time delay td (hours) applied to the rainfall input

Numerical scheme (chosen so that mass is conserved to machine precision):
  * Linear reservoirs: exact solution for constant inflow over the step;
    the step-mean outflow of reservoir 1 is the inflow to reservoir 2.
  * Cubic store: implicit (backward Euler) step solved by Newton iteration;
    outflow volume = inflow volume - storage change.

Units: rainfall, PE, storage in mm; fluxes in mm/h; dt in hours.
Use to_m3s() to convert catchment-average mm/h to m3/s.
"""
import numpy as np

try:                                     # numba gives ~100x speed-up for calibration
    from numba import njit
    HAS_NUMBA = True
except ImportError:                      # pure-Python fallback (slow but identical)
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

# Parameter order used throughout. st is the drainage threshold as a FRACTION of Smax.
PARAM_NAMES = ["cmax", "b", "be", "kg", "st", "bg", "k1", "k2", "kb", "td"]


@njit(cache=True)
def _pdm_core(P, E, cmax, b, be, kg, st, bg, k1, k2, kb, td, dt, S0, S1_0, S2_0, Sg0):
    """Run the PDM over the whole series. Returns fluxes (mm/h) and states (mm)."""
    n = P.size
    td = int(td)
    smax = cmax / (b + 1.0)              # maximum storage of the soil-moisture store
    s_thr = st * smax                    # drainage threshold St (mm)
    e1 = np.exp(-dt / k1)
    e2 = np.exp(-dt / k2)

    q = np.zeros(n); qs = np.zeros(n); qb = np.zeros(n)
    ae = np.zeros(n); dr = np.zeros(n); p_used = np.zeros(n)
    S_out = np.zeros(n); S1_out = np.zeros(n); S2_out = np.zeros(n); Sg_out = np.zeros(n)

    S, S1, S2, Sg = S0, S1_0, S2_0, Sg0
    for t in range(n):
        p = P[t - td] if t >= td else 0.0          # delayed rainfall (mm/h)
        p_used[t] = p

        # --- losses from the soil-moisture store (evaporation + drainage) -----
        a = E[t] * (1.0 - ((smax - S) / smax) ** be)
        d = (S - s_thr) ** bg / kg if S > s_thr else 0.0
        loss = (a + d) * dt
        avail = S + p * dt
        if loss > avail and loss > 0.0:            # cannot lose more water than exists
            f = avail / loss
            a *= f
            d *= f
        pi = p - a - d                             # net input rate (mm/h)

        # --- direct runoff from the Pareto-distributed stores ---------------
        if pi > 0.0:
            c_star = cmax * (1.0 - (1.0 - S / smax) ** (1.0 / (b + 1.0)))   # critical capacity
            c_new = min(c_star + pi * dt, cmax)
            S_new = smax * (1.0 - (1.0 - c_new / cmax) ** (b + 1.0))
            v = pi * dt - (S_new - S)              # direct runoff volume (mm)
        else:
            S_new = max(S + pi * dt, 0.0)
            v = 0.0
        S = S_new
        u = v / dt                                 # inflow rate to the fast pathway

        # --- fast pathway: two linear reservoirs in series (exact solution) --
        S1_new = u * k1 + (S1 - u * k1) * e1
        o1 = (u * dt - (S1_new - S1)) / dt         # step-mean outflow of reservoir 1
        S1 = S1_new
        S2_new = o1 * k2 + (S2 - o1 * k2) * e2
        o2 = (o1 * dt - (S2_new - S2)) / dt        # step-mean outflow of reservoir 2
        S2 = S2_new

        # --- slow pathway: cubic store, implicit step (Newton) --------------
        rhs = Sg + d * dt
        x = Sg if Sg > 0.0 else rhs
        for _ in range(30):
            g = x + dt * kb * x ** 3 - rhs
            x_new = x - g / (1.0 + 3.0 * dt * kb * x * x)
            if x_new < 0.0:
                x_new = 0.5 * x
            if abs(x_new - x) < 1e-12:
                x = x_new
                break
            x = x_new
        ob = (rhs - x) / dt                        # step-mean baseflow (mm/h)
        Sg = x

        q[t] = o2 + ob; qs[t] = o2; qb[t] = ob
        ae[t] = a; dr[t] = d
        S_out[t] = S; S1_out[t] = S1; S2_out[t] = S2; Sg_out[t] = Sg
    return q, qs, qb, ae, dr, p_used, S_out, S1_out, S2_out, Sg_out


def simulate(params, P, E, dt=1.0, q0=None, states=None):
    """
    Run the PDM.
    params : dict with keys PARAM_NAMES
    P, E   : rainfall and potential evaporation (mm/h), 1-D arrays
    q0     : observed flow at the start (mm/h) - sets the groundwater store so
             that initial baseflow = q0 (used when no states are given)
    states : optional dict {S, S1, S2, Sg} to start from saved states (forecasting)
    Returns a dict of arrays (mm/h for fluxes, mm for states) + initial states.
    """
    p = {k: float(params[k]) for k in PARAM_NAMES}
    smax = p["cmax"] / (p["b"] + 1.0)
    if states is None:
        qb0 = 0.0 if q0 is None or not np.isfinite(q0) else max(float(q0), 0.0)
        states = {"S": 0.5 * smax, "S1": 0.0, "S2": 0.0, "Sg": (qb0 / p["kb"]) ** (1.0 / 3.0)}
    out = _pdm_core(np.asarray(P, dtype=np.float64), np.asarray(E, dtype=np.float64),
                    p["cmax"], p["b"], p["be"], p["kg"], p["st"], p["bg"],
                    p["k1"], p["k2"], p["kb"], p["td"], float(dt),
                    states["S"], states["S1"], states["S2"], states["Sg"])
    keys = ["q", "qs", "qb", "ae", "d", "p_used", "S", "S1", "S2", "Sg"]
    res = dict(zip(keys, out))
    res["init"] = dict(states)
    res["smax"] = smax
    return res


def water_balance(res, dt=1.0):
    """Closure (mm): rain in - evaporation - flow out - change in all four stores."""
    i = res["init"]
    d_store = ((res["S"][-1] - i["S"]) + (res["S1"][-1] - i["S1"])
               + (res["S2"][-1] - i["S2"]) + (res["Sg"][-1] - i["Sg"]))
    p_in = res["p_used"].sum() * dt
    e_out = res["ae"].sum() * dt
    q_out = res["q"].sum() * dt
    return {"rain": p_in, "evap": e_out, "flow": q_out, "d_storage": d_store,
            "closure": p_in - e_out - q_out - d_store}


def to_m3s(q_mmh, area_km2):
    """Catchment-average mm/h -> m3/s."""
    return np.asarray(q_mmh) * area_km2 / 3.6


def to_mmh(q_m3s, area_km2):
    """m3/s -> catchment-average mm/h."""
    return np.asarray(q_m3s) * 3.6 / area_km2
