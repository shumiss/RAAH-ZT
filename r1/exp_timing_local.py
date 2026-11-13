"""A9 separated latency (per-event decision, per-event shadow update, periodic update) and A11 local benign-MFA windows."""
from __future__ import annotations

import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import r1_lib as L  # noqa: E402

R = L.R


def q(v):
    v = np.asarray(v) / 1000.0
    return {"n": int(len(v)), "p50_us": float(np.quantile(v, .5)), "p95_us": float(np.quantile(v, .95)),
            "p99_us": float(np.quantile(v, .99)), "max_us": float(v.max())}


def a9_timing(test):
    n = len(test["label"])
    trust, base, group = test["trust"], test["base_action"], test["group"]
    h = L.static_index(0.02, "hysteresis")
    lo, hi = L.BANK[h]["parameters"]["low"], L.BANK[h]["parameters"]["high"]
    sample = set(np.linspace(1024, n - 1, 4096, dtype=int).tolist())
    # (1) per-event decision: one selected threshold pair, dictionary state per proxy
    st, t1 = {}, []
    for i in range(n):
        if i in sample:
            s0 = time.perf_counter_ns()
        s = st.get(group[i], 0)
        x = trust[i]
        s = 1 if x > hi else (0 if x < lo else s)
        st[group[i]] = s
        out = 1 if base[i] == 2 and s == 0 else base[i]
        if i in sample:
            t1.append(time.perf_counter_ns() - s0)
    # (2) per-event shadow update of all 50 hysteresis candidates (NumPy vector per event)
    hl = np.asarray(L.PARAMS[0]); hh = np.asarray(L.PARAMS[1])
    states, t2 = {}, []
    for i in range(n):
        if i in sample:
            s0 = time.perf_counter_ns()
        sv = states.get(group[i])
        if sv is None:
            sv = np.zeros(50, dtype=np.int8)
            states[group[i]] = sv
        x = trust[i]
        sv[x > hh] = 1
        sv[x < hl] = 0
        outv = np.where((base[i] == 2) & (sv == 0), 1, base[i])
        if i in sample:
            t2.append(time.perf_counter_ns() - s0)
    # (3) compiled batch shadow update of all 250 candidates (amortized per event)
    R.build_actions(base[:1000], trust[:1000], group[:1000], *L.PARAMS)  # JIT warm-up
    s0 = time.perf_counter()
    acts, hs = R.build_actions(base, trust, group, *L.PARAMS)
    t3 = (time.perf_counter() - s0) / n * 1e6
    # (4) periodic update: window metrics for 250 candidates + GP-LCB for 5 families x 3 caps
    g = R.global_order(test, acts, hs)
    rng = np.random.default_rng(0)
    coords = [np.asarray([L.BANK[50 * f + j]["coordinate"] for j in range(50)], float) for f in range(5)]
    idx = [np.arange(f * 50, (f + 1) * 50) for f in range(5)]
    tm, tg, ta = [], [], []
    for i in range(16384, n, 16384 * 3)[:12]:
        left = i - 16384
        labels = g["label"][left:i].astype(np.int8)
        s0 = time.perf_counter()
        in_w = g["previous"][left:i] >= left
        sr = g["candidate_switches"][left:i][in_w].sum(axis=0) / int(in_w.sum())
        far, fdr, mfa = L.window_rates(g["actions"][left:i], labels, "raw", 0.0)
        s1 = time.perf_counter()
        for b in L.BUDGETS:
            for f in range(5):
                L.choose("gp", idx[f], coords[f], sr, far, fdr, mfa, b, rng)
        s2 = time.perf_counter()
        for b in L.BUDGETS:
            for f in range(5):
                L.choose("all50", idx[f], coords[f], sr, far, fdr, mfa, b, rng)
        s3 = time.perf_counter()
        tm.append(s1 - s0); tg.append(s2 - s1); ta.append(s3 - s2)
    res = {"runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                       "processor": platform.processor()},
           "per_event_decision_selected_pair": q(t1),
           "per_event_shadow_update_50_hysteresis_numpy": q(t2),
           "per_event_amortized_compiled_shadow_250_candidates_us": t3,
           "periodic_update_window_metrics_250_candidates_ms": float(np.mean(tm) * 1e3),
           "periodic_update_gp_lcb_5_families_3_caps_ms": float(np.mean(tg) * 1e3),
           "periodic_update_exhaustive_50_5_families_3_caps_ms": float(np.mean(ta) * 1e3),
           "update_interval_events": 4096,
           "note": "Python/NumPy on one workstation; engineering measurements, not deployment latency."}
    L.save("A9_timing.json", res)
    return res


def local_windows(label, actions_global, win=16384, step=4096, cap=0.02):
    benign = label == 0
    rates = []
    for s in range(0, len(label) - win + 1, step):
        b = benign[s:s + win]
        rates.append(np.mean(actions_global[s:s + win][b] == 1))
    r = np.asarray(rates)
    return {"windows": int(len(r)), "max_benign_mfa": float(r.max()), "p95_benign_mfa": float(np.quantile(r, .95)),
            "share_windows_above_cap": float(np.mean(r > cap)), "global_benign_mfa": float(np.mean(actions_global[benign] == 1))}


def a11_local(test):
    order = np.lexsort((test["event_id"], test["timestamp"]))
    y = test["label"][order]
    rows = []
    z = np.load(L.ORIG / "selected_test_actions.npz")
    for name, key in (("fixed_hysteresis", "rho0.02_hysteresis"), ("adaptive_hysteresis_cleanD256", "clean_D256_rho0.02_hysteresis"),
                      ("fixed_sma", "rho0.02_sma"), ("adaptive_sma_cleanD256", "clean_D256_rho0.02_sma"), ("base", "base")):
        rows.append({"setting": "unshifted", "policy": name, **local_windows(y, z[key][order])})
    zs = np.load(L.ORIG / "score_shift" / "score_shift_selected_actions.npz")
    for name in ("fixed_rho0.02_hysteresis", "adaptive_rho0.02_hysteresis", "fixed_rho0.02_sma", "adaptive_rho0.02_sma"):
        rows.append({"setting": "score_shift_0_+12_-12_0", "policy": name, **local_windows(y, zs[name][order])})
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "A11_local_mfa_windows.csv", index=False)
    return df


if __name__ == "__main__":
    val, test = L.load()
    print(a11_local(test).round(4).to_string(index=False), flush=True)
    import json
    print(json.dumps(a9_timing(test), indent=1))
