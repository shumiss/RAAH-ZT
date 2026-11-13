"""B1 validation-selected frontier comparisons with source-IP cluster bootstrap CIs.
B2 replicated selector / adaptation comparison over feedback seeds and update phases."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import r1_lib as L  # noqa: E402

R = L.R
LEVELS = (0.0125, 0.014, 0.015, 0.0175, 0.02)


def cluster_boot(a_actions, b_actions, tl, reps=5000, seed=20261113):
    """Paired source-IP bootstrap of (A - B) for SR, direct FAR and benign MFA (percentage points)."""
    g, y = tl["group"], tl["label"]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    ends = np.r_[starts[1:], len(g)]

    def per_cluster(a):
        sw = np.array([np.count_nonzero(a[s + 1:e] != a[s:e - 1]) for s, e in zip(starts, ends)])
        pr = ends - starts - 1
        aa = np.array([np.count_nonzero((y[s:e] == 1) & (a[s:e] == 2)) for s, e in zip(starts, ends)])
        na = np.array([np.count_nonzero(y[s:e] == 1) for s, e in zip(starts, ends)])
        bm = np.array([np.count_nonzero((y[s:e] == 0) & (a[s:e] == 1)) for s, e in zip(starts, ends)])
        nb = np.array([np.count_nonzero(y[s:e] == 0) for s, e in zip(starts, ends)])
        return sw, pr, aa, na, bm, nb

    A, B = per_cluster(a_actions), per_cluster(b_actions)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(starts), size=(reps, len(starts)))

    def stat(X, ix):
        sw, pr, aa, na, bm, nb = (v[ix].sum(axis=-1) for v in X)
        return np.stack([sw / pr, aa / np.maximum(na, 1), bm / np.maximum(nb, 1)], axis=-1)

    full = 100 * (stat(A, np.arange(len(starts))) - stat(B, np.arange(len(starts))))
    boot = 100 * (stat(A, idx) - stat(B, idx))
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    names = ("sr", "far", "mfa")
    return {f"d_{n}_pp": float(full[k]) for k, n in enumerate(names)} | \
        {f"d_{n}_ci": [float(lo[k]), float(hi[k])] for k, n in enumerate(names)} | \
        {"share_boot_sr_below_0": float(np.mean(boot[:, 0] < 0))}


def b1(val, test, tag):
    va, vs = L.actions_for(val)
    ta, ts = L.actions_for(test)
    vm = [R.evaluate(va[:, i], val) for i in range(250)]
    tm = [R.evaluate(ta[:, i], test) for i in range(250)]
    rows = []
    for lvl in LEVELS:
        pick = {}
        for fi, f in enumerate(L.FAMILIES):
            allowed = [i for i in range(50 * fi, 50 * fi + 50) if R.feasible(vm[i], lvl, L.CAPS)]
            pick[f] = min(allowed, key=lambda i: R.rank(vm[i], L.BANK[i]["id"])) if allowed else None
        h = pick["hysteresis"]
        for f in L.FAMILIES[1:]:
            row = {"timeline": tag, "mfa_cap": lvl, "comparator": f,
                   "hyst_id": None if h is None else L.BANK[h]["id"], "comp_id": None if pick[f] is None else L.BANK[pick[f]]["id"]}
            if h is None or pick[f] is None:
                rows.append({**row, "status": "NO_SELECTION"})
                continue
            row.update({"status": "OK", "hyst_sr": tm[h]["sr_action"] * 100, "comp_sr": tm[pick[f]]["sr_action"] * 100,
                        "hyst_far": tm[h]["far_direct"] * 100, "comp_far": tm[pick[f]]["far_direct"] * 100,
                        "hyst_mfa": tm[h]["mfa_benign"] * 100, "comp_mfa": tm[pick[f]]["mfa_benign"] * 100,
                        "hyst_test_caps": R.feasible(tm[h], lvl, L.CAPS), "comp_test_caps": R.feasible(tm[pick[f]], lvl, L.CAPS),
                        **cluster_boot(ta[:, h], ta[:, pick[f]], test)})
            rows.append(row)
        print("B1", tag, lvl, flush=True)
    return rows


def start_feature_timelines(val, test):
    z = np.load(L.OUT / "A10_start_features" / "start_feature_timelines.npz")
    return ({**val, "trust": z["val_trust"], "base_action": z["val_base"]},
            {**test, "trust": z["test_trust"], "base_action": z["test_base"]})


def b2(test, g):
    rows = []
    reps = [("quarter_clean_seed", {"name": "quarter_clean_D256", "delay_events": 256, "coverage": 0.25, "flip_probability": 0.0},
             {"feedback_seed": 20261100 + k}) for k in range(10)]
    reps += [("full_clean_phase", {"name": "clean_D256", "delay_events": 256, "coverage": 1.0, "flip_probability": 0.0},
              {"first_update": 4096 + 512 * k}) for k in range(8)]
    reps += [("flip5_corrected_seed", {"name": "full_flip0.05_D256", "delay_events": 256, "coverage": 1.0, "flip_probability": 0.05},
              {"feedback_seed": 20261200 + k}) for k in range(5)]
    configs = [("gp", "base", "retain"), ("all50", "base", "retain"), ("greedy10", "base", "retain"),
               ("random10", "base", "retain"), ("gp", "validated", "validated"), ("all50", "validated", "validated")]
    for r_i, (kind, cond, kw) in enumerate(reps):
        for sel, init, fb in configs:
            est = "corrected" if kind.startswith("flip") else "raw"
            out, *_ = L.run_dynamic(test, g, cond, selector=sel, initial=init, fallback=fb, estimator=est,
                                    budgets=[0.02], seed=r_i, **kw)
            for x in out:
                x.update({"replicate_kind": kind, "replicate": r_i, "config": f"{sel}|{init}|{fb}"})
            rows += out
        print("B2", kind, r_i, flush=True)
        pd.DataFrame(rows).to_csv(L.OUT / "B2_replicated_selectors.csv", index=False)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    part = sys.argv[1]
    val, test = L.load()
    if part == "b1":
        rows = b1(val, test, "primary")
        v2, t2 = start_feature_timelines(val, test)
        rows += b1(v2, t2, "start_features")
        pd.DataFrame(rows).to_csv(L.OUT / "B1_frontier_bootstrap.csv", index=False)
    else:
        ta, ts = L.actions_for(test)
        g = R.global_order(test, ta, ts)
        assert L.check_engine(test, g) == 0.0
        b2(test, g)
    print("done", f"{time.time()-t0:.0f}s")
