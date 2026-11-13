"""A3 fallback designs, A4 label-noise curve (raw vs corrected estimator), A6 selector comparison, A5 adaptation decomposition."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import r1_lib as L  # noqa: E402

R = L.R


def cond(name, delay, coverage, flip):
    return {"name": name, "delay_events": delay, "coverage": coverage, "flip_probability": flip}


CONDS = [cond("clean_D0", 0, 1.0, 0.0), cond("clean_D256", 256, 1.0, 0.0), cond("quarter_clean_D256", 256, 0.25, 0.0),
         cond("full_noisy_D256", 256, 1.0, 0.1), cond("quarter_noisy_D256", 256, 0.25, 0.1)]


def a3_fallback(test, g):
    rows = []
    configs = [("base", "retain"), ("validated", "retain"), ("validated", "validated"), ("base", "conservative")]
    for c in CONDS:
        for init, fb in configs:
            r, *_ , tu = L.run_dynamic(test, g, c, initial=init, fallback=fb)
            rows += r
            print("A3", c["name"], init, fb, f"update_time={tu:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "A3_fallback.csv", index=False)
    return df


def a4_noise(test, g):
    rows = []
    for eps in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10):
        c = cond(f"full_flip{eps:g}_D256", 256, 1.0, eps)
        for init, est, assumed in (("base", "raw", None), ("validated", "raw", None), ("validated", "corrected", None)):
            r, *_ = L.run_dynamic(test, g, c, initial=init, estimator=est, eps_assumed=assumed, budgets=[0.02])
            for x in r:
                x.update({"flip": eps, "eps_assumed": eps if assumed is None else assumed})
            rows += r
            print("A4", eps, init, est, flush=True)
    for eps, assumed in ((0.05, 0.025), (0.05, 0.10), (0.10, 0.05)):
        c = cond(f"full_flip{eps:g}_D256", 256, 1.0, eps)
        r, *_ = L.run_dynamic(test, g, c, initial="validated", estimator="corrected", eps_assumed=assumed, budgets=[0.02])
        for x in r:
            x.update({"flip": eps, "eps_assumed": assumed})
        rows += r
        print("A4 misspecified", eps, assumed, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "A4_noise_curve.csv", index=False)
    return df


def a6_selectors(test, g):
    rows = []
    for c in (CONDS[0], CONDS[1]):
        for sel, seeds in (("gp", [0]), ("all50", [0]), ("greedy10", [0]), ("random10", [1, 2, 3, 4, 5])):
            for s in seeds:
                r, _, _, tu = L.run_dynamic(test, g, c, selector=sel, seed=s, budgets=[0.02])
                for x in r:
                    x.update({"seed": s, "update_time_s_all_families": tu})
                rows += r
                print("A6", c["name"], sel, s, f"{tu:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "A6_selectors.csv", index=False)
    return df


def a5_decomposition(test, g):
    c = CONDS[1]
    out = []
    for init in ("base", "validated"):
        rows, arrays, act, _ = L.run_dynamic(test, g, c, initial=init, budgets=[0.02], collect_active=True)
        prev = g["previous"]
        adj = prev >= 0
        ps = np.maximum(prev, 0)
        for fi, fam in enumerate(L.FAMILIES):
            a_out = arrays[(0.02, fam)]
            act_f = act[:, 0, fi]
            sw = adj & (a_out != a_out[ps])
            vb = adj & (act_f != act_f[ps])
            stat = L.static_index(0.02, fam)
            stat_actions = g["actions"][:, stat]
            sw_static = adj & (stat_actions != stat_actions[ps])
            first_commit = int(np.argmax(act_f != act_f[0])) if np.any(act_f != act_f[0]) else len(act_f)
            out.append({"initial": init, "family": fam, "pairs": int(adj.sum()),
                        "switches_total": int(sw.sum()), "switches_on_version_edges": int((sw & vb).sum()),
                        "version_edges": int(vb.sum()),
                        "sr_total": float(sw.sum() / adj.sum()), "sr_excluding_version_edges": float((sw & ~vb).sum() / (adj & ~vb).sum()),
                        "events_before_first_commit": first_commit,
                        "share_time_on_static_selection": float(np.mean(act_f == stat)),
                        "distinct_candidates_used": int(len(np.unique(act_f))),
                        "static_selection_sr_same_trace": float(sw_static.sum() / adj.sum())})
            print("A5", init, fam, flush=True)
    df = pd.DataFrame(out)
    df.to_csv(L.OUT / "A5_decomposition.csv", index=False)
    return df


if __name__ == "__main__":
    t0 = time.time()
    val, test = L.load()
    ta, ts = L.actions_for(test)
    g = R.global_order(test, ta, ts)
    which = sys.argv[1:] or ["a5", "a6", "a4", "a3"]
    if "a5" in which:
        a5_decomposition(test, g)
    if "a6" in which:
        a6_selectors(test, g)
    if "a4" in which:
        a4_noise(test, g)
    if "a3" in which:
        a3_fallback(test, g)
    print("done", f"{time.time()-t0:.0f}s")
