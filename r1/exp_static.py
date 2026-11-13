"""A7 frontier/decomposition, A8 Theorem-4 tightness, A1 random same-second permutations, A2 session timeouts."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import r1_lib as L  # noqa: E402

R = L.R
FAM = L.FAMILIES
CAPS = L.CAPS


def all_candidate_metrics(tl, acts, states):
    rows = []
    for i in range(250):
        m = R.evaluate(acts[:, i], tl, states[:, i] if i < 50 else None)
        rows.append({"index": i, "id": L.BANK[i]["id"], "family": L.BANK[i]["family"],
                     "params": L.BANK[i]["parameters"], **{k: m[k] for k in
                     ("sr_action", "far_direct", "fdr_hard", "mfa_benign", "mfa_attack", "proxy_cost")}})
    return pd.DataFrame(rows)


def pareto(df):
    d = df.sort_values(["mfa_benign", "sr_action"])
    best, keep = np.inf, []
    for _, r in d.iterrows():
        if r.sr_action < best - 1e-15:
            keep.append(r["index"])
            best = r.sr_action
    return keep


def a7_frontier(val, test, va, vs, ta, ts):
    tv = all_candidate_metrics(val, va, vs)
    tt = all_candidate_metrics(test, ta, ts)
    tv.to_csv(L.OUT / "A7_all_candidates_validation.csv", index=False)
    tt.to_csv(L.OUT / "A7_all_candidates_test.csv", index=False)
    out = {}
    for name, df in (("validation", tv), ("test", tt)):
        ok = df[(df.far_direct <= CAPS["direct_attack_allow_max"] + 1e-12) & (df.fdr_hard <= CAPS["hard_benign_deny_max"] + 1e-12)]
        front = pareto(ok)
        fr = ok[ok["index"].isin(front)].sort_values("mfa_benign")
        out[name] = {"safe_candidates": int(len(ok)), "global_front": fr[["id", "family", "mfa_benign", "sr_action", "far_direct"]].to_dict("records"),
                     "front_family_counts": fr.family.value_counts().to_dict()}
        # lower envelope: minimal SR per family under benign-MFA levels
        env = []
        for lvl in [0.010, 0.0125, 0.014, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.05]:
            row = {"mfa_level": lvl}
            for f in FAM:
                sub = ok[(ok.family == f) & (ok.mfa_benign <= lvl + 1e-12)]
                row[f] = float(sub.sr_action.min()) if len(sub) else None
                row[f + "_far"] = float(sub.loc[sub.sr_action.idxmin(), "far_direct"]) if len(sub) else None
            env.append(row)
        out[name]["envelope"] = env
    # decomposition at the selected hysteresis operating point
    h = L.static_index(0.02, "hysteresis")
    th = tt.iloc[h]
    single_ok = tt[(tt.family == "single") & (tt.mfa_benign <= th.mfa_benign + 1e-12)]
    base = R.evaluate(test["base_action"], test)
    out["decomposition_at_selected_hysteresis"] = {
        "base": {k: base[k] for k in ("sr_action", "far_direct", "mfa_benign")},
        "hysteresis": th[["sr_action", "far_direct", "mfa_benign"]].to_dict(),
        "best_single_threshold_with_no_more_benign_MFA": single_ok.sort_values("sr_action").iloc[0][["id", "params", "sr_action", "far_direct", "mfa_benign"]].to_dict() if len(single_ok) else None,
        "always_low_conservative_fusion_only": tt.iloc[0][["sr_action", "far_direct", "mfa_benign"]].to_dict(),
    }
    L.save("A7_frontier.json", out)
    return out


def a8_theorem(test, ts):
    h = L.static_index(0.02, "hysteresis")
    lo, hi = L.BANK[h]["parameters"]["low"], L.BANK[h]["parameters"]["high"]
    delta = hi - lo
    x, gid, s = test["trust"], test["group"], ts[:, h].astype(int)
    rows = []
    for gg in np.unique(gid):
        m = gid == gg
        xs, ss = x[m], s[m]
        if len(xs) < 2:
            continue
        vn = float(np.abs(np.diff(xs)).sum())
        kh = int(np.count_nonzero(np.diff(ss)))
        rows.append({"group": int(gg), "events": int(m.sum()), "V_n": vn, "bound": 1 + vn / delta, "K_h": kh,
                     "single_threshold_flips_at_high": int(np.count_nonzero(np.diff((xs > hi).astype(int))))})
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "A8_theorem4_per_proxy.csv", index=False)
    res = {"thresholds": [lo, hi], "delta": delta, "proxies": len(df), "sum_V_n": df.V_n.sum(),
           "sum_bound": df.bound.sum(), "sum_K_h": int(df.K_h.sum()), "ratio_K_over_bound": df.K_h.sum() / df.bound.sum(),
           "median_ratio_per_proxy": float((df.K_h / df.bound).median()),
           "sum_single_threshold_flips_same_high": int(df.single_threshold_flips_at_high.sum())}
    L.save("A8_theorem4.json", res)
    return res


SELECTED = [(b, f) for b in (0.02, 0.05) for f in FAM]


def eval_selected(tl, acts, states):
    out = {"base": R.evaluate(tl["base_action"], tl)}
    for b, f in SELECTED:
        i = L.static_index(b, f)
        out[f"rho{b:g}_{f}"] = R.evaluate(acts[:, i], tl, states[:, i] if i < 50 else None)
    return out


def a1_permutations(test, n_perm=20):
    rows = []
    for p in range(n_perm):
        rng = np.random.default_rng(1000 + p)
        key = rng.random(len(test["label"]))
        perm = np.lexsort((key, test["timestamp"], test["group"]))
        tl = {k: v[perm] for k, v in test.items()}
        acts, states = L.actions_for(tl)
        for name, m in eval_selected(tl, acts, states).items():
            rows.append({"perm": p, "policy": name, **{k: m[k] for k in ("sr_action", "far_direct", "mfa_benign", "fdr_hard")}})
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "A1_random_tie_permutations.csv", index=False)
    summ = df.groupby("policy")[["sr_action", "far_direct", "mfa_benign"]].agg(["mean", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(L.OUT / "A1_random_tie_summary.csv", index=False)
    return summ


def sessionize(tl, tau):
    g, t = tl["group"], tl["timestamp"]
    new = np.r_[True, (g[1:] != g[:-1]) | ((t[1:] - t[:-1]) > tau)]
    return np.cumsum(new) - 1


def a2_timeouts(val, test, taus=(1, 5, 30, 60, 300, 1800, None)):
    rows = []
    for tau in taus:
        vtl, ttl = dict(val), dict(test)
        if tau is not None:
            vtl["group"] = sessionize(val, tau)
            ttl["group"] = sessionize(test, tau)
        va, vs = L.actions_for(vtl)
        ta, ts = L.actions_for(ttl)
        sessions = int(len(np.unique(ttl["group"])))
        for b in L.BUDGETS:
            for fi, f in enumerate(FAM):
                vm = [R.evaluate(va[:, i], val) for i in range(50 * fi, 50 * fi + 50)]   # SR on original proxy edges
                allowed = [i for i, m in zip(range(50 * fi, 50 * fi + 50), vm) if R.feasible(m, b, CAPS)]
                if not allowed:
                    rows.append({"tau_s": tau, "budget": b, "family": f, "status": "NO_SELECTION", "test_sessions": sessions})
                    continue
                ch = min(allowed, key=lambda i: R.rank(vm[i - 50 * fi], L.BANK[i]["id"]))
                tm = R.evaluate(ta[:, ch], test)                     # original proxy adjacency
                tw = R.evaluate(ta[:, ch], ttl)                      # within-session adjacency only
                rows.append({"tau_s": tau, "budget": b, "family": f, "status": "SELECTED", "candidate": L.BANK[ch]["id"],
                             "test_sessions": sessions, "sr_action": tm["sr_action"], "sr_within_session": tw["sr_action"],
                             "far_direct": tm["far_direct"], "fdr_hard": tm["fdr_hard"], "mfa_benign": tm["mfa_benign"],
                             "test_caps_met": R.feasible(tm, b, CAPS)})
        bm = R.evaluate(test["base_action"], test)
        rows.append({"tau_s": tau, "budget": None, "family": "base", "status": "BASE", "test_sessions": sessions,
                     "sr_action": bm["sr_action"], "far_direct": bm["far_direct"], "mfa_benign": bm["mfa_benign"]})
        print("tau", tau, "sessions", sessions, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "A2_session_timeouts.csv", index=False)
    return df


if __name__ == "__main__":
    t0 = time.time()
    val, test = L.load()
    va, vs = L.actions_for(val)
    ta, ts = L.actions_for(test)
    print("A7", a7_frontier(val, test, va, vs, ta, ts)["test"]["front_family_counts"], f"{time.time()-t0:.0f}s", flush=True)
    print("A8", a8_theorem(test, ts), f"{time.time()-t0:.0f}s", flush=True)
    print(a1_permutations(test).round(5).to_string(), f"{time.time()-t0:.0f}s", flush=True)
    a2_timeouts(val, test)
    print("done", f"{time.time()-t0:.0f}s")
