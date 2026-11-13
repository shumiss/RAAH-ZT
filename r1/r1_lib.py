"""Shared helpers for the R1 (IoT-76550) supplementary experiments.

Reuses an unchanged copy of the frozen runner (repro_check/run_experiment.py)
and adds a configurable delayed-feedback engine. With
initial="base", fallback="retain", selector="gp", estimator="raw" the engine is
the frozen dynamic_analysis; check_engine() verifies this numerically.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent / "frozen"
os.environ.setdefault("NUMBA_CACHE_DIR", str(REPRO / "nc"))
sys.path.insert(0, str(REPRO))
import run_experiment as R  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel  # noqa: E402

ORIG = Path(__file__).resolve().parent.parent / "frozen" / "original"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
PROTOCOL = json.loads((REPRO / "protocol.lock.json").read_text(encoding="utf-8"))
LOCK = json.loads((REPRO / "validation_selection.lock.json").read_text(encoding="utf-8"))
FAMILIES = R.FAMILIES
BUDGETS = PROTOCOL["budgets"]["benign_mfa_max_grid"]
CAPS = PROTOCOL["budgets"]
PARAMS = R.parameter_arrays(PROTOCOL["candidate_bank"])
BANK = PROTOCOL["candidate_bank"]


DATA = HERE.parent / "data"


def load():
    """Load the hashed timelines from ./data (size and SHA-256 are checked against the protocol)."""
    spec = {k: {**v, "path": str(DATA / Path(v["path"].replace("\\", "/")).name)}
            for k, v in PROTOCOL["inputs"].items()}
    return R.load_input(spec["validation"]), R.load_input(spec["test"])


def actions_for(tl):
    return R.build_actions(tl["base_action"], tl["trust"], tl["group"], *PARAMS)


def static_index(budget, family):
    return LOCK["selected"][f"{budget:g}"][family]["selected_index"]


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=float) + "\n",
                            encoding="utf-8")


# ----------------------------------------------------------------------------- selection helpers
def _gp_queries(coordinates, local_loss):
    queried = [0, 1, 2, 3]
    while len(queried) < 10:
        remaining = np.asarray([i for i in range(50) if i not in queried], dtype=np.int64)
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=np.full(coordinates.shape[1], 0.2), length_scale_bounds="fixed", nu=2.5) + \
            WhiteKernel(1e-5, noise_level_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-6, normalize_y=True, random_state=20261008)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(coordinates[queried], local_loss[queried])
            mean, std = gp.predict(coordinates[remaining], return_std=True)
        queried.append(int(remaining[np.argmin(mean - 2 * std)]))
    return queried


def choose(selector, indices, coordinates, sr, far, fdr, mfa, budget, rng):
    loss = sr[indices] + 200 * np.maximum(0.0, far[indices] - CAPS["direct_attack_allow_max"]) + \
        50 * np.maximum(0.0, fdr[indices] - CAPS["hard_benign_deny_max"]) + 5 * np.maximum(0.0, mfa[indices] - budget)
    if selector == "gp":
        queried = _gp_queries(coordinates, loss)
    elif selector == "all50":
        queried = list(range(50))
    elif selector == "random10":
        queried = sorted(rng.choice(50, 10, replace=False).tolist())
    elif selector == "greedy10":  # 4 anchors, then repeatedly the unqueried nearest neighbour of the best-so-far
        queried = [0, 1, 2, 3]
        while len(queried) < 10:
            best = queried[int(np.argmin(loss[queried]))]
            rem = [i for i in range(50) if i not in queried]
            d = np.linalg.norm(coordinates[rem] - coordinates[best], axis=1)
            queried.append(rem[int(np.argmin(d))])
    else:
        raise KeyError(selector)
    passed = [j for j in queried if far[indices[j]] <= CAPS["direct_attack_allow_max"] + 1e-12
              and fdr[indices[j]] <= CAPS["hard_benign_deny_max"] + 1e-12 and mfa[indices[j]] <= budget + 1e-12]
    if not passed:
        return None
    j = min(passed, key=lambda j: (sr[indices[j]], mfa[indices[j]], far[indices[j]], int(indices[j])))
    return int(indices[j])


def window_rates(window_actions, labels, estimator, eps):
    """Per-candidate FAR/FDR/benign-MFA from arrived labels; optional symmetric-flip correction."""
    oa = labels == 1
    ob = labels == 0
    if estimator == "raw":
        far = np.mean(window_actions[oa] == 2, axis=0)
        fdr = np.mean(window_actions[ob] == 0, axis=0)
        mfa = np.mean(window_actions[ob] == 1, axis=0)
        return far, fdr, mfa
    # corrected: invert the 2x2 symmetric-noise channel for every (action, class) count
    e = eps
    k = 1.0 - 2.0 * e

    def unmix(c_a, c_b):
        a = ((1 - e) * c_a - e * c_b) / k
        b = ((1 - e) * c_b - e * c_a) / k
        return np.maximum(a, 0.0), np.maximum(b, 0.0)

    na, nb = unmix(float(oa.sum()), float(ob.sum()))
    rates = []
    for act, cls in ((2, "a"), (0, "b"), (1, "b")):
        ca = np.count_nonzero(window_actions[oa] == act, axis=0).astype(float)
        cb = np.count_nonzero(window_actions[ob] == act, axis=0).astype(float)
        ta, tb = unmix(ca, cb)
        rates.append(ta / max(na, 1e-9) if cls == "a" else tb / max(nb, 1e-9))
    return rates[0], rates[1], rates[2]


def conservative_action(base):
    return np.where(base == 2, 1, base).astype(np.int8)


def run_dynamic(test, g, condition, *, initial="base", fallback="retain", selector="gp", estimator="raw",
                eps_assumed=None, budgets=None, families=None, seed=0, collect_active=False,
                feedback_seed=None, first_update=None):
    """Configurable version of R.dynamic_analysis.

    initial:   'base' (frozen) or 'validated' (validation-selected static candidate of the same family/cap)
    fallback:  'retain' keep incumbent when no queried candidate is feasible (frozen);
               'conservative' switch to base-Allow->MFA until a feasible commit;
               'validated' revert to the validation-selected static candidate
    """
    budgets = BUDGETS if budgets is None else budgets
    families = list(range(5)) if families is None else families
    spec = PROTOCOL["dynamic_search"]
    n = len(g["label"])
    a = g["actions"]
    csw = g["candidate_switches"]
    prev = g["previous"]
    present, flipped = R.feedback_mask(g["event_id"], condition,
                                       PROTOCOL["feedback_seed"] if feedback_seed is None else feedback_seed)
    arrived = np.full(n, -1, dtype=np.int8)
    coords = [np.asarray([BANK[50 * f + j]["coordinate"] for j in range(50)], float) for f in range(5)]
    idx = [np.arange(f * 50, (f + 1) * 50) for f in range(5)]
    B, F = len(budgets), len(families)
    init = np.full((B, F), -1, dtype=np.int32)
    if initial == "validated":
        for bi, b in enumerate(budgets):
            for fi, f in enumerate(families):
                s = static_index(b, FAMILIES[f])
                init[bi, fi] = -1 if s is None else s
    active = init.copy()
    version = np.zeros((B, F), dtype=np.int32)
    outputs = np.empty((n, B, F), dtype=np.int8)
    act_series = np.empty((n, B, F), dtype=np.int32) if collect_active else None
    cons = conservative_action(g["base"])
    rng = np.random.default_rng(seed)
    eps = condition["flip_probability"] if eps_assumed is None else eps_assumed
    infeasible_updates = np.zeros((B, F), dtype=np.int32)
    t_update = 0.0
    first, interval, window = spec["first_update_event"], spec["update_interval_events"], spec["window_events"]
    if first_update is not None:
        first = first_update
    for i in range(n):
        due = i - 1 - condition["delay_events"]
        if due >= 0 and present[due]:
            arrived[due] = int(g["label"][due]) ^ int(flipped[due])
        if i >= first and (i - first) % interval == 0:
            t0 = time.perf_counter()
            left = max(0, i - window)
            labels = arrived[left:i]
            nb, na = int((labels == 0).sum()), int((labels == 1).sum())
            support = nb >= spec["minimum_arrived_benign"] and na >= spec["minimum_arrived_attack"] and \
                (nb + na) / len(labels) >= spec["minimum_coverage"]
            if support:
                in_w = prev[left:i] >= left
                sr = csw[left:i][in_w].sum(axis=0) / int(in_w.sum())
                far, fdr, mfa = window_rates(a[left:i], labels, estimator, eps)
                for bi, b in enumerate(budgets):
                    for fi, f in enumerate(families):
                        chosen = choose(selector, idx[f], coords[f], sr, far, fdr, mfa, b, rng)
                        if chosen is None:
                            infeasible_updates[bi, fi] += 1
                            if fallback == "conservative":
                                chosen = -2
                            elif fallback == "validated":
                                chosen = int(init[bi, fi])
                            else:
                                chosen = int(active[bi, fi])
                        if chosen != active[bi, fi]:
                            active[bi, fi] = chosen
                            version[bi, fi] += 1
            t_update += time.perf_counter() - t0
        for bi in range(B):
            for fi in range(F):
                k = active[bi, fi]
                outputs[i, bi, fi] = g["base"][i] if k == -1 else (cons[i] if k == -2 else a[i, k])
        if collect_active:
            act_series[i] = active
    rows, arrays = [], {}
    for bi, b in enumerate(budgets):
        for fi, f in enumerate(families):
            grouped = np.empty(n, dtype=np.int8)
            grouped[g["order"]] = outputs[:, bi, fi]
            m = R.evaluate(grouped, test)
            rows.append({"condition": condition["name"], "budget": b, "family": FAMILIES[f], "initial": initial,
                         "fallback": fallback, "selector": selector, "estimator": estimator,
                         "version_changes": int(version[bi, fi]), "infeasible_updates": int(infeasible_updates[bi, fi]),
                         "test_caps_met": R.feasible(m, b, CAPS), **{k: m[k] for k in
                         ("sr_action", "far_direct", "fdr_hard", "mfa_benign", "mfa_attack", "proxy_cost", "action_switches")}})
            arrays[(b, FAMILIES[f])] = outputs[:, bi, fi].copy()
    return rows, arrays, act_series, t_update


def check_engine(test, g):
    """The frozen configuration must reproduce the saved clean_D256 and full_noisy_D256 rows."""
    saved = json.loads((ORIG / "adaptive_results.json").read_text(encoding="utf-8"))["dynamic_rows"] + \
        json.loads((ORIG / "adaptive_results_supplement.json").read_text(encoding="utf-8"))["dynamic_rows"]
    ref = {(r["feedback_condition"], r["budget_benign_mfa"], r["family"]): r for r in saved}
    conds = {c["name"]: c for c in PROTOCOL["feedback_conditions"]}
    conds["full_noisy_D256"] = {"name": "full_noisy_D256", "delay_events": 256, "coverage": 1.0, "flip_probability": 0.1}
    worst = 0.0
    for name in ("clean_D256", "full_noisy_D256"):
        rows, *_ = run_dynamic(test, g, conds[name])
        for r in rows:
            s = ref[(name, r["budget"], r["family"])]
            worst = max(worst, *(abs(r[k] - s[k]) for k in ("sr_action", "far_direct", "mfa_benign")),
                        abs(r["version_changes"] - s["version_changes"]))
    return worst
