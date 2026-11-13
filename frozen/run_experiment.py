"""Exploratory replay of the original hysteresis question on frozen UNSW scores.

Every method and endpoint is fixed in protocol.lock.json. The test period was
previously inspected, so this script cannot produce confirmatory evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import linprog
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.lock.json"
FAMILIES = ("hysteresis", "single", "sma", "ema", "counter")
N_PER_FAMILY = 50
N_CANDIDATES = 250


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_input(spec: dict) -> dict[str, np.ndarray]:
    path = Path(spec["path"])
    assert path.is_file() and path.stat().st_size == spec["bytes"]
    assert sha256(path) == spec["sha256"], path
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert set(arrays) == {"event_id", "label", "group", "timestamp", "trust", "base_action"}
    n = len(arrays["label"])
    assert n > 0 and all(len(value) == n for value in arrays.values())
    assert np.all(np.isin(arrays["label"], [0, 1]))
    assert np.all(np.isin(arrays["base_action"], [0, 1, 2]))
    assert np.all(np.isfinite(arrays["trust"])) and np.all((arrays["trust"] >= 0) & (arrays["trust"] <= 100))
    assert np.all(np.diff(arrays["group"]) >= 0)
    within = arrays["group"][1:] == arrays["group"][:-1]
    assert np.all(arrays["timestamp"][1:][within] >= arrays["timestamp"][:-1][within])
    tied = within & (arrays["timestamp"][1:] == arrays["timestamp"][:-1])
    assert np.all(arrays["event_id"][1:][tied] >= arrays["event_id"][:-1][tied])
    assert len(np.unique(arrays["event_id"])) == n
    return arrays


def parameter_arrays(rows: list[dict]) -> tuple[np.ndarray, ...]:
    assert len(rows) == N_CANDIDATES
    for family_number, family in enumerate(FAMILIES):
        assert all(row["family"] == family for row in rows[family_number * 50:(family_number + 1) * 50])
    h_low = np.asarray([r["parameters"]["low"] for r in rows[0:50]], np.float64)
    h_high = np.asarray([r["parameters"]["high"] for r in rows[0:50]], np.float64)
    single_t = np.asarray([r["parameters"]["threshold"] for r in rows[50:100]], np.float64)
    sma_t = np.asarray([r["parameters"]["threshold"] for r in rows[100:150]], np.float64)
    sma_w = np.asarray([r["parameters"]["window"] for r in rows[100:150]], np.int64)
    ema_t = np.asarray([r["parameters"]["threshold"] for r in rows[150:200]], np.float64)
    ema_alpha = np.asarray([r["parameters"]["alpha"] for r in rows[150:200]], np.float64)
    counter_k = np.asarray([r["parameters"]["k"] for r in rows[200:250]], np.int64)
    assert len(np.unique(counter_k)) == 50 and set(counter_k.tolist()) == set(range(1, 51))
    return h_low, h_high, single_t, sma_t, sma_w, ema_t, ema_alpha, counter_k


@njit(cache=True)
def build_actions(base, trust, group, h_low, h_high, single_t, sma_t, sma_w, ema_t, ema_alpha, counter_k):
    n = len(base)
    result = np.empty((n, N_CANDIDATES), dtype=np.int8)
    hyst_state = np.empty((n, N_PER_FAMILY), dtype=np.int8)
    previous_group = -1
    group_start = 0
    running_sum = 0.0
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    h_states = np.zeros(N_PER_FAMILY, dtype=np.int8)
    ema_values = np.zeros(N_PER_FAMILY, dtype=np.float64)
    allow_streak = 0
    for i in range(n):
        score = trust[i]
        proposed = base[i]
        if group[i] != previous_group:
            previous_group = group[i]
            group_start = i
            running_sum = 0.0
            allow_streak = 0
            for j in range(N_PER_FAMILY):
                h_states[j] = 0
                ema_values[j] = score
        running_sum += score
        prefix[i + 1] = prefix[i] + score
        if proposed == 2:
            allow_streak += 1
        else:
            allow_streak = 0
        for j in range(N_PER_FAMILY):
            if score > h_high[j]:
                h_states[j] = 1
            elif score < h_low[j]:
                h_states[j] = 0
            hyst_state[i, j] = h_states[j]
            result[i, j] = 1 if proposed == 2 and h_states[j] == 0 else proposed

            single_high = score > single_t[j]
            result[i, 50 + j] = 1 if proposed == 2 and not single_high else proposed

            k = sma_w[j]
            available = i - group_start + 1
            if k > available:
                k = available
            average = (prefix[i + 1] - prefix[i + 1 - k]) / k
            result[i, 100 + j] = 1 if proposed == 2 and average <= sma_t[j] else proposed

            if i > group_start:
                ema_values[j] = ema_alpha[j] * score + (1.0 - ema_alpha[j]) * ema_values[j]
            result[i, 150 + j] = 1 if proposed == 2 and ema_values[j] <= ema_t[j] else proposed

            result[i, 200 + j] = 1 if proposed == 2 and allow_streak < counter_k[j] else proposed
    return result, hyst_state


@njit(cache=True)
def fixed_hysteresis_state(trust, group, low, high):
    state = np.empty(len(trust), dtype=np.int8)
    old_group = -1
    current = 0
    for i in range(len(trust)):
        if group[i] != old_group:
            old_group = group[i]
            current = 0
        if trust[i] > high:
            current = 1
        elif trust[i] < low:
            current = 0
        state[i] = current
    return state


def evaluate(actions: np.ndarray, timeline: dict, state: np.ndarray | None = None) -> dict:
    y = timeline["label"]
    group = timeline["group"]
    a = np.asarray(actions, dtype=np.int8)
    assert a.shape == y.shape and np.all(np.isin(a, [0, 1, 2]))
    good = y == 0
    attack = y == 1
    pairs = group[1:] == group[:-1]
    switches = int(np.count_nonzero(pairs & (a[1:] != a[:-1])))
    n = len(a)
    attack_allow = int(np.count_nonzero(attack & (a == 2)))
    good_deny = int(np.count_nonzero(good & (a == 0)))
    mfa = int(np.count_nonzero(a == 1))
    result = {
        "events": n,
        "benign": int(good.sum()),
        "attack": int(attack.sum()),
        "eligible_pairs": int(pairs.sum()),
        "action_switches": switches,
        "sr_action": switches / int(pairs.sum()),
        "far_direct": attack_allow / int(attack.sum()),
        "fdr_hard": good_deny / int(good.sum()),
        "mfa_benign": int(np.count_nonzero(good & (a == 1))) / int(good.sum()),
        "mfa_attack": int(np.count_nonzero(attack & (a == 1))) / int(attack.sum()),
        "strict_accuracy": int(np.count_nonzero(good & (a == 2)) + np.count_nonzero(attack & (a == 0))) / n,
        "permissive_accuracy_diagnostic": int(np.count_nonzero(good & (a != 0)) + np.count_nonzero(attack & (a != 2))) / n,
        "deny_count": int(np.count_nonzero(a == 0)),
        "mfa_count": mfa,
        "allow_count": int(np.count_nonzero(a == 2)),
        "proxy_cost": (10 * attack_allow + good_deny + 0.25 * mfa + 0.05 * switches) / n,
    }
    if state is not None:
        s = np.asarray(state, dtype=np.int8)
        assert s.shape == y.shape
        valid_state = pairs & (s[1:] >= 0) & (s[:-1] >= 0)
        state_pairs = int(valid_state.sum())
        state_switches = int(np.count_nonzero(valid_state & (s[1:] != s[:-1])))
        result["state_pairs"] = state_pairs
        result["state_switches"] = state_switches
        result["sr_state"] = state_switches / state_pairs if state_pairs else None
    return result


def feasible(row: dict, budget: float, caps: dict) -> bool:
    return row["far_direct"] <= caps["direct_attack_allow_max"] + 1e-12 and \
        row["fdr_hard"] <= caps["hard_benign_deny_max"] + 1e-12 and \
        row["mfa_benign"] <= budget + 1e-12


def rank(row: dict, candidate_id: str) -> tuple:
    return row["sr_action"], row["mfa_benign"], row["far_direct"], candidate_id


def solve_match(target: dict, simple: list[dict], row_ids: list[str]) -> dict:
    values = np.asarray([[r["sr_action"], r["far_direct"], r["mfa_benign"], r["fdr_hard"]] for r in simple], dtype=float)
    a_eq = np.vstack((np.ones(len(simple)), values[:, 1], values[:, 2]))
    b_eq = np.asarray([1.0, target["far_direct"], target["mfa_benign"]])
    solution = linprog(values[:, 0], A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs",
                       options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    if not solution.success:
        return {"status": "INFEASIBLE_OR_SOLVER_FAILURE", "solver_status": int(solution.status),
                "message": solution.message, "target_far": target["far_direct"],
                "target_mfa_benign": target["mfa_benign"]}
    x = np.asarray(solution.x)
    residual = float(np.max(np.abs(a_eq @ x - b_eq)))
    assert residual < 1e-8 and np.min(x) >= -1e-9
    nonzero = [(row_ids[i], float(w)) for i, w in enumerate(x) if w > 1e-9]
    return {"status": "PASS", "solver_status": 0, "maximum_equality_residual": residual,
            "weights": nonzero, "expected_sr": float(x @ values[:, 0]),
            "expected_far": float(x @ values[:, 1]), "expected_mfa_benign": float(x @ values[:, 2]),
            "expected_fdr_hard": float(x @ values[:, 3])}


def evaluate_frozen_mixture(weights: list[list], test_metrics: list[dict], row_ids: list[str]) -> dict:
    lookup = {name: row for name, row in zip(row_ids, test_metrics)}
    names = ("sr_action", "far_direct", "fdr_hard", "mfa_benign", "mfa_attack", "strict_accuracy", "proxy_cost")
    return {name: float(sum(float(weight) * lookup[cid][name] for cid, weight in weights)) for name in names}


def group_bootstrap(h_actions: np.ndarray, simple_actions: np.ndarray, timeline: dict,
                    reps: int, seed: int) -> dict:
    group = timeline["group"]
    start = np.r_[0, np.flatnonzero(group[1:] != group[:-1]) + 1]
    end = np.r_[start[1:], len(group)]
    pairs = end - start - 1
    h_sw = np.asarray([np.count_nonzero(h_actions[s + 1:e] != h_actions[s:e - 1]) for s, e in zip(start, end)])
    s_sw = np.asarray([np.count_nonzero(simple_actions[s + 1:e] != simple_actions[s:e - 1]) for s, e in zip(start, end)])
    observed = float(h_sw.sum() / pairs.sum() - s_sw.sum() / pairs.sum())
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(start), size=(reps, len(start)))
    denominator = pairs[samples].sum(axis=1)
    differences = h_sw[samples].sum(axis=1) / denominator - s_sw[samples].sum(axis=1) / denominator
    return {"clusters": len(start), "replicates": reps, "seed": seed,
            "observed_hysteresis_minus_simple_pp": 100 * observed,
            "descriptive_95_interval_pp": (100 * np.quantile(differences, [0.025, 0.975])).tolist(),
            "scope": "Paired source-IP proxy resampling on one previously inspected test period; no p-value or population guarantee."}


def lock_validation(protocol: dict, validation: dict, val_actions: np.ndarray,
                    val_h_state: np.ndarray) -> dict:
    """Write all static choices and matched mixtures before reading test labels."""
    path = HERE / "validation_selection.lock.json"
    if path.exists():
        locked = json.loads(path.read_text(encoding="utf-8"))
        assert locked["protocol_sha256"] == sha256(PROTOCOL_PATH)
        assert locked["runner_sha256"] == sha256(Path(__file__))
        return locked
    rows = protocol["candidate_bank"]
    caps = protocol["budgets"]
    candidate_metrics = [evaluate(val_actions[:, i], validation, val_h_state[:, i] if i < 50 else None)
                         for i in range(N_CANDIDATES)]
    base_metrics = evaluate(validation["base_action"], validation)
    selected = {}
    matched = {}
    simple_ids = [r["id"] for r in rows[50:]]
    for budget in caps["benign_mfa_max_grid"]:
        key = f"{budget:g}"
        selected[key] = {}
        for family_number, family in enumerate(FAMILIES):
            begin = 50 * family_number
            allowed = [i for i in range(begin, begin + 50) if feasible(candidate_metrics[i], budget, caps)]
            chosen = min(allowed, key=lambda i: rank(candidate_metrics[i], rows[i]["id"])) if allowed else None
            selected[key][family] = {"selected_index": chosen, "feasible_candidates": len(allowed)}
        h_index = selected[key]["hysteresis"]["selected_index"]
        matched[key] = solve_match(candidate_metrics[h_index], candidate_metrics[50:], simple_ids) if h_index is not None \
            else {"status": "NO_HYSTERESIS_SELECTION"}
    locked = {"status": "LOCKED_FROM_VALIDATION_BEFORE_TEST_LABEL_READ",
              "protocol_sha256": sha256(PROTOCOL_PATH), "runner_sha256": sha256(Path(__file__)),
              "validation_input_sha256": protocol["inputs"]["validation"]["sha256"],
              "candidate_validation_metrics": candidate_metrics, "base_validation": base_metrics,
              "selected": selected, "matched_lp": matched}
    save_json(path, locked)
    print("VALIDATION_SELECTION_LOCK", sha256(path), flush=True)
    return locked


def static_analysis(protocol: dict, validation: dict, test: dict,
                    val_actions: np.ndarray, test_actions: np.ndarray,
                    val_h_state: np.ndarray, test_h_state: np.ndarray,
                    validation_lock: dict) -> tuple[dict, dict[str, np.ndarray]]:
    rows = protocol["candidate_bank"]
    caps = protocol["budgets"]
    budgets = caps["benign_mfa_max_grid"]
    val_metrics = validation_lock["candidate_validation_metrics"]
    test_metrics = [evaluate(test_actions[:, i], test, test_h_state[:, i] if i < 50 else None)
                    for i in range(N_CANDIDATES)]
    val_base = validation_lock["base_validation"]
    test_base = evaluate(test["base_action"], test)
    output: dict = {"base_validation": val_base, "base_test": test_base,
                    "budget_rows": [], "matched_endpoint_rows": [], "bootstrap_rows": []}
    selected_arrays: dict[str, np.ndarray] = {"base": test["base_action"].copy()}
    simple_ids = [r["id"] for r in rows[50:]]
    for budget in budgets:
        decisions = []
        for family_number, family in enumerate(FAMILIES):
            begin = 50 * family_number
            allowed = [i for i in range(begin, begin + 50) if feasible(val_metrics[i], budget, caps)]
            frozen = validation_lock["selected"][f"{budget:g}"][family]
            selected = frozen["selected_index"]
            assert len(allowed) == frozen["feasible_candidates"]
            assert selected == (min(allowed, key=lambda i: rank(val_metrics[i], rows[i]["id"])) if allowed else None)
            record = {"budget_benign_mfa": budget, "family": family,
                      "validation_feasible_candidates": len(allowed),
                      "selection_status": "SELECTED" if selected is not None else "NO_FEASIBLE_VALIDATION_CANDIDATE"}
            if selected is not None:
                record.update({"candidate_id": rows[selected]["id"], "parameters": rows[selected]["parameters"],
                               "validation": val_metrics[selected], "test": test_metrics[selected],
                               "test_caps_met": feasible(test_metrics[selected], budget, caps)})
                selected_arrays[f"rho{budget:g}_{family}"] = test_actions[:, selected].copy()
            decisions.append(record)
            output["budget_rows"].append(record)
        hysteresis = decisions[0]
        if hysteresis["selection_status"] != "SELECTED":
            output["matched_endpoint_rows"].append({"budget_benign_mfa": budget, "status": "NO_HYSTERESIS_SELECTION"})
            output["bootstrap_rows"].append({"budget_benign_mfa": budget, "status": "NO_HYSTERESIS_SELECTION"})
            continue
        target_id = hysteresis["candidate_id"]
        h_index = next(i for i, row in enumerate(rows) if row["id"] == target_id)
        valid_lp = validation_lock["matched_lp"][f"{budget:g}"]
        oracle_lp = solve_match(test_metrics[h_index], test_metrics[50:], simple_ids)
        match_row = {"budget_benign_mfa": budget, "hysteresis_id": target_id,
                     "hysteresis_validation": val_metrics[h_index], "hysteresis_test": test_metrics[h_index],
                     "validation_matched_lp": valid_lp, "test_oracle_lp": oracle_lp}
        if valid_lp["status"] == "PASS":
            mixture_test = evaluate_frozen_mixture(valid_lp["weights"], test_metrics[50:], simple_ids)
            match_row["frozen_validation_mixture_test"] = mixture_test
            match_row["test_rate_shifts_mixture_minus_hysteresis"] = {
                name: mixture_test[name] - test_metrics[h_index][name]
                for name in ("far_direct", "mfa_benign", "fdr_hard", "sr_action")}
        output["matched_endpoint_rows"].append(match_row)

        simple_records = [row for row in decisions[1:] if row["selection_status"] == "SELECTED"]
        if feasible(val_base, budget, caps):
            simple_records.append({"family": "base", "candidate_id": "base", "validation": val_base,
                                   "test": test_base})
        if simple_records:
            best_simple = min(simple_records, key=lambda row: rank(row["validation"], row["candidate_id"]))
            simple_id = best_simple["candidate_id"]
            simple_actions = test["base_action"] if simple_id == "base" else test_actions[:, next(i for i, row in enumerate(rows) if row["id"] == simple_id)]
            output["bootstrap_rows"].append({"budget_benign_mfa": budget, "hysteresis_id": target_id,
                                             "best_simple_id_selected_on_validation": simple_id,
                                             **group_bootstrap(test_actions[:, h_index], simple_actions, test,
                                                               protocol["bootstrap"]["replicates"],
                                                               protocol["bootstrap"]["seed"])})
        else:
            output["bootstrap_rows"].append({"budget_benign_mfa": budget, "status": "NO_FEASIBLE_SIMPLE_VALIDATION_CANDIDATE"})

    fixed_val_state = fixed_hysteresis_state(validation["trust"], validation["group"], 40.0, 60.0)
    fixed_test_state = fixed_hysteresis_state(test["trust"], test["group"], 40.0, 60.0)
    const_test_state = fixed_hysteresis_state(np.full(len(test["trust"]), 50.0), test["group"], 40.0, 60.0)
    fixed_val = np.where((validation["base_action"] == 2) & (fixed_val_state == 0), 1, validation["base_action"]).astype(np.int8)
    fixed_test = np.where((test["base_action"] == 2) & (fixed_test_state == 0), 1, test["base_action"]).astype(np.int8)
    const_test = np.where((test["base_action"] == 2) & (const_test_state == 0), 1, test["base_action"]).astype(np.int8)
    legacy_test = np.asarray([[0, 1], [1, 2], [1, 2]], dtype=np.int8)[test["base_action"], fixed_test_state]
    output["diagnostics"] = {
        "base_no_stabilizer": {"validation": val_base, "test": test_base},
        "always_MFA": {"test": evaluate(np.ones(len(test["label"]), dtype=np.int8), test)},
        "constant_trust_50_safe_fixed_40_60": {"test": evaluate(const_test, test, const_test_state)},
        "fixed_hysteresis_40_60": {"validation": evaluate(fixed_val, validation, fixed_val_state),
                                    "test": evaluate(fixed_test, test, fixed_test_state)},
        "unsafe_legacy_fusion_40_60_offline_only": {"test": evaluate(legacy_test, test, fixed_test_state)},
    }
    selected_arrays.update({"fixed40_60": fixed_test, "constant50_safe": const_test,
                            "legacy_unsafe_offline": legacy_test})
    return output, selected_arrays


def global_order(timeline: dict, actions: np.ndarray, h_state: np.ndarray) -> dict:
    order = np.lexsort((timeline["event_id"], timeline["timestamp"]))
    groups = timeline["group"][order]
    group_count = int(groups.max()) + 1
    previous = np.empty(len(order), dtype=np.int64)
    last = np.full(group_count, -1, dtype=np.int64)
    for i, group in enumerate(groups):
        previous[i] = last[group]
        last[group] = i
    assert np.all((previous < 0) | (timeline["timestamp"][order][previous.clip(0)] <= timeline["timestamp"][order]))
    action_global = actions[order].copy()
    state_global = h_state[order].copy()
    switches = np.zeros_like(action_global, dtype=np.uint8)
    valid = np.flatnonzero(previous >= 0)
    switches[valid] = action_global[valid] != action_global[previous[valid]]
    return {"order": order, "group": groups, "previous": previous,
            "label": timeline["label"][order], "base": timeline["base_action"][order],
            "event_id": timeline["event_id"][order], "timestamp": timeline["timestamp"][order],
            "actions": action_global, "h_state": state_global, "candidate_switches": switches}


def feedback_mask(event_ids: np.ndarray, condition: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    present = np.empty(len(event_ids), dtype=np.bool_)
    flip = np.empty(len(event_ids), dtype=np.bool_)
    for i, event_id in enumerate(event_ids):
        key = f"{seed}|{condition['name']}|{int(event_id)}|"
        h = hashlib.sha256((key + "coverage").encode()).digest()
        present[i] = int.from_bytes(h[:8], "big") / 2**64 < condition["coverage"]
        h = hashlib.sha256((key + "noise").encode()).digest()
        flip[i] = present[i] and int.from_bytes(h[:8], "big") / 2**64 < condition["flip_probability"]
    return present, flip


def query_family(indices: np.ndarray, coordinates: np.ndarray, sr: np.ndarray,
                 far: np.ndarray, fdr: np.ndarray, mfa: np.ndarray,
                 budget: float, caps: dict) -> tuple[int | None, dict]:
    local_loss = sr[indices] + 200 * np.maximum(0.0, far[indices] - caps["direct_attack_allow_max"]) + \
        50 * np.maximum(0.0, fdr[indices] - caps["hard_benign_deny_max"]) + \
        5 * np.maximum(0.0, mfa[indices] - budget)
    queried = [0, 1, 2, 3]
    while len(queried) < 10:
        remaining = np.asarray([i for i in range(50) if i not in queried], dtype=np.int64)
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=np.full(coordinates.shape[1], 0.2), length_scale_bounds="fixed", nu=2.5) + \
            WhiteKernel(1e-5, noise_level_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-6,
                                      normalize_y=True, random_state=20261008)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(coordinates[queried], local_loss[queried])
            mean, std = gp.predict(coordinates[remaining], return_std=True)
        queried.append(int(remaining[np.argmin(mean - 2 * std)]))
    passed = [j for j in queried if far[indices[j]] <= caps["direct_attack_allow_max"] + 1e-12
              and fdr[indices[j]] <= caps["hard_benign_deny_max"] + 1e-12
              and mfa[indices[j]] <= budget + 1e-12]
    selected = min(passed, key=lambda j: (sr[indices[j]], mfa[indices[j]], far[indices[j]], int(indices[j]))) if passed else None
    log = {"queried_global_indices": [int(indices[j]) for j in queried],
           "queried_acquisition_losses": [float(local_loss[j]) for j in queried],
           "feasible_queried": len(passed), "selected_global_index": int(indices[selected]) if selected is not None else None}
    return int(indices[selected]) if selected is not None else None, log


def dynamic_analysis(protocol: dict, timeline: dict, global_data: dict) -> tuple[dict, dict[str, np.ndarray]]:
    n = len(global_data["label"])
    a = global_data["actions"]
    candidate_switches = global_data["candidate_switches"]
    previous = global_data["previous"]
    true_label = global_data["label"]
    budgets = protocol["budgets"]["benign_mfa_max_grid"]
    spec = protocol["dynamic_search"]
    coordinate = [np.asarray([protocol["candidate_bank"][50 * f + j]["coordinate"] for j in range(50)], dtype=float)
                  for f in range(5)]
    indices = [np.arange(f * 50, (f + 1) * 50, dtype=np.int64) for f in range(5)]
    all_results = []
    selected_arrays = {}
    for condition in protocol["feedback_conditions"]:
        print("DYNAMIC", condition["name"], flush=True)
        present, flipped = feedback_mask(global_data["event_id"], condition, protocol["feedback_seed"])
        arrived = np.full(n, -1, dtype=np.int8)
        active = np.full((len(budgets), len(FAMILIES)), -1, dtype=np.int16)
        version = np.zeros_like(active, dtype=np.int16)
        outputs = np.empty((n, len(budgets), len(FAMILIES)), dtype=np.int8)
        h_states = np.full((n, len(budgets)), -1, dtype=np.int8)
        audit_updates = []
        first = spec["first_update_event"]
        interval = spec["update_interval_events"]
        window = spec["window_events"]
        for i in range(n):
            due = i - 1 - condition["delay_events"]
            if due >= 0 and present[due]:
                arrived[due] = int(true_label[due]) ^ int(flipped[due])
            if i >= first and i % interval == 0:
                left = max(0, i - window)
                labels = arrived[left:i]
                benign = labels == 0
                attack = labels == 1
                nb = int(benign.sum())
                na = int(attack.sum())
                support = nb >= spec["minimum_arrived_benign"] and na >= spec["minimum_arrived_attack"] and \
                    (nb + na) / len(labels) >= spec["minimum_coverage"]
                row = {"decision_index": i, "window_left": left, "arrived_benign": nb,
                       "arrived_attack": na, "arrived_coverage": (nb + na) / len(labels),
                       "latest_arrived_decision_index": int(np.flatnonzero(arrived[:i] >= 0)[-1]) if np.any(arrived[:i] >= 0) else -1,
                       "supported": bool(support), "selections": []}
                assert row["latest_arrived_decision_index"] < i
                if support:
                    in_window = previous[left:i] >= left
                    eligible = int(in_window.sum())
                    assert eligible > 0
                    sr = candidate_switches[left:i][in_window].sum(axis=0) / eligible
                    window_actions = a[left:i]
                    far = np.mean(window_actions[attack] == 2, axis=0)
                    fdr = np.mean(window_actions[benign] == 0, axis=0)
                    mfa = np.mean(window_actions[benign] == 1, axis=0)
                    for b, budget in enumerate(budgets):
                        for f, family in enumerate(FAMILIES):
                            chosen, qlog = query_family(indices[f], coordinate[f], sr, far, fdr, mfa, budget,
                                                        protocol["budgets"])
                            before = int(active[b, f])
                            if chosen is not None and chosen != before:
                                active[b, f] = chosen
                                version[b, f] += 1
                            row["selections"].append({"budget": budget, "family": family,
                                                       "active_before": before, "active_after": int(active[b, f]),
                                                       "version_after": int(version[b, f]), **qlog})
                audit_updates.append(row)
            for b in range(len(budgets)):
                for f in range(len(FAMILIES)):
                    index = active[b, f]
                    outputs[i, b, f] = global_data["base"][i] if index < 0 else a[i, index]
                index = active[b, 0]
                if index >= 0:
                    h_states[i, b] = global_data["h_state"][i, index]
        assert np.all((outputs == global_data["base"][:, None, None]) |
                      ((global_data["base"][:, None, None] == 2) & (outputs == 1)))
        assert not np.any(arrived[n - 1 - condition["delay_events"] + 1:] >= 0)
        for b, budget in enumerate(budgets):
            for f, family in enumerate(FAMILIES):
                group_actions = np.empty(n, dtype=np.int8)
                group_actions[global_data["order"]] = outputs[:, b, f]
                group_states = None
                if f == 0:
                    group_states = np.empty(n, dtype=np.int8)
                    group_states[global_data["order"]] = h_states[:, b]
                metrics = evaluate(group_actions, timeline, group_states)
                all_results.append({"feedback_condition": condition["name"], "budget_benign_mfa": budget,
                                    "family": family, "updates": len(audit_updates),
                                    "supported_updates": sum(u["supported"] for u in audit_updates),
                                    "version_changes": int(version[b, f]),
                                    "final_active_candidate": int(active[b, f]),
                                    "test_caps_met": feasible(metrics, budget, protocol["budgets"]), **metrics})
                selected_arrays[f"{condition['name']}_rho{budget:g}_{family}"] = group_actions
        save_json(HERE / f"adaptive_updates_{condition['name']}.json", {
            "condition": condition, "events": n, "arrived_labels": int(np.count_nonzero(arrived >= 0)),
            "scheduled_coverage": int(present.sum()), "scheduled_flips": int(flipped.sum()),
            "updates": audit_updates})
    return {"dynamic_rows": all_results}, selected_arrays


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_BEFORE_THIS_REPLAY_BUT_AFTER_PRIOR_TEST_INSPECTION"
    assert sha256(Path(__file__)) == protocol["runner_sha256"]
    validation = load_input(protocol["inputs"]["validation"])
    params = parameter_arrays(protocol["candidate_bank"])
    val_actions, val_h_state = build_actions(validation["base_action"], validation["trust"], validation["group"], *params)
    frozen_validation = lock_validation(protocol, validation, val_actions, val_h_state)
    test = load_input(protocol["inputs"]["test"])
    print("INPUTS", len(validation["label"]), len(test["label"]), flush=True)
    test_actions, test_h_state = build_actions(test["base_action"], test["trust"], test["group"], *params)
    print("CANDIDATES", val_actions.shape, test_actions.shape, flush=True)
    static, static_arrays = static_analysis(protocol, validation, test, val_actions, test_actions,
                                            val_h_state, test_h_state, frozen_validation)
    save_json(HERE / "static_and_matched_results.json", static)
    del val_actions, val_h_state
    global_data = global_order(test, test_actions, test_h_state)
    print("GLOBAL_ORDER", len(global_data["order"]), flush=True)
    dynamic, dynamic_arrays = dynamic_analysis(protocol, test, global_data)
    save_json(HERE / "adaptive_results.json", dynamic)
    np.savez_compressed(HERE / "selected_test_actions.npz", **static_arrays, **dynamic_arrays)
    summary = {
        "status": "COMPLETE_EXPLORATORY", "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)), "input_sha256": {name: value["sha256"] for name, value in protocol["inputs"].items()},
        "validation_selection_sha256": sha256(HERE / "validation_selection.lock.json"),
        "validation_events": len(validation["label"]), "test_events": len(test["label"]),
        "test_source_ip_proxies": int(len(np.unique(test["group"]))),
        "test_same_second_adjacent_pairs": int(np.count_nonzero((test["group"][1:] == test["group"][:-1]) &
                                                                 (test["timestamp"][1:] == test["timestamp"][:-1]))),
        "candidate_count": N_CANDIDATES, "budget_rows": len(static["budget_rows"]),
        "matched_lp_rows": len(static["matched_endpoint_rows"]),
        "dynamic_rows": len(dynamic["dynamic_rows"]),
        "output_sha256": {name: sha256(HERE / name) for name in
                          ("static_and_matched_results.json", "adaptive_results.json", "selected_test_actions.npz")},
        "limits": "Previously inspected flow-label test and source-IP proxies; synthetic delayed/noisy label injection; no measured authentication completion, actual authorized delivery or independent confirmatory test.",
    }
    save_json(HERE / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
