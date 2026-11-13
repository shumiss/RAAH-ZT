"""C2: score-calibration shift [0,+12,-12,0] re-run with the revised fallback (start from / revert to the
validated configuration). Also re-runs the original configuration to check that the engine reproduces the
frozen score-shift rows. Outputs full-trace and per-quarter rows plus 16,384-event local benign-MFA maxima."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import r1_lib as L  # noqa: E402

R = L.R
SS = L.ORIG / "score_shift"
LOCK = json.loads((SS / "score_shift_protocol.lock.json").read_text(encoding="utf-8"))
BUDGETS = LOCK["budgets_mfa_benign"]


def quarter_rows(y, a, phase, prev, budget):
    ps = np.maximum(prev, 0)
    adj = prev >= 0
    sw = adj & (a != a[ps])
    out = []
    for k in range(4):
        m = phase == k
        b, t = m & (y == 0), m & (y == 1)
        far = np.count_nonzero(t & (a == 2)) / t.sum()
        fdr = np.count_nonzero(b & (a == 0)) / b.sum()
        mfa = np.count_nonzero(b & (a == 1)) / b.sum()
        out.append({"phase": k + 1, "sr_action": np.count_nonzero(m & sw) / np.count_nonzero(m & adj),
                    "far_direct": far, "fdr_hard": fdr, "mfa_benign": mfa,
                    "phase_caps_met": bool(far <= 0.005 + 1e-12 and fdr <= 0.05 + 1e-12 and mfa <= budget + 1e-12)})
    return out


def local_max(y, a, win=16384, step=4096):
    b = y == 0
    v = [np.mean(a[s:s + win][b[s:s + win]] == 1) for s in range(0, len(y) - win + 1, step)]
    v = np.asarray(v)
    return float(v.max()), float(np.mean(v > 0.02))


if __name__ == "__main__":
    val, test = L.load()
    n = len(test["label"])
    order = np.lexsort((test["event_id"], test["timestamp"]))
    offsets = np.asarray(LOCK["phase_offsets_on_trust_score_points"], float)
    phase = np.repeat(np.arange(4), n // 4)
    tr = np.empty(n)
    tr[order] = np.clip(test["trust"][order] + offsets[phase], 0, 100)
    shifted = {**test, "trust": tr}
    ta, ts = L.actions_for(shifted)
    g = R.global_order(shifted, ta, ts)
    assert np.array_equal(g["order"], order)
    cond = LOCK["adaptive_feedback_condition"]
    frozen = json.loads((SS / "score_shift_results.json").read_text(encoding="utf-8"))
    rows, phase_rows = [], []
    saved = {"order": order}
    for tag, init, fb in (("adaptive_original", "base", "retain"), ("adaptive_validated_fallback", "validated", "validated")):
        out, arrays, _, _ = L.run_dynamic(shifted, g, cond, initial=init, fallback=fb, budgets=BUDGETS)
        for r in out:
            a = arrays[(r["budget"], r["family"])]
            saved[f"{tag}_rho{r['budget']:g}_{r['family']}_global"] = a
            lm, share = local_max(g["label"], a)
            rows.append({"selection": tag, **r, "local_mfa_max": lm, "local_windows_over_2pct": share})
            for q in quarter_rows(g["label"], a, phase, g["previous"], r["budget"]):
                phase_rows.append({"selection": tag, "budget": r["budget"], "family": r["family"], **q})
    for b in BUDGETS:
        for f in L.FAMILIES:
            i = LOCK["fixed_selected_indices_from_unshifted_validation"][f"{b:g}"][f]
            m = R.evaluate(ta[:, i], shifted)
            a = ta[:, i][order]
            saved[f"fixed_rho{b:g}_{f}_global"] = a
            lm, share = local_max(g["label"], a)
            rows.append({"selection": "fixed", "budget": b, "family": f, "test_caps_met": R.feasible(m, b, L.CAPS),
                         **{k: m[k] for k in ("sr_action", "far_direct", "fdr_hard", "mfa_benign")},
                         "local_mfa_max": lm, "local_windows_over_2pct": share})
            for q in quarter_rows(g["label"], a, phase, g["previous"], b):
                phase_rows.append({"selection": "fixed", "budget": b, "family": f, **q})
    np.savez_compressed(L.OUT / "C2_score_shift_actions_global.npz", label_global=g["label"], **saved)
    d = pd.DataFrame(rows)
    # engine check against the frozen adaptive rows
    worst = 0.0
    for r in frozen["method_rows"]:
        if r["selection"].startswith("adaptive"):
            m = d[(d.selection == "adaptive_original") & (d.budget == r["budget_mfa_benign"]) & (d.family == r["family"])].iloc[0]
            worst = max(worst, abs(m.sr_action - r["full"]["sr_action"]), abs(m.mfa_benign - r["full"]["mfa_benign"]))
    print("engine check vs frozen score-shift rows:", worst)
    d.to_csv(L.OUT / "C2_score_shift_rows.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(L.OUT / "C2_score_shift_quarters.csv", index=False)
    show = d.copy()
    for c in ("sr_action", "far_direct", "mfa_benign", "local_mfa_max", "local_windows_over_2pct"):
        show[c] = (100 * show[c]).round(3)
    print(show[["selection", "budget", "family", "version_changes", "infeasible_updates", "test_caps_met", "sr_action",
                "far_direct", "mfa_benign", "local_mfa_max", "local_windows_over_2pct"]].to_string())
    p = pd.DataFrame(phase_rows)
    p = p[p.family.isin(["hysteresis", "sma", "ema"])]
    p["mfa_benign"] = (100 * p.mfa_benign).round(3); p["sr_action"] = (100 * p.sr_action).round(3)
    p["far_direct"] = (100 * p.far_direct).round(3)
    print(p.to_string())
