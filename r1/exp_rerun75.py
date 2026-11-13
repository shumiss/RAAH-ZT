"""Re-run the 75 dynamic rows (5 feedback conditions x 3 budgets x 5 families) with the revised default:
start from the validation-selected static configuration and revert to it when no queried candidate is feasible.
Primary rows use raw window rates (no knowledge of the flip rate). For the two noisy conditions a sensitivity
variant with the flip-corrected estimator (assumed eps = true eps = 0.10) is also produced."""
from __future__ import annotations

import json
import sys
import time

import pandas as pd

sys.path.insert(0, ".")
import r1_lib as L  # noqa: E402

R = L.R
CONDS = list(L.PROTOCOL["feedback_conditions"]) + [
    {"name": "quarter_clean_D256", "delay_events": 256, "coverage": 0.25, "flip_probability": 0.0},
    {"name": "full_noisy_D256", "delay_events": 256, "coverage": 1.0, "flip_probability": 0.1}]

if __name__ == "__main__":
    t0 = time.time()
    val, test = L.load()
    ta, ts = L.actions_for(test)
    g = R.global_order(test, ta, ts)
    assert L.check_engine(test, g) == 0.0
    rows = []
    for c in CONDS:
        out, *_ = L.run_dynamic(test, g, c, initial="validated", fallback="validated", estimator="raw")
        for r in out:
            r["variant"] = "primary_raw"
        rows += out
        if c["flip_probability"] > 0:
            out, *_ = L.run_dynamic(test, g, c, initial="validated", fallback="validated", estimator="corrected")
            for r in out:
                r["variant"] = "sensitivity_corrected"
            rows += out
        print(c["name"], f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(L.OUT / "C1_dynamic75_validated_fallback.csv", index=False)
    # static reference rows for the same budgets
    st = []
    for b in L.BUDGETS:
        for f in L.FAMILIES:
            i = L.static_index(b, f)
            m = R.evaluate(ta[:, i], test) if i is not None else R.evaluate(test["base_action"], test)
            st.append({"budget": b, "family": f, "static_index": i, "test_caps_met": R.feasible(m, b, L.CAPS),
                       **{k: m[k] for k in ("sr_action", "far_direct", "fdr_hard", "mfa_benign")}})
    pd.DataFrame(st).to_csv(L.OUT / "C1_static_reference.csv", index=False)
    print("done", f"{time.time() - t0:.0f}s")
