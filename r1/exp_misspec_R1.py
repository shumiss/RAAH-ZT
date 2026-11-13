"""C3: misspecified flip rate for the corrected estimator under the revised fallback (hysteresis, 2% cap)."""
import sys

import pandas as pd

sys.path.insert(0, ".")
import r1_lib as L  # noqa: E402

R = L.R
if __name__ == "__main__":
    val, test = L.load()
    ta, ts = L.actions_for(test)
    g = R.global_order(test, ta, ts)
    rows = []
    for true, assumed in ((0.05, 0.025), (0.05, 0.05), (0.05, 0.10), (0.10, 0.05), (0.10, 0.10)):
        c = {"name": f"full_flip{true:g}_D256", "delay_events": 256, "coverage": 1.0, "flip_probability": true}
        out, *_ = L.run_dynamic(test, g, c, initial="validated", fallback="validated", estimator="corrected",
                                eps_assumed=assumed, budgets=[0.02], families=[0])
        for r in out:
            rows.append({"flip": true, "eps_assumed": assumed, **r})
        print(true, assumed, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(L.OUT / "C3_misspecified_flip_validated_fallback.csv", index=False)
    for c in ("sr_action", "far_direct", "mfa_benign"):
        d[c] = (100 * d[c]).round(3)
    print(d[["flip", "eps_assumed", "version_changes", "infeasible_updates", "test_caps_met", "sr_action", "far_direct", "mfa_benign"]].to_string())
