"""A10 look-ahead sensitivity: refit the purged HGB pipeline on connection-start features only.

Same split, boundary purge, seed, HGB settings, sigmoid calibration, base-action
rule (Allow p<1.25/5.75, Deny p>=1.75/3.25) and frozen event order/scopes as the
primary replay; only the model feature set changes. Excluded are fields that
summarise the completed flow (duration, byte/packet counts, loads, jitter, final
TCP state, response-side TTL/window, HTTP/FTP content and RTT measurements).
Retained connection-history counters (ct_*) count earlier connections.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WD = Path(os.environ.get("RAAH_UNSW_WORKDIR", "unsw_workdir"))  # see README, step A10
sys.path.insert(0, str(WD / "raah_revision" / "code"))
import run_unsw_temporal as original  # noqa: E402
from raah.modeling import BenignCentroidTrust, clean_features, make_evaluator  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1_lib as L  # noqa: E402

SEED = 20260924
CAL_END = 1424233663.0
START_NUMERIC = ["sport", "dsport", "sttl", "swin", "stcpb", "is_sm_ips_ports", "ct_srv_src", "ct_srv_dst",
                 "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"]
START_CATEGORICAL = ["proto", "service"]
OUTD = L.OUT / "A10_start_features"
OUTD.mkdir(exist_ok=True)


def fit_and_score():
    manifest, features, train_path, _ = original.load_inputs(original.MATERIALIZED_ROOT, original.FEATURE_SPEC, SEED,
                                                             verify_hashes=True)
    purged = json.loads((WD / "revision_completion" / "purged_hgb" / "purged_validation_manifest.json").read_text(encoding="utf-8"))
    train = pd.read_csv(train_path, low_memory=False)
    train = train.loc[train["__end_time"] <= manifest["train_end_time"]].copy()
    assert len(train) == 49999
    num, cat = START_NUMERIC, START_CATEGORICAL
    assert set(num) <= set(features["numeric"]) and set(cat) <= set(features["categorical"])
    x = clean_features(train, num, cat)
    y = train["__label"].to_numpy(dtype=np.int8)
    hashes = set(int(v) for v in pd.util.hash_pandas_object(x, index=False).to_numpy(dtype=np.uint64))
    model = make_evaluator("hgb", num, cat, seed=SEED, threads=8)
    with threadpool_limits(limits=8):
        model.fit(x, y)
        centroid = BenignCentroidTrust(num).fit(x, y)
        calib = original.load_calibration(purged, CAL_END, num, cat)
        assert len(calib) == 254056
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        cal.fit(clean_features(calib, num, cat), calib["__label"].to_numpy(dtype=np.int8))
        for split in ("val_tune", "test"):
            original.score_stream(split, purged, CAL_END, num, cat, model, cal, centroid,
                                  OUTD / f"{split}_predictions.csv", hashes)


def timeline_from(frozen, pred_csv):
    p = pd.read_csv(pred_csv, usecols=["__global_row_id", "p_attack_calibrated"])
    m = pd.Series(p["p_attack_calibrated"].to_numpy(), index=p["__global_row_id"].to_numpy())
    prob = m.loc[frozen["event_id"]].to_numpy(dtype=np.float64)
    base = np.where(prob < 1.25 / 5.75, 2, np.where(prob >= 1.75 / 3.25, 0, 1)).astype(np.int8)
    return {**frozen, "trust": 100.0 * (1.0 - prob), "base_action": base}, prob


def evaluate_selection():
    R = L.R
    valf, testf = L.load()
    val, pv = timeline_from(valf, OUTD / "val_tune_predictions.csv")
    test, pt = timeline_from(testf, OUTD / "test_predictions.csv")
    from sklearn.metrics import roc_auc_score
    rows = [{"policy": "detector", "val_auc": float(roc_auc_score(val["label"], pv)), "test_auc": float(roc_auc_score(test["label"], pt))}]
    va, vs = L.actions_for(val)
    ta, ts = L.actions_for(test)
    bm = R.evaluate(test["base_action"], test)
    rows.append({"policy": "base", **{k: bm[k] for k in ("sr_action", "far_direct", "fdr_hard", "mfa_benign")}})
    vm = [R.evaluate(va[:, i], val) for i in range(250)]
    tm = [R.evaluate(ta[:, i], test, ts[:, i] if i < 50 else None) for i in range(250)]
    for b in L.BUDGETS:
        for fi, f in enumerate(L.FAMILIES):
            allowed = [i for i in range(50 * fi, 50 * fi + 50) if R.feasible(vm[i], b, L.CAPS)]
            if not allowed:
                rows.append({"policy": f, "budget": b, "status": "NO_SELECTION"})
                continue
            ch = min(allowed, key=lambda i: R.rank(vm[i], L.BANK[i]["id"]))
            rows.append({"policy": f, "budget": b, "status": "SELECTED", "candidate": L.BANK[ch]["id"],
                         "params": json.dumps(L.BANK[ch]["parameters"]), "test_caps_met": R.feasible(tm[ch], b, L.CAPS),
                         **{k: tm[ch][k] for k in ("sr_action", "far_direct", "fdr_hard", "mfa_benign")}})
    # frontier envelope on test under the FAR/FDR screens
    env = []
    for lvl in (0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.05, 0.10):
        r = {"mfa_level": lvl}
        for fi, f in enumerate(L.FAMILIES):
            ok = [tm[i]["sr_action"] for i in range(50 * fi, 50 * fi + 50)
                  if tm[i]["far_direct"] <= 0.005 and tm[i]["fdr_hard"] <= 0.05 and tm[i]["mfa_benign"] <= lvl]
            r[f] = min(ok) if ok else None
        env.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "A10_start_features_selection.csv", index=False)
    pd.DataFrame(env).to_csv(L.OUT / "A10_start_features_envelope.csv", index=False)
    np.savez_compressed(OUTD / "start_feature_timelines.npz", val_trust=val["trust"], val_base=val["base_action"],
                        test_trust=test["trust"], test_base=test["base_action"])
    return df, pd.DataFrame(env)


if __name__ == "__main__":
    t0 = time.time()
    if not (OUTD / "test_predictions.csv").exists():
        fit_and_score()
        print("fitted and scored", f"{time.time()-t0:.0f}s", flush=True)
    df, env = evaluate_selection()
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    print(env.to_string(index=False))
