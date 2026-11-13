"""Generate supp_s7.tex (R1 revision evidence) from the CSV/JSON results in ../r1_experiments/results."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "r1" / "results"
FAM = {"hysteresis": "Hyst.", "single": "Single", "sma": "SMA", "ema": "EMA", "counter": "Counter", "base": "Base"}
COND = {"clean_D0": "Clean D0", "clean_D256": "Clean D256", "quarter_clean_D256": "25\\% clean D256",
        "quarter_noisy_D256": "25\\% 10\\% flip D256", "full_noisy_D256": "Full 10\\% flip D256"}
out: list[str] = []


def p(x, d=3):
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def yes(x):
    return "Yes" if bool(x) else "\\textbf{No}"


def longtable(caption, label, cols, header, rows, note=None):
    out.append(f"{{\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}\n\\begin{{longtable}}{{{cols}}}")
    out.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{header}\\\\\n\\midrule\n\\endfirsthead")
    out.append(f"\\toprule\n{header}\\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot")
    out.extend(" & ".join(map(str, r)) + "\\\\" for r in rows)
    out.append("\\end{longtable}}")
    if note:
        out.append(f"\\noindent{{\\footnotesize {note}}}\\par\\medskip")


DYN_H = "Condition & Cap & Family & Ver. & Infeas. & SR & FAR$_d$ & MFA$_b$ & Caps"


def dyn_rows(d):
    return [[COND.get(r.condition, r.condition.replace("_", "\\_")), f"{100 * r.budget:g}", FAM[r.family], r.version_changes,
             r.infeasible_updates, p(r.sr_action), p(r.far_direct), p(r.mfa_benign), yes(r.test_caps_met)] for r in d.itertuples()]


out.append(r"""\section{R1 revision evidence}
\label{sec:supp_r1}
This section contains every result added in the first revision. Rates are percentages; ``Ver.'' counts active-version
changes and ``Infeas.'' counts supported updates with no feasible queried candidate (186 scheduled updates).
All runs use the frozen validation and test timelines, the frozen candidate bank and the frozen runner. A
reimplementation of the delayed-feedback engine reproduces the submitted clean-D256 and full-noisy rows and the
submitted score-shift rows with zero difference before the start and fallback rules are changed. All code,
protocols and results are available at \url{https://github.com/shumiss/RAAH-ZT}. Scripts (folder \texttt{r1}):
\texttt{r1\_lib.py}, \texttt{exp\_static.py}, \texttt{exp\_dynamic.py}, \texttt{exp\_robust.py},
\texttt{exp\_timing\_local.py}, \texttt{exp\_causal\_features.py}, \texttt{exp\_rerun75.py},
\texttt{exp\_score\_shift\_R1.py}, \texttt{exp\_misspec\_R1.py}; figures: \texttt{make\_figures.py}.

\subsection{Revised delayed-feedback controller: all 75 rows}""")
c1 = pd.read_csv(RES / "C1_dynamic75_validated_fallback.csv")
order = list(COND)
c1["o"] = c1.condition.map(order.index)
prim = c1[c1.variant == "primary_raw"].sort_values(["o", "budget"])
longtable("Revised controller (start from and revert to the validation-selected configuration; raw window rates).",
          "tab:s7_dyn75", "lllrrrrrl", DYN_H, dyn_rows(prim),
          "At the 1\\% cap validation selects no feasible configuration, so the start and fallback configuration is the base.")
corr = c1[c1.variant == "sensitivity_corrected"].sort_values(["o", "budget"])
longtable("Revised controller with the noise-corrected estimator (assumed flip rate equal to the true 10\\%).",
          "tab:s7_dyn_corr", "lllrrrrrl", DYN_H, dyn_rows(corr))
st = pd.read_csv(RES / "C1_static_reference.csv")
longtable("Fixed validation-selected reference for the same caps.", "tab:s7_static_ref", "llrrrl",
          "Cap & Family & SR & FAR$_d$ & MFA$_b$ & Caps",
          [[f"{100 * r.budget:g}", FAM[r.family], p(r.sr_action), p(r.far_direct), p(r.mfa_benign), yes(r.test_caps_met)]
           for r in st.itertuples()])

out.append(r"\subsection{Start and fallback designs}")
a3 = pd.read_csv(RES / "A3_fallback.csv")
a3["o"] = a3.condition.map(order.index)
a3 = a3[a3.family == "hysteresis"].sort_values(["o", "budget", "initial", "fallback"])
longtable("Hysteresis family under four start/fallback rules (GP-LCB, raw rates).", "tab:s7_fallback", "llllrrrrrl",
          "Condition & Cap & Start & Fallback & Ver. & Infeas. & SR & FAR$_d$ & MFA$_b$ & Caps",
          [[COND[r.condition], f"{100 * r.budget:g}", r.initial, r.fallback, r.version_changes, r.infeasible_updates,
            p(r.sr_action), p(r.far_direct), p(r.mfa_benign), yes(r.test_caps_met)] for r in a3.itertuples()],
          "The other four families are in \\texttt{A3\\_fallback.csv}.")

out.append(r"\subsection{Label-noise curve and flip-rate misspecification}")
a4 = pd.read_csv(RES / "A4_noise_curve.csv")
a4 = a4[(a4.family == "hysteresis") & (a4.eps_assumed == a4.flip)]
a4b = pd.read_csv(RES / "A4b_noise_curve_recommended_fallback.csv")
rows = [[f"{100 * r.flip:g}", r.initial, r.fallback, r.estimator, r.version_changes, r.infeasible_updates, p(r.sr_action),
         p(r.far_direct), p(r.mfa_benign), yes(r.test_caps_met)]
        for r in pd.concat([a4, a4b]).sort_values(["flip", "initial", "fallback", "estimator"]).itertuples()]
longtable("Hysteresis at the 2\\% cap, full coverage, D256, symmetric flips 0--10\\%.", "tab:s7_noise", "rlllrrrrrl",
          "Flip & Start & Fallback & Estimator & Ver. & Infeas. & SR & FAR$_d$ & MFA$_b$ & Caps", rows)
c3 = pd.read_csv(RES / "C3_misspecified_flip_validated_fallback.csv")
longtable("Corrected estimator with a misspecified flip rate (revised controller, hysteresis, 2\\% cap).", "tab:s7_misspec",
          "rrrrrrrl", "True flip & Assumed & Ver. & Infeas. & SR & FAR$_d$ & MFA$_b$ & Caps",
          [[f"{100 * r.flip:g}", f"{100 * r.eps_assumed:g}", r.version_changes, r.infeasible_updates, p(r.sr_action),
            p(r.far_direct), p(r.mfa_benign), yes(r.test_caps_met)] for r in c3.itertuples()])

out.append(r"\subsection{Why adaptation trails the fixed policy on stationary data}")
a5 = pd.read_csv(RES / "A5_decomposition.csv")
longtable("Decomposition of adaptive switching (clean D256, 2\\% cap, retain fallback).", "tab:s7_decomp", "llrrrrrrr",
          "Start & Family & Switches & On version edges & SR & SR excl. edges & Static share & Candidates & Static SR",
          [[r.initial, FAM[r.family], r.switches_total, r.switches_on_version_edges, p(r.sr_total),
            p(r.sr_excluding_version_edges), f"{100 * r.share_time_on_static_selection:.1f}", r.distinct_candidates_used,
            p(r.static_selection_sr_same_trace)] for r in a5.itertuples()],
          "``Static share'' is the percentage of events on which the validation-selected candidate is active.")

out.append(r"\subsection{Query rules and replicates}")
a6 = pd.read_csv(RES / "A6_selectors.csv")
g = a6.groupby(["condition", "family", "selector"], sort=False).agg(
    ver=("version_changes", "mean"), sr=("sr_action", "mean"), far=("far_direct", "mean"), mfa=("mfa_benign", "mean"),
    t=("update_time_s_all_families", "mean")).reset_index()
longtable("Query rules at the 2\\% cap (submitted base/retain controller; random10 averaged over five seeds).",
          "tab:s7_selectors", "lllrrrrr", "Condition & Family & Rule & Ver. & SR & FAR$_d$ & MFA$_b$ & Update time (s)",
          [[COND[r.condition], FAM[r.family], r.selector, f"{r.ver:.0f}", p(r.sr), p(r.far), p(r.mfa), f"{r.t:.1f}"]
           for r in g.itertuples()],
          "Update time is the total wall time of all updates for all families and caps in one replay.")
b2 = pd.read_csv(RES / "B2_replicated_selectors.csv")
b2 = b2[b2.family == "hysteresis"]
g = b2.groupby(["replicate_kind", "config"], sort=False).agg(n=("sr_action", "size"), sr=("sr_action", "mean"),
                                                           sd=("sr_action", "std"), far=("far_direct", "mean"),
                                                           mfa=("mfa_benign", "mean"), ver=("version_changes", "mean")).reset_index()
kind = {"quarter_clean_seed": "25\\% clean, 10 seeds", "full_clean_phase": "Full clean, 8 phases",
        "flip5_corrected_seed": "5\\% flip + corr., 5 seeds"}
longtable("Replicated comparison (hysteresis, 2\\% cap): mean over replicates.", "tab:s7_rep", "llrrrrr",
          "Replicates & Rule$|$start$|$fallback & SR & SD & FAR$_d$ & MFA$_b$ & Ver.",
          [[kind[r.replicate_kind], r.config.replace("|", "$|$"), p(r.sr), f"{100 * r.sd:.3f}", p(r.far), p(r.mfa),
            f"{r.ver:.1f}"] for r in g.itertuples()])

out.append(r"\subsection{Validation-selected frontier with source-IP bootstrap intervals}")
b1 = pd.read_csv(RES / "B1_frontier_bootstrap.csv")
b1 = b1[b1.status == "OK"]


def ci(s):
    a, b = ast.literal_eval(s)
    return f"[{a:+.3f}, {b:+.3f}]"


longtable("Hysteresis minus comparator (percentage points) with 95\\% paired bootstrap intervals over source-IP histories "
          "(5,000 resamples). Policies are selected on validation at each cap.", "tab:s7_boot", "llllrlrl",
          "Evaluator & Cap & vs & $\\Delta$SR & & $\\Delta$FAR$_d$ & & $\\Delta$MFA$_b$",
          [["Primary" if r.timeline == "primary" else "Start-time", f"{100 * r.mfa_cap:g}", FAM[r.comparator],
            f"{r.d_sr_pp:+.3f}", ci(r.d_sr_ci), f"{r.d_far_pp:+.3f}", ci(r.d_far_ci), f"{r.d_mfa_pp:+.3f} {ci(r.d_mfa_ci)}"]
           for r in b1.itertuples()])

out.append(r"\subsection{Ordering, inactivity timeouts and start-time features}")
a1 = pd.read_csv(RES / "A1_random_tie_summary.csv")
longtable("Twenty random within-second permutations (selected parameters retained).", "tab:s7_ties", "lrrrrr",
          "Policy & SR mean & SR min & SR max & FAR$_d$ & MFA$_b$",
          [[r.policy.replace("_", " ").replace("rho", "$\\rho$="), p(r.sr_action_mean), p(r.sr_action_min),
            p(r.sr_action_max), p(r.far_direct_mean), p(r.mfa_benign_mean)] for r in a1.itertuples()])
a2 = pd.read_csv(RES / "A2_session_timeouts.csv")
a2 = a2[a2.status.isin(["SELECTED", "BASE"])]
longtable("Inactivity-timeout resets with re-selection on validation (empty $\\tau$: no timeout).", "tab:s7_timeout",
          "rrlllrrrr", "$\\tau$ (s) & Sessions & Cap & Family & Candidate & SR & SR within & FAR$_d$ & MFA$_b$",
          [["--" if pd.isna(r.tau_s) else f"{r.tau_s:g}", r.test_sessions, "--" if pd.isna(r.budget) else f"{100 * r.budget:g}",
            FAM[r.family], "--" if pd.isna(r.candidate) else str(r.candidate).replace("_", "\\_"), p(r.sr_action),
            p(r.sr_within_session), p(r.far_direct), p(r.mfa_benign)] for r in a2.itertuples()],
          "``SR within'' excludes pairs that cross a timeout reset.")
a10 = pd.read_csv(RES / "A10_start_features_selection.csv")
a10 = a10[a10.policy != "detector"]
longtable("Start-time-feature evaluator (test AUC 0.999): base and validation selections.", "tab:s7_start", "llllrrrl",
          "Policy & Cap & Status & Candidate & SR & FAR$_d$ & MFA$_b$ & Caps",
          [[FAM[r.policy], "--" if pd.isna(r.budget) else f"{100 * r.budget:g}", "--" if pd.isna(r.status) else r.status.replace("_", "\\_"),
            "--" if pd.isna(r.candidate) else str(r.candidate).replace("_", "\\_"), p(r.sr_action), p(r.far_direct),
            p(r.mfa_benign), "--" if pd.isna(r.test_caps_met) else yes(r.test_caps_met)] for r in a10.itertuples()])

out.append(r"\subsection{Tightness of Theorem 4 per source-IP history}")
a8 = pd.read_csv(RES / "A8_theorem4_per_proxy.csv")
j = json.loads((RES / "A8_theorem4.json").read_text(encoding="utf-8"))
longtable(f"Selected pair (66.83, 98.83), $\\Delta={j['delta']:.2f}$. Two proxies with a single flow are omitted. "
          f"Totals: $\\sum V_n={j['sum_V_n']:,.0f}$, bound {j['sum_bound']:,.0f}, $K_h={j['sum_K_h']:,}$, single-threshold flips "
          f"{j['sum_single_threshold_flips_same_high']:,}.", "tab:s7_thm4", "rrrrrr",
          "Proxy & Flows & $V_n$ & $1+V_n/\\Delta$ & $K_h$ & Single flips",
          [[r.group, f"{r.events:,}", f"{r.V_n:,.1f}", f"{r.bound:,.1f}", r.K_h, r.single_threshold_flips_at_high]
           for r in a8.itertuples()])

out.append(r"\subsection{Score-calibration shift with the revised controller}")
c2 = pd.read_csv(RES / "C2_score_shift_rows.csv")
sel = {"fixed": "Fixed", "adaptive_original": "Adaptive (submitted)", "adaptive_validated_fallback": "Adaptive (revised)"}
c2["o"] = c2.selection.map(list(sel).index)
longtable("Full-trace rows under offsets $0,+12,-12,0$ (clean D256 feedback).", "tab:s7_shift", "lllrrrrrl",
          "Selection & Cap & Family & SR & FAR$_d$ & MFA$_b$ & Local max & Windows $>$2\\% & Caps",
          [[sel[r.selection], f"{100 * r.budget:g}", FAM[r.family], p(r.sr_action), p(r.far_direct), p(r.mfa_benign),
            f"{100 * r.local_mfa_max:.2f}", f"{100 * r.local_windows_over_2pct:.1f}", yes(r.test_caps_met)]
           for r in c2.sort_values(["budget", "family", "o"]).itertuples()],
          "Local values use trailing 16,384-event windows with step 4,096.")
q = pd.read_csv(RES / "C2_score_shift_quarters.csv")
q["o"] = q.selection.map(list(sel).index)
longtable("Per-quarter rows (edges belong to the quarter of their later event).", "tab:s7_shift_q", "lllrrrrl",
          "Selection & Cap & Family & Quarter & SR & FAR$_d$ & MFA$_b$ & Screen",
          [[sel[r.selection], f"{100 * r.budget:g}", FAM[r.family], r.phase, p(r.sr_action), p(r.far_direct),
            p(r.mfa_benign), yes(r.phase_caps_met)] for r in q.sort_values(["budget", "family", "o", "phase"]).itertuples()])

out.append(r"\subsection{Local benign-MFA burden in the primary replay}")
a11 = pd.read_csv(RES / "A11_local_mfa_windows.csv")
longtable("Benign MFA in trailing 16,384-event windows (step 4,096).", "tab:s7_local", "llrrrrr",
          "Setting & Policy & Windows & Global & p95 & Max & Windows $>$2\\%",
          [["Primary" if r.setting == "unshifted" else "Score shift",
            r.policy.replace("_", " ").replace("rho0.02 ", "").replace("cleanD256", "(clean D256)"), r.windows, p(r.global_benign_mfa, 2), p(r.p95_benign_mfa, 2),
            p(r.max_benign_mfa, 2), f"{100 * r.share_windows_above_cap:.1f}"] for r in a11.itertuples()],
          "The score-shift rows in this table use the submitted adaptive controller; Table~\\ref{tab:s7_shift} gives the revised one.")

out.append(r"\subsection{Per-layer timing of the adaptive controller}")
t = json.loads((RES / "A9_timing.json").read_text(encoding="utf-8"))
e, s = t["per_event_decision_selected_pair"], t["per_event_shadow_update_50_hysteresis_numpy"]
longtable(f"Python {t['runtime']['python']}, NumPy {t['runtime']['numpy']}, {t['runtime']['platform']}.", "tab:s7_timing",
          "lr", "Layer & Time",
          [["Per-event decision, selected pair (p50/p95/p99)", f"{e['p50_us']}/{e['p95_us']}/{e['p99_us']} \\textmu s"],
           ["50 hysteresis shadow states per event, NumPy (p50/p95/p99)", f"{s['p50_us']}/{s['p95_us']}/{s['p99_us']} \\textmu s"],
           ["250 shadow candidates, compiled, amortized per event",
            f"{t['per_event_amortized_compiled_shadow_250_candidates_us']:.2f} \\textmu s"],
           ["Window metrics, 250 candidates, per update", f"{t['periodic_update_window_metrics_250_candidates_ms']:.1f} ms"],
           ["GP-LCB, 5 families $\\times$ 3 caps, per update", f"{t['periodic_update_gp_lcb_5_families_3_caps_ms']:.0f} ms"],
           ["Exhaustive 50, 5 families $\\times$ 3 caps, per update",
            f"{t['periodic_update_exhaustive_50_5_families_3_caps_ms']:.2f} ms"],
           ["Update interval", f"{t['update_interval_events']:,} events"]])

(HERE / "supp_s7.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print("supp_s7.tex written:", len(out), "blocks")
