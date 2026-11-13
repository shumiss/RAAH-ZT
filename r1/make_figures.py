"""New figures for the IoT-76550 revision (all except the framework figure)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, ".")
import r1_lib as L  # noqa: E402

FIG = L.HERE.parent / "figures_new"
FIG.mkdir(exist_ok=True)
for f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    font_manager.fontManager.addfont(str(Path("C:/Windows/Fonts") / f))
plt.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5,
                     "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7.5, "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                     "mathtext.it": "Times New Roman:italic", "axes.linewidth": 0.7})
NAVY, BLUE, TEAL, CORAL, PURPLE, GRAY = "#1d3557", "#457b9d", "#0a9396", "#e76f51", "#6d597a", "#6c757d"
COL = {"hysteresis": BLUE, "single": NAVY, "sma": CORAL, "ema": TEAL, "counter": PURPLE, "base": GRAY}
NAME = {"hysteresis": "Hysteresis", "single": "Single", "sma": "SMA", "ema": "EMA", "counter": "Counter", "base": "Base"}
MARK = {"hysteresis": "o", "single": "s", "sma": "^", "ema": "D", "counter": "P"}
ACOL = {0: "#b23a48", 1: "#e9b44c", 2: "#2a9d8f"}  # Deny, MFA, Allow
R = L.R


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel(ax, k):
    ax.set_title(f"({'abcdef'[k]})", loc="left", fontweight="bold", fontsize=9)


# ------------------------------------------------------------------ 1. mechanism on a real segment
def fig_mechanism(test, ta, ts, start=214565, w=70):
    h = L.static_index(0.02, "hysteresis"); s = L.static_index(0.02, "single")
    lo, hi = L.BANK[h]["parameters"]["low"], L.BANK[h]["parameters"]["high"]
    st = L.BANK[s]["parameters"]["threshold"]
    seg = slice(start, start + w)
    x = np.arange(w)
    trust, base = test["trust"][seg], test["base_action"][seg]
    hs, ah, asg = ts[seg, h], ta[seg, h], ta[seg, s]
    fig = plt.figure(figsize=(7.1, 4.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.3, 0.75, 1.25], hspace=0.36)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, trust, "-", color=GRAY, lw=0.7, zorder=1)
    ax.scatter(x, trust, s=9, c=[ACOL[b] for b in base], zorder=2, edgecolors="none")
    ax.axhline(hi, color=BLUE, lw=1.1); ax.axhline(lo, color=BLUE, lw=1.1)
    ax.axhline(st, color=NAVY, lw=0.9, ls=(0, (4, 2)))
    ax.axhline(100 * (1 - 1.25 / 5.75), color=GRAY, lw=0.8, ls=":")
    ax.text(w + 0.6, hi + 0.8, r"$\theta_{\rm high}$=98.83", va="bottom", fontsize=7.5, color=BLUE)
    ax.text(w + 0.6, st - 0.8, "single=97.80", va="top", fontsize=7.5, color=NAVY)
    ax.text(w + 0.6, 78.3, "base Allow\nboundary 78.26", va="center", fontsize=7, color=GRAY)
    ax.text(w + 0.6, lo, r"$\theta_{\rm low}$=66.83", va="center", fontsize=7.5, color=BLUE)
    ax.set_ylim(max(0, trust.min() - 8), 101.5); ax.set_xlim(-1, w)
    ax.set_ylabel("Trust score $100(1-p)$"); ax.set_xticklabels([])
    for k, lab in ((0, "Deny"), (1, "MFA"), (2, "Allow")):
        ax.scatter([], [], s=12, c=ACOL[k], label=f"base {lab}")
    ax.legend(loc="lower left", ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8)
    panel(ax, 0)
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.step(x, hs + 0.04, where="mid", color=BLUE, lw=1.3, label="hysteresis state (High=1)")
    ax2.step(x, (trust > st).astype(int) - 0.04, where="mid", color=NAVY, lw=1.0, ls=(0, (3, 1.5)), label="single threshold exceeded")
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Low", "High"]); ax2.set_ylim(-0.3, 1.3)
    ax2.legend(loc="lower right", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(1.0, 0.98), borderaxespad=0.1); ax2.set_xticklabels([])
    panel(ax2, 1)
    ax3 = fig.add_subplot(gs[2], sharex=ax)
    rows = [("Base", base), ("Single + fusion", asg), ("Hysteresis + fusion", ah)]
    for r_i, (lab, arr) in enumerate(rows):
        yy = 2 - r_i
        for i, a in enumerate(arr):
            ax3.add_patch(Rectangle((i - 0.5, yy - 0.38), 1, 0.76, color=ACOL[int(a)], lw=0))
        ax3.text(w + 0.6, yy, f"{int(np.count_nonzero(np.diff(arr)))} switches", va="center", fontsize=7.5)
    ax3.set_yticks([2, 1, 0]); ax3.set_yticklabels([r[0] for r in rows]); ax3.set_ylim(-0.6, 2.6)
    ax3.set_xlabel(f"Consecutive flows of one source-IP history in the test replay (rows {start:,}–{start + w - 1:,})")
    panel(ax3, 2)
    save(fig, "fig_mechanism_segment")


# ------------------------------------------------------------------ 2. adaptation timeline schematic
def box(ax, xy, w, h, text, fc="#f4f7fb", ec=NAVY, fs=6.6, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)


def arrow(ax, a, b, color=NAVY, ls="-", lw=0.9, cs="arc3"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, ls=ls, connectionstyle=cs))


def fig_timeline():
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    x0, x1 = 0.03, 0.555
    ups = [x0 + k * (x1 - x0) / 5.5 for k in range(1, 6)]
    yax = 0.88
    ax.annotate("", xy=(x1 + 0.015, yax), xytext=(x0, yax), arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))
    ax.text(x0, yax - 0.035, "global event index", ha="left", va="top", fontsize=6.6)
    for k, u in enumerate(ups):
        ax.plot([u, u], [0.12, yax + 0.015], color=GRAY, lw=0.6, ls=(0, (2, 2)))
        ax.text(u, yax + 0.03, f"update {k + 1}", ha="center", fontsize=6.6)
    u = ups[3]; W = 0.25
    ax.add_patch(Rectangle((u - W, 0.12), W, 0.72, color="#dbe7f3", alpha=0.55, lw=0))
    ax.annotate("", xy=(u, 0.80), xytext=(u - W, 0.80), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.8))
    ax.text(u - W / 2, 0.815, "trailing window $W$", ha="center", fontsize=7, color=BLUE)
    lanes = [(0.68, "Arrived labels"), (0.46, "Shadow candidates\n(50 per family)"), (0.22, "Active output")]
    for y, lab in lanes:
        ax.text(x0 - 0.005, y, lab, ha="right", va="center", fontsize=7.2)
    rng = np.random.default_rng(3)
    ev = np.linspace(x0 + 0.01, x1 - 0.01, 46)
    for e in ev:
        kind = rng.random()
        c = "#2a9d8f" if kind < 0.72 else ("#b23a48" if kind < 0.9 else "white")
        ec = "#b23a48" if c == "white" else c
        ax.scatter(e, 0.68, s=10, c=c, edgecolors=ec, lw=0.6, zorder=3)
    ax.annotate("", xy=(ev[20] + 0.055, 0.645), xytext=(ev[20], 0.645), arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=0.8))
    ax.text(ev[20] + 0.027, 0.61, "delay $D$", ha="center", fontsize=6.8, color=CORAL)
    ax.text(x0, 0.715, "labels arrive after a delay; some missing or flipped", fontsize=6.3, va="bottom", color=GRAY)
    xs = np.linspace(x0 + 0.01, x1 - 0.01, 300)
    for k in range(6):
        ax.plot(xs, 0.46 + 0.012 * np.sin(14 * xs + k) + (k - 2.5) * 0.017, color=BLUE, lw=0.6, alpha=0.75)
    ax.text(x0, 0.53, "all candidate states advance on every event", fontsize=6.3, va="bottom", color=GRAY)
    seg = [(x0, ups[0], "validated\nconfiguration", "#cfe0d9"), (ups[0], ups[1], "$v_1$", "#dbe7f3"),
           (ups[1], ups[2], "$v_1$", "#dbe7f3"), (ups[2], ups[3], "$v_2$", "#e7dff0"),
           (ups[3], ups[4], "validated\n(revert)", "#cfe0d9"), (ups[4], x1, "$v_3$", "#dbe7f3")]
    ax.text(x0, 0.285, "emits the fused Deny/MFA/Allow action", fontsize=6.3, va="bottom", color=GRAY)
    for a, b, lab, c in seg:
        ax.add_patch(Rectangle((a, 0.17), b - a, 0.10, fc=c, ec=NAVY, lw=0.6))
        ax.text((a + b) / 2, 0.22, lab, ha="center", va="center", fontsize=6.6)
    # decision flow at the update marked by the window
    bx, bw = 0.635, 0.355
    box(ax, (bx, 0.80), bw, 0.10, "Support gate: ≥200 benign, ≥20 attack\nlabels and ≥10% coverage in window")
    box(ax, (bx, 0.635), bw, 0.10, "Window rates from arrived labels\n(raw, or flip-corrected)")
    box(ax, (bx, 0.47), bw, 0.10, "GP-LCB reveals $B$=10 of 50\ncandidate losses per family")
    box(ax, (bx, 0.305), bw, 0.10, "Any queried candidate meets\nFAR, FDR and MFA caps?", fc="#fff6e5", ec=CORAL)
    box(ax, (bx, 0.07), 0.17, 0.15, "Yes: commit the\nlowest-SR feasible\nqueried candidate", fc="#e8f1fa")
    box(ax, (bx + bw - 0.17, 0.07), 0.17, 0.15, "No: revert to the\nvalidation-selected\nconfiguration", fc="#e6f2ee", ec=TEAL)
    for y0, y1 in ((0.80, 0.735), (0.635, 0.57), (0.47, 0.405)):
        arrow(ax, (bx + bw / 2, y0), (bx + bw / 2, y1))
    arrow(ax, (bx + 0.12, 0.305), (bx + 0.085, 0.22)); arrow(ax, (bx + bw - 0.12, 0.305), (bx + bw - 0.085, 0.22))
    ax.text(bx + bw / 2 + 0.01, 0.767, "gate fails: keep current", fontsize=5.8, va="center", color=GRAY)
    arrow(ax, (u + 0.004, 0.70), (bx - 0.004, 0.85), color=BLUE, ls=(0, (3, 2)), cs="arc3,rad=-0.2")
    save(fig, "fig_adaptation_timeline")


# ------------------------------------------------------------------ 3. frontier + bootstrap CIs
def fig_frontier():
    d = pd.read_csv(L.OUT / "B1_frontier_bootstrap.csv")
    d = d[(d.timeline == "primary") & (d.status == "OK")]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.58})
    ax = axs[0]
    hp = d.drop_duplicates("mfa_cap")[["mfa_cap", "hyst_mfa", "hyst_sr"]].sort_values("mfa_cap")
    ax.plot(hp.hyst_mfa, hp.hyst_sr, "-o", color=COL["hysteresis"], ms=4, lw=1, label=NAME["hysteresis"])
    for f in ("single", "sma", "ema"):
        c = d[d.comparator == f].sort_values("mfa_cap")
        ax.plot(c.comp_mfa, c.comp_sr, "-" + MARK[f], color=COL[f], ms=3.6, lw=0.9, mfc="white" if f != "single" else COL[f], label=NAME[f])
    lab = hp.groupby(["hyst_mfa", "hyst_sr"], sort=False).mfa_cap.apply(lambda c: "/".join(f"{100 * v:.2f}" for v in c))
    for k, ((mx, sy), txt) in enumerate(lab.items()):
        off = (-6, -14) if k == 0 else ((-4 - 3.2 * len(txt), -3) if k < len(lab) - 1 else (-10, -12))
        ax.annotate(txt, (mx, sy), textcoords="offset points", xytext=off, fontsize=6.3, color=BLUE)
    ax.set_xlim(left=0.96)
    ax.text(0.03, 0.03, "labels: benign-MFA cap (%)", transform=ax.transAxes, fontsize=6.3, color=BLUE)
    ax.set_xlabel("Benign MFA rate on test (%)"); ax.set_ylabel("Final-action SR on test (%)")
    ax.legend(frameon=False, loc="upper right"); ax.grid(alpha=0.25, lw=0.4)
    panel(ax, 0)
    caps = sorted(d.mfa_cap.unique())
    for k, (col, lab) in enumerate((("sr", "SR difference (pp)"), ("far", "Direct-FAR difference (pp)"))):
        ax = axs[k + 1]
        for j, f in enumerate(("single", "sma", "ema")):
            c = d[d.comparator == f].set_index("mfa_cap")
            xs, ys, lo, hi = [], [], [], []
            for i, cap in enumerate(caps):
                if cap not in c.index:
                    continue
                ci = ast.literal_eval(c.loc[cap, f"d_{col}_ci"])
                xs.append(i + (j - 1) * 0.22); ys.append(c.loc[cap, f"d_{col}_pp"]); lo.append(ci[0]); hi.append(ci[1])
            ys, lo, hi = map(np.asarray, (ys, lo, hi))
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=MARK[f], color=COL[f], ms=3.6, lw=0.9, capsize=1.8,
                        mfc="white" if f != "single" else COL[f], label=f"vs {NAME[f]}")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(caps))); ax.set_xticklabels([f"{100 * c:.2f}" for c in caps], rotation=0)
        ax.set_xlabel("Benign-MFA cap (%)"); ax.set_ylabel(f"Hysteresis − comparator\n{lab}")
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        if k == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=7)
        panel(ax, k + 1)
    save(fig, "fig_frontier_bootstrap")


# ------------------------------------------------------------------ 4. noise curve + fallback
def fig_noise():
    a = pd.read_csv(L.OUT / "A4_noise_curve.csv")
    a = a[(a.family == "hysteresis") & (a.eps_assumed == a.flip)]
    b = pd.read_csv(L.OUT / "A4b_noise_curve_recommended_fallback.csv")
    flips = sorted(b.flip.unique())
    xs = np.arange(len(flips))
    series = [("Original: base start, raw rates", a[(a.initial == "base") & (a.estimator == "raw")], NAVY, "s", "-"),
              ("Validated fallback, raw rates", b[b.estimator == "raw"], TEAL, "o", "-"),
              ("Validated fallback, flip-corrected", b[b.estimator == "corrected"], CORAL, "^", "-")]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw={"wspace": 0.38})
    for k, (col, lab) in enumerate((("committed", "Updates with a feasible commit"), ("far_direct", "Direct FAR on test (%)"),
                                    ("mfa_benign", "Benign MFA on test (%)"))):
        ax = axs[k]
        for name, s, c, m, ls in series:
            s = s.set_index("flip").loc[flips]
            y = 186 - s.infeasible_updates if col == "committed" else 100 * s[col]
            ax.plot(xs, y, ls, marker=m, color=c, ms=3.8, lw=1, mfc="white", label=name)
        if col == "far_direct":
            ax.axhline(0.5, color=GRAY, lw=0.8, ls="--"); ax.text(len(flips) - 1, 0.52, "0.5% screen", ha="right", fontsize=7, color=GRAY)
        if col == "mfa_benign":
            ax.axhline(2.0, color=GRAY, lw=0.8, ls="--"); ax.text(len(flips) - 1, 1.95, "2% cap", ha="right", va="top", fontsize=7, color=GRAY)
        ax.set_xticks(xs); ax.set_xticklabels([f"{100 * f:g}" for f in flips])
        ax.set_xlabel("Symmetric label-flip rate (%)"); ax.set_ylabel(lab); ax.grid(alpha=0.25, lw=0.4)
        panel(ax, k)
    axs[0].legend(frameon=False, fontsize=7.2, loc="lower center", ncol=3, bbox_to_anchor=(1.8, 1.08))
    save(fig, "fig_noise_fallback")


# ------------------------------------------------------------------ 5. local benign-MFA windows
def windows(y, a, win=16384, step=4096):
    b = y == 0
    out = []
    for s in range(0, len(y) - win + 1, step):
        bb = b[s:s + win]
        out.append((s + win, 100 * np.mean(a[s:s + win][bb] == 1)))
    return np.asarray(out)


def fig_local(test):
    order = np.lexsort((test["event_id"], test["timestamp"]))
    y = test["label"][order]
    z = np.load(L.ORIG / "selected_test_actions.npz")
    zs = np.load(L.OUT / "C2_score_shift_actions_global.npz")  # global order, revised fallback
    assert np.array_equal(zs["label_global"], y)
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for name, key, c in (("Base", "base", GRAY), ("Fixed hysteresis", "rho0.02_hysteresis", BLUE), ("Fixed SMA", "rho0.02_sma", CORAL)):
        w = windows(y, z[key][order])
        axs[0].plot(w[:, 0] / 1e5, w[:, 1], color=c, lw=0.9, label=f"{name} (global {100 * np.mean(z[key][order][y == 0] == 1):.2f}%)")
    for name, key, c, ls in (("Fixed hysteresis", "fixed_rho0.02_hysteresis_global", BLUE, "-"),
                             ("Adaptive hysteresis", "adaptive_validated_fallback_rho0.02_hysteresis_global", NAVY, "--")):
        w = windows(y, zs[key])
        axs[1].plot(w[:, 0] / 1e5, w[:, 1], color=c, lw=0.9, ls=ls, label=f"{name} (global {100 * np.mean(zs[key][y == 0] == 1):.2f}%)")
    n = len(y)
    for k, off in enumerate(("+0", "+12", "−12", "+0")):
        axs[1].axvspan(k * n / 4 / 1e5, (k + 1) * n / 4 / 1e5, color="#eef2f6" if k % 2 else "white", lw=0, zorder=0)
        axs[1].text((k + 0.5) * n / 4 / 1e5, 0.97, f"offset {off}", transform=axs[1].get_xaxis_transform(), ha="center", va="top", fontsize=7)
    for k, ax in enumerate(axs):
        ax.axhline(2.0, color="black", lw=0.7, ls=":")
        ax.set_ylim(0, 9.6); ax.set_xlabel("Global test event (×10$^5$), window end")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.8, loc="upper left",
                  bbox_to_anchor=(0, 1) if k == 0 else (0, 0.9), ncol=1); ax.grid(alpha=0.25, lw=0.4)
        panel(ax, k)
    axs[0].set_ylabel("Benign MFA in trailing\n16,384-event window (%)")
    axs[1].text(4.55, 2.12, "2% cap", fontsize=7, ha="center")
    wa = windows(y, zs["adaptive_validated_fallback_rho0.02_hysteresis_global"]); k = int(np.argmax(wa[:, 1]))
    axs[1].annotate(f"clipped; peak {wa[k, 1]:.1f}%", (wa[k, 0] / 1e5 + 0.05, 9.4), xytext=(5.05, 7.6),
                    fontsize=6.8, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=NAVY))
    save(fig, "fig_local_mfa")


# ------------------------------------------------------------------ 6. feedback conditions (revised fallback)
def stacked_axes(n=3, h=5.2):
    fig, axs = plt.subplots(n, 1, figsize=(7.1, h), sharex=True, gridspec_kw={"hspace": 0.12})
    return fig, axs


LS = {"hysteresis": "-", "single": "--", "sma": "-.", "ema": ":", "counter": (0, (4, 1, 1, 1))}


def fig_feedback_R1():
    d = pd.read_csv(L.OUT / "C1_dynamic75_validated_fallback.csv")
    d = d[(d.variant == "primary_raw") & (d.budget == 0.02)]
    conds = ["clean_D0", "clean_D256", "quarter_clean_D256", "full_noisy_D256", "quarter_noisy_D256"]
    labels = ["Full clean\nD0", "Full clean\nD256", "25% clean\nD256", "Full 10% flip\nD256", "25% 10% flip\nD256"]
    fig, axs = stacked_axes()
    x = np.arange(len(conds))
    for ax, (col, lab) in zip(axs, (("sr_action", "SR (%)"), ("far_direct", "Direct FAR (%)"), ("mfa_benign", "Benign MFA (%)"))):
        ax.axvspan(2.5, 4.5, color="#f0f0f0", lw=0, zorder=0)
        for f in L.FAMILIES:
            s = d[d.family == f].set_index("condition").loc[conds]
            ax.plot(x, 100 * s[col], ls=LS[f], marker=MARK[f], color=COL[f], ms=4, lw=1.1, label=NAME[f])
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.25, lw=0.4)
        if col == "far_direct":
            ax.axhline(0.5, color=GRAY, lw=0.7, ls="--"); ax.text(4.45, 0.505, "FAR screen 0.5%", fontsize=7, va="bottom", ha="right")
        if col == "mfa_benign":
            ax.axhline(2.0, color=GRAY, lw=0.7, ls="--"); ax.text(4.45, 2.005, "Benign-MFA screen 2%", fontsize=7, va="bottom", ha="right")
        if col == "sr_action":
            ax.axhline(2.1212, color=GRAY, lw=0.7, ls=":"); ax.text(4.45, 2.124, "base SR", fontsize=7, va="bottom", ha="right")
    axs[1].set_ylim(0, 0.55); axs[2].set_ylim(1.2, 2.12); axs[0].set_ylim(1.7, 2.16)
    axs[-1].set_xticks(x); axs[-1].set_xticklabels(labels)
    axs[0].legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig_feedback_conditions_R1")


def fig_shift_R1():
    q = pd.read_csv(L.OUT / "C2_score_shift_quarters.csv")
    q = q[(q.budget == 0.02) & q.selection.isin(["fixed", "adaptive_validated_fallback"])]
    fig, axs = stacked_axes()
    x = np.arange(4)
    series = [("hysteresis", "fixed", "Hyst fixed", "-", "o"), ("hysteresis", "adaptive_validated_fallback", "Hyst adaptive", "--", "o"),
              ("sma", "fixed", "SMA fixed", "-", "s"), ("sma", "adaptive_validated_fallback", "SMA adaptive", "--", "s")]
    for ax, (col, lab) in zip(axs, (("sr_action", "SR (%)"), ("far_direct", "Direct FAR (%)"), ("mfa_benign", "Benign MFA (%)"))):
        for f, sel, name, ls, m in series:
            s = q[(q.family == f) & (q.selection == sel)].sort_values("phase")
            c = COL[f]
            ax.plot(x, 100 * s[col], ls=ls, color=c, lw=1.1, label=name)
            ok = s.phase_caps_met.values
            ax.scatter(x[ok], 100 * s[col].values[ok], marker=m, color=c, s=22, zorder=3)
            ax.scatter(x[~ok], 100 * s[col].values[~ok], marker=m, facecolor="white", edgecolor=c, s=22, zorder=3, lw=1)
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.25, lw=0.4)
        if col == "far_direct":
            ax.axhline(0.5, color=GRAY, lw=0.7, ls="--")
        if col == "mfa_benign":
            ax.axhline(2.0, color=GRAY, lw=0.7, ls="--")
    axs[0].scatter([], [], marker="o", facecolor="white", edgecolor="black", label="Phase screen failed")
    axs[-1].set_xticks(x); axs[-1].set_xticklabels(["Q1\n+0 points", "Q2\n+12 points", "Q3\n−12 points", "Q4\n+0 points"])
    axs[0].legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig_score_shift_R1")


if __name__ == "__main__":
    val, test = L.load()
    ta, ts = L.actions_for(test)
    fig_mechanism(test, ta, ts)
    fig_timeline()
    fig_frontier()
    fig_noise()
    fig_local(test)
    fig_feedback_R1()
    fig_shift_R1()
    print("figures in", FIG)
