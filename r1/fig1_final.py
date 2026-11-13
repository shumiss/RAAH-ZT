"""Figure 1 (R1, final): RAAH-ZT framework at final print width (7.16 in), pure vector, Times New Roman.
(a) per-event decision; (b) offline selection on the real 50-pair hysteresis bank; (c) optional periodic
adaptation with the real GP-LCB queries of the first supported clean-D256 update.
All text >= 7 pt except axis tick labels (6.5 pt)."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle, Arc, Circle, Ellipse, Wedge
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures_new"
RES = HERE / "results"
ORIG = Path(__file__).resolve().parent.parent / "frozen" / "original"
for f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    font_manager.fontManager.addfont(str(Path("C:/Windows/Fonts") / f))
plt.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42, "mathtext.fontset": "custom",
                     "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
                     "mathtext.bf": "Times New Roman:bold"})

INK, NAVY, BLUE, LIGHT, GRAY = "#233448", "#355F80", "#6D91AC", "#dbe6ef", "#6b7a86"
GREEN, GREEN_D, OCHRE, OCHRE_D = "#6D8B77", "#3f6b52", "#BA916C", "#9a6235"
RED, YEL, GRN = "#b65f63", "#d9a441", "#50876e"
BAND = {"a": "#f2f6fa", "b": "#f2f7f3", "c": "#fbf5ef"}
W, H = 7.16, 3.78
S_HEAD, S_TITLE, S_TXT, S_NOTE, S_TICK = 8.5, 7.6, 7.2, 7.0, 6.5

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def sub_axes(x0, y0, x1, y1):
    a = fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])
    a.patch.set_alpha(0)
    return a


def txt(x, y, s, size=S_TXT, w="normal", c=INK, ha="center", va="center", style="normal", z=6, **k):
    return ax.text(x, y, s, fontsize=size, fontweight=w, color=c, ha=ha, va=va, style=style, zorder=z, **k)


def path(pts, c=INK, lw=0.95, ls="-", head=True, z=3, ms=7.5):
    xs, ys = zip(*pts)
    ax.add_line(Line2D(xs, ys, color=c, lw=lw, ls=ls, zorder=z, solid_capstyle="butt", solid_joinstyle="miter",
                       dash_capstyle="butt"))
    if head:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=ms, color=c, lw=0.01,
                                     zorder=z + 0.1, shrinkA=0, shrinkB=0))


def band(key, x0, y0, x1, y1, label, title, note=None, nx=0.10):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=BAND[key], ec="none", zorder=0))
    txt(x0 + 0.10, y1 - 0.13, f"({label})", S_HEAD, "bold", NAVY, ha="left")
    txt(x0 + 0.36, y1 - 0.13, title, S_HEAD, "bold", NAVY, ha="left")
    if note:
        txt(x1 - nx, y1 - 0.13, note, S_NOTE, c=GRAY, ha="right", style="italic")


def bridge(x, y, key):
    ax.add_patch(Rectangle((x - 0.04, y - 0.035), 0.08, 0.07, fc=BAND[key], ec="none", zorder=3.5))


# ------------------------------------------------------------------ bands
band("a", 0.04, 2.02, 7.12, 3.74, "a", "Per-event decision", "every request; state scoped by principal")
band("b", 0.04, 0.04, 2.62, 1.94, "b", "Offline selection")
band("c", 2.70, 0.04, 7.12, 1.94, "c", "Periodic adaptation", r"optional; every $J$ = 4,096 events", nx=0.22)

# ------------------------------------------------------------------ (a) request
yc = 2.84
ax.add_patch(FancyBboxPatch((0.22, 2.80), 0.30, 0.11, boxstyle="round,pad=0,rounding_size=0.02", fc=BLUE, ec="none", zorder=4))
for r in (0.075, 0.13):
    ax.add_patch(Arc((0.37, 2.93), 2 * r, 2 * r, theta1=40, theta2=140, color=BLUE, lw=1.1, zorder=4))
for dx in (-0.07, -0.03, 0.01):
    ax.add_patch(Circle((0.37 + dx, 2.855), 0.013, fc="white", ec="none", zorder=5))
txt(0.37, 2.62, "IoT request", S_TITLE, "bold")
path([(0.60, yc), (0.80, yc)])

# risk evaluator: circle with a gauge glyph
ex, er = 1.04, 0.22
ax.add_patch(Circle((ex, yc), er, fc=NAVY, ec="none", zorder=4))
ax.add_patch(Arc((ex, yc - 0.04), 0.26, 0.26, theta1=0, theta2=180, color="white", lw=1.4, zorder=5))
ax.plot([ex, ex + 0.08], [yc - 0.04, yc + 0.05], color="white", lw=1.4, zorder=5, solid_capstyle="round")
ax.add_patch(Circle((ex, yc - 0.04), 0.022, fc="white", ec="none", zorder=5))
txt(ex, 3.26, "risk evaluator", S_TITLE, "bold")
txt(ex, 3.13, "(e.g., calibrated HGB)", S_NOTE, c=GRAY, style="italic")

# risk score -> trust score -> hysteresis
path([(ex + er, yc), (2.46, yc)])
txt(1.86, yc + 0.10, r"risk score $p$", S_TXT)
txt(1.86, yc - 0.10, "trust score", S_NOTE, c=GRAY, style="italic")
txt(1.86, yc - 0.23, r"$T=100(1-p)$", S_TXT)

# ------------------------------------------------------------------ (a) hysteresis trace
x0h, x1h, y0h, y1h = 2.50, 4.02, 2.46, 3.28
ah = sub_axes(x0h, y0h, x1h, y1h)
TH, TL, YMIN, YMAX = 80, 55, 25, 100
t = np.arange(120)
rng = np.random.default_rng(7)
base = np.interp(t, [0, 14, 20, 34, 46, 58, 62, 70, 82, 94, 100, 119], [44, 46, 88, 72, 84, 66, 40, 63, 74, 60, 90, 86])
trace = np.clip(base + rng.normal(0, 4.0, t.size), 28, 97)
state = np.zeros(t.size, int)
for i in range(1, t.size):
    state[i] = 1 if trace[i] > TH else 0 if trace[i] < TL else state[i - 1]
ah.axhspan(TL, TH, color=LIGHT, lw=0, zorder=0)
ah.axhline(TH, color=NAVY, lw=0.9, zorder=1); ah.axhline(TL, color=NAVY, lw=0.9, zorder=1)
ah.plot(t, trace, color=INK, lw=0.85, zorder=2)
ah.text(1, 97, "schematic", fontsize=S_NOTE, style="italic", color=GRAY, ha="left", va="center")
ah.set_xlim(0, 119); ah.set_ylim(YMIN, YMAX); ah.axis("off")
yv = lambda v: y0h + (v - YMIN) / (YMAX - YMIN) * (y1h - y0h)
txt(x1h + 0.05, yv(TH), r"$\theta^{*}_{\rm high}$", S_TXT, c=NAVY, ha="left")
txt(x1h + 0.05, yv(TL), r"$\theta^{*}_{\rm low}$", S_TXT, c=NAVY, ha="left")
txt((x0h + x1h) / 2, 3.41, "scoped hysteresis state", S_TITLE, "bold")
ar = sub_axes(x0h, 2.30, x1h, 2.39)
start = 0
for i in range(1, state.size + 1):
    if i == state.size or state[i] != state[start]:
        ar.add_patch(Rectangle((start, 0), i - start, 1, fc=NAVY if state[start] else "#c9d6e1", ec="none"))
        start = i
ar.set_xlim(0, 119); ar.set_ylim(0, 1); ar.axis("off")
txt(x0h - 0.05, 2.345, "state (dark: High)", S_NOTE, c=GRAY, ha="right", style="italic")

# ------------------------------------------------------------------ (a) fusion matrix
path([(4.10, 3.27), (4.70, 3.27)])
txt(4.40, 3.36, "state", S_NOTE, c=GRAY, style="italic")
cols = {"High": 4.98, "Low": 5.29}
rows = {"Deny": 3.05, "MFA": 2.84, "Allow": 2.63}
chip = {"Deny": RED, "MFA": YEL, "Allow": GRN}
txt(5.03, 3.45, "conservative fusion", S_TITLE, "bold")
for k, xx in cols.items():
    txt(xx, 3.27, k, S_TXT, "bold", NAVY)
for r, yy in rows.items():
    txt(4.81, yy, r, S_TXT, "bold", ha="right")
    for k, xx in cols.items():
        hi = (r == "Allow" and k == "Low")
        out = "MFA" if hi else r
        ax.add_patch(FancyBboxPatch((xx - 0.14, yy - 0.085), 0.28, 0.17, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=chip[out], ec=OCHRE_D if hi else "none", lw=1.2 if hi else 0, zorder=4))
        txt(xx, yy, out, S_TXT, "bold", INK if out == "MFA" else "white")

# ------------------------------------------------------------------ (a) PEP and outcomes
path([(5.45, yc), (5.64, yc)])
sx, sy, s = 5.84, 2.86, 0.165
ax.add_patch(Polygon([(sx - s, sy + 0.8 * s), (sx, sy + 1.05 * s), (sx + s, sy + 0.8 * s), (sx + s, sy), (sx, sy - 1.0 * s),
                      (sx - s, sy)], closed=True, fc=NAVY, ec="none", zorder=4))
ax.plot([sx - 0.45 * s, sx - 0.1 * s, sx + 0.5 * s], [sy + 0.05 * s, sy - 0.35 * s, sy + 0.5 * s], color="white",
        lw=1.4, zorder=5, solid_capstyle="round")
txt(sx, 3.20, "PEP guard", S_TITLE, "bold")
txt(sx, 2.45, "no delivery while", S_NOTE, c=GRAY, style="italic")
txt(sx, 2.33, "MFA is pending", S_NOTE, c=GRAY, style="italic")
outs = [("Deny", RED, 3.12), ("Pending MFA", YEL, 2.84), ("Allow", GRN, 2.56)]
ax.plot([6.00, 6.10], [yc, yc], color=INK, lw=0.95, zorder=3)
ax.plot([6.10, 6.10], [2.56, 3.12], color=INK, lw=0.95, zorder=3)
for lab, col, yy in outs:
    path([(6.10, yy), (6.21, yy)])
    ax.add_patch(FancyBboxPatch((6.21, yy - 0.10), 0.74, 0.20, boxstyle="round,pad=0,rounding_size=0.05", fc=col,
                                ec="none", zorder=4))
    txt(6.58, yy, lab, S_TITLE, "bold", INK if col == YEL else "white")
DASH = (0, (3.2, 2.2))
for _, _, yy in outs:
    ax.plot([6.95, 7.04], [yy, yy], color=OCHRE_D, lw=0.95, ls=DASH, zorder=3)
txt(6.98, 2.36, "labels of all events", S_NOTE, c=OCHRE_D, style="italic", ha="right")

# ------------------------------------------------------------------ base action path
yb = 2.14
path([(ex, yc - er), (ex, yb), (4.42, yb), (4.42, 2.84), (4.55, 2.84)])
txt(ex + 0.08, yb + 0.075, "base action (Deny / MFA / Allow)", S_NOTE, c=INK, style="italic", ha="left")

# ------------------------------------------------------------------ (b) offline selection (real bank)
proto = json.loads((HERE.parent / "repro_check" / "protocol.lock.json").read_text(encoding="utf-8"))
bank = [b for b in proto["candidate_bank"] if b["family"] == "hysteresis"]
lo = np.array([b["parameters"]["low"] for b in bank]); hi = np.array([b["parameters"]["high"] for b in bank])
v = pd.read_csv(RES / "A7_all_candidates_validation.csv")
v = v[v.family == "hysteresis"].set_index("id")
feas = np.array([(v.loc[b["id"], "far_direct"] <= 0.005) and (v.loc[b["id"], "fdr_hard"] <= 0.05)
                 and (v.loc[b["id"], "mfa_benign"] <= 0.02) for b in bank])
sel = int(np.argmin([v.loc[b["id"], "sr_action"] if f else 9 for b, f in zip(bank, feas)]))
assert bank[sel]["id"] == "hysteresis_27"
bx0, bx1, by0, by1 = 0.44, 1.40, 0.40, 1.36
ab = sub_axes(bx0, by0, bx1, by1)
ab.scatter(lo[~feas], hi[~feas], s=10, facecolor="white", edgecolor="#a3afb9", lw=0.6, zorder=2)
ab.scatter(lo[feas], hi[feas], s=11, color=GREEN, lw=0, zorder=3)
ab.scatter([lo[sel]], [hi[sel]], s=80, marker="*", color=GREEN_D, edgecolor="white", lw=0.4, zorder=4)
ab.plot([0, 100], [0, 100], color="#c9d2d9", lw=0.6, ls=(0, (2, 2)), zorder=1)
ab.set_xlim(-3, 103); ab.set_ylim(-3, 103)
for sp in ("top", "right"):
    ab.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ab.spines[sp].set_color("#9aa7b2"); ab.spines[sp].set_linewidth(0.6)
ab.set_xticks([0, 50, 100]); ab.set_yticks([0, 50, 100])
ab.tick_params(labelsize=S_TICK, length=2, width=0.5, colors=GRAY, pad=1)
ab.set_xlabel(r"$\theta_{\rm low}$", fontsize=S_TXT, labelpad=0)
ab.set_ylabel(r"$\theta_{\rm high}$", fontsize=S_TXT, labelpad=0)
tx = 1.54
txt(tx, 1.52, "50 candidate pairs", S_TXT, ha="left")
ax.add_patch(Circle((tx + 0.035, 1.30), 0.03, fc=GREEN, ec="none", zorder=5))
txt(tx + 0.10, 1.30, "meets validation", S_NOTE, c=GRAY, ha="left")
txt(tx + 0.10, 1.18, "caps (Eq. 4)", S_NOTE, c=GRAY, ha="left")
txt(tx + 0.035, 0.96, r"$\bigstar$", S_TXT, c=GREEN_D)
txt(tx + 0.10, 0.96, "lowest validation SR", S_NOTE, c=GREEN_D, ha="left")
txt(tx, 0.74, r"$\theta^{*}=(66.83,\,98.83)$", S_TITLE, "bold", GREEN_D, ha="left")
# default theta* -> through the b/c gap -> hysteresis state
gx = 2.66
bridge(2.80, yb, "a")
path([(2.52, 0.74), (gx, 0.74), (gx, 1.98), (2.80, 1.98), (2.80, 2.30)], c=GREEN_D, lw=1.05)
txt(2.86, 2.222, r"default $\theta^{*}$", S_NOTE, c=GREEN_D, style="italic", ha="left")

# ------------------------------------------------------------------ (c) adaptation loop
cx, cy, rx, ry = 4.88, 0.84, 1.13, 0.52
ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fill=False, ec=OCHRE, lw=1.2, zorder=2))
for ang in (0, -90, 180, 90):  # clockwise flow
    a = math.radians(ang)
    px, py = cx + rx * math.cos(a), cy + ry * math.sin(a)
    tx_, ty_ = rx * math.sin(a), -ry * math.cos(a)
    n = math.hypot(tx_, ty_)
    ax.add_patch(FancyArrowPatch((px - 0.02 * tx_ / n, py - 0.02 * ty_ / n), (px + 0.02 * tx_ / n, py + 0.02 * ty_ / n),
                                 arrowstyle="-|>", mutation_scale=8.5, color=OCHRE, lw=0.01, zorder=3))
st = {}
labels = {40: ("delayed labels", r"after $D$ events", "left"),
          -40: ("enough labels?", r"if not, keep current $\theta$", "left"),
          -140: (r"GP-LCB on window $W$", "10 of 50 candidates", "right"),
          140: ("feasible?  commit", r"else revert to $\theta^{*}$", "right")}
for ang, (t1, t2, side) in labels.items():
    a = math.radians(ang)
    px, py = cx + rx * math.cos(a), cy + ry * math.sin(a)
    st[ang] = (px, py)
    ax.add_patch(Circle((px, py), 0.038, fc="white", ec=OCHRE_D, lw=1.1, zorder=4))
    dx = 0.10 if side == "left" else -0.10
    ha = "left" if side == "left" else "right"
    oy = 0.09 if py > cy else -0.09
    txt(px + dx, py + oy + 0.06, t1, S_TITLE, "bold", ha=ha)
    txt(px + dx, py + oy - 0.065, t2, S_NOTE, c=GRAY, ha=ha, style="italic")
# inside the loop: real candidate plane with the GP-LCB queries of the first supported update (2% cap)
upd = json.loads((ORIG / "adaptive_updates_clean_D256.json").read_text(encoding="utf-8"))["updates"]
first = next(u for u in upd if u["supported"])
q = next(s for s in first["selections"] if s["family"] == "hysteresis" and abs(s["budget"] - 0.02) < 1e-9)
qi, chosen = list(q["queried_global_indices"]), q["selected_global_index"]
ax.add_patch(FancyBboxPatch((cx - 0.43, cy - 0.33), 0.86, 0.66, boxstyle="round,pad=0,rounding_size=0.04",
                            fc="white", ec="#ead7c6", lw=0.6, zorder=1.5))
ac = sub_axes(cx - 0.40, cy - 0.30, cx + 0.40, cy + 0.30)
ac.scatter(lo, hi, s=7, facecolor="white", edgecolor="#b9c2c9", lw=0.5, zorder=2)
ac.scatter(lo[qi], hi[qi], s=14, facecolor=OCHRE, edgecolor="none", zorder=3)
ac.scatter([lo[chosen]], [hi[chosen]], s=55, marker="*", color=OCHRE_D, edgecolor="white", lw=0.4, zorder=4)
ac.set_xlim(-5, 105); ac.set_ylim(-5, 105); ac.axis("off")
# delayed labels into the loop
dlx, dly = st[40]
path([(7.04, 3.12), (7.04, 1.58), (dlx, 1.58), (dlx, dly + 0.045)], c=OCHRE_D, ls=DASH)
# configuration update: from 'feasible?' station, around the heading, into the hysteresis state
fx, fy = st[140]
bridge(3.72, yb, "a")
path([(fx, fy + 0.045), (fx, 1.62), (4.28, 1.62), (4.28, 1.98), (3.72, 1.98), (3.72, 2.30)], c=OCHRE_D, lw=1.1)
txt(fx + 0.06, 1.47, "configuration update", S_NOTE, c=OCHRE_D, style="italic", ha="left")

fig.savefig(OUT / "RAAH_ZT_framework_final.pdf")
fig.savefig(OUT / "RAAH_ZT_framework_final.png", dpi=300)

# ------------------------------------------------------------------ automatic overlap check of text boxes
fig.canvas.draw()
r = fig.canvas.get_renderer()
boxes = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
bad = [(a[0], b[0]) for i, a in enumerate(boxes) for b in boxes[i + 1:] if a[1].overlaps(b[1])]
print("text overlaps:", bad if bad else "none")
fw, fh = fig.bbox.width, fig.bbox.height
out_of_bounds = [t for t, bb in boxes if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fw or bb.y1 > fh]
print("text outside figure:", out_of_bounds if out_of_bounds else "none")
sizes = sorted({round(t.get_fontsize(), 1) for t in ax.texts if t.get_text().strip()})
print("font sizes used:", sizes)
hits = []
for ln in ax.lines:
    xy = ax.transData.transform(ln.get_xydata())
    for (x0_, y0_), (x1_, y1_) in zip(xy[:-1], xy[1:]):
        for k in range(41):
            px_, py_ = x0_ + (x1_ - x0_) * k / 40, y0_ + (y1_ - y0_) * k / 40
            for t_, bb in boxes:
                if bb.x0 + 1 < px_ < bb.x1 - 1 and bb.y0 + 1 < py_ < bb.y1 - 1:
                    hits.append(t_)
print("lines crossing text:", sorted(set(hits)) if hits else "none")
