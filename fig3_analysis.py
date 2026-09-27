"""
fig3_analysis.py  --  Figures 3(d), 3(e) and 3(f) from the measured photon series.

Input   photon_series.xlsx  (three sheets, see read_data below)
          sheet 1  per-photon-level series: diameter, arm, background condition,
                   N, frames, estimator mean, estimator sd, relative precision
          sheet 2  per-frame normalized estimates at the operating point (insets)
          sheet 3  per-size costs with bead-to-bead standard deviations

Conventions (Supporting Information)
  N        detected photons in one resolution cell (A_eff = lambda^2/(pi NA^2),
           M_cell ~ 10 pixels of 65 nm): signal + background in fluorescence,
           reference photons summed over the four phase steps in the label-free arm
  bound    fluorescence  sigma_A/A = sqrt{[1 + beta(1 + M_cell/M_ann)] / A},  A = N/(1+beta)
           label-free    sigma_G/G = sqrt{2/(N G^2)}
           both multiplied by the read-noise factor sqrt(1 + 4 M_cell sigma_r^2 / N)
  cost     N at the target 1/SNR* = 0.1, from a free-slope log-log fit of the series
  crossover  intersection of the fitted label-free power law N_L ~ d^-alpha with the
             fluorescence cost of that background condition

Outputs  Fig_3d_fluor_series.png, Fig_3e_labelfree_series.png, Fig_3f_crossover.png
         and a printed summary. Figure S4 is produced by figS4_panels.py, which
         imports the bounds and the reader defined here.

Run:  python fig3_analysis.py [photon_series.xlsx]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
from openpyxl import load_workbook

# ---------------------------------------------------------------- parameters
SNR = 10.0
TARGET = 1.0 / SNR
M_CELL = 10           # pixels in one resolution cell
M_ANN = 196           # pixels in the background annulus
SIG_R = {"F": 1.3, "L": 1.4}      # read noise, e- rms, per camera
G_MEAS = 0.16         # measured modulation of the 0.5 um bead
BETA = {"3": 3.0, "9.3": 9.3}
FL, LF = "#6a1b9a", "#1565c0"
STAR_FC, STAR_EC = "#B48A30", "#2F2A20"

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 15, "axes.labelsize": 19,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 12, "axes.linewidth": 1.2})


# ---------------------------------------------------------------- data
def read_data(path):
    wb = load_workbook(path, data_only=True)
    series, frames, sizes = [], {"F": [], "L": []}, []
    for r in wb[wb.sheetnames[0]].iter_rows(min_row=2, values_only=True):
        if isinstance(r[0], (int, float)):
            series.append(dict(d=float(r[0]), arm=r[1], cond=str(r[2]), N=float(r[3]),
                               nframes=int(r[4]), mean=float(r[5]), sd=float(r[6]), rel=float(r[7])))
    for r in wb[wb.sheetnames[1]].iter_rows(min_row=2, values_only=True):
        if r[0] in ("F", "L") and isinstance(r[2], (int, float)):
            frames[r[0]].append(float(r[2]))
    for r in wb[wb.sheetnames[2]].iter_rows(min_row=2, values_only=True):
        if isinstance(r[0], (int, float)):
            sizes.append(dict(d=float(r[0]), NF=float(r[1]), NF_sd=float(r[2]), NL=float(r[3]),
                              NL_sd=float(r[4]), G=float(r[5]), G_sd=float(r[6]), n=int(r[7])))
    return series, {k: np.array(v) for k, v in frames.items()}, sizes


def pick(series, arm, cond=None):
    sel = [s for s in series if s["arm"] == arm and (cond is None or s["cond"] == cond)]
    sel.sort(key=lambda s: s["N"])
    N = np.array([s["N"] for s in sel]); rel = np.array([s["rel"] for s in sel])
    nfr = np.array([s["nframes"] for s in sel])
    return N, rel, rel / np.sqrt(2 * (nfr - 1)), sel      # error on a standard deviation


# ---------------------------------------------------------------- model
def readnoise_factor(N, arm):
    return np.sqrt(1 + 4 * M_CELL * SIG_R[arm] ** 2 / N)


def bound_fluor(N, beta):
    A = N / (1 + beta)
    return np.sqrt((1 + beta * (1 + M_CELL / M_ANN)) / A) * readnoise_factor(N, "F")


def bound_labelfree(N, G=G_MEAS):
    return np.sqrt(2 / (N * G ** 2)) * readnoise_factor(N, "L")


def fit_cost(N, rel, target=TARGET):
    """free-slope log-log fit; returns cost at target, slope, and their standard errors."""
    x, y = np.log(N), np.log(rel)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = (resid ** 2).sum() / (len(x) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    slope, inter = coef
    logNt = (np.log(target) - inter) / slope
    # propagate to the cost
    g = np.array([-logNt / slope, -1 / slope])
    var = g @ cov @ g
    return np.exp(logNt), slope, np.sqrt(cov[0, 0]), np.exp(logNt) * np.sqrt(var)


def fit_power(d, y, w=None):
    """weighted log-log power-law fit y = C d^-alpha; returns alpha, sigma_alpha, log C."""
    x, ly = np.log(d), np.log(y)
    W = np.ones_like(x) if w is None else 1.0 / np.asarray(w) ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    Aw, yw = A * W[:, None] ** 0.5, ly * W ** 0.5
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    resid = ly - A @ coef
    s2 = (W * resid ** 2).sum() / (len(x) - 2) / W.mean()
    cov = s2 * np.linalg.inv(Aw.T @ Aw) * W.mean()
    return -coef[0], np.sqrt(cov[0, 0]), coef[1]


def crossover(alpha, logC, NF):
    return np.exp((logC - np.log(NF)) / alpha)


# ---------------------------------------------------------------- panels
def panel_series(N, rel, err, bound, colour, ylabel, cost, label_cost, out,
                 frames=None, inset_title="", inset_xlabel="", bound_label=""):
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    Ng = np.geomspace(N.min() * 0.6, N.max() * 1.6, 300)
    ax.plot(Ng, bound(Ng), color=colour, lw=2.8, zorder=4, label=bound_label)
    ax.errorbar(N, rel, yerr=err, fmt="o", color=colour, mec="k", mew=0.8, ms=8.5,
                capsize=2.5, zorder=6, label="measured points")
    ax.axhline(TARGET, ls="--", color="0.38", lw=1.5, zorder=3)
    ax.text(N.min() * 0.65, TARGET * 1.08, "target  $1/\\mathrm{SNR}^{\\ast}=0.1$",
            color="0.30", fontsize=13, va="bottom", ha="left")
    ax.plot([cost], [TARGET], marker="*", linestyle="None", markerfacecolor=STAR_FC,
            markeredgecolor=STAR_EC, markeredgewidth=0.9, ms=18, zorder=8, label="photon cost at target")
    ax.annotate(label_cost, (cost, TARGET), textcoords="offset points", xytext=(12, 12),
                fontsize=14, color="k", arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3))
    ax.plot([cost, cost], [rel.min() * 0.4, TARGET], ls=":", color=colour, lw=1.5, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(N.min() * 0.6, N.max() * 1.6); ax.set_ylim(rel.min() * 0.5, rel.max() * 2)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("detected photons  $N$"); ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.18); ax.legend(loc="lower left", framealpha=0.95)
    if frames is not None:
        axin = fig.add_axes([0.585, 0.60, 0.29, 0.25])
        axin.hist(frames - 1, bins=18, density=True, color=colour, alpha=0.65,
                  edgecolor="white", lw=0.7)
        xx = np.linspace(-0.45, 0.45, 300)
        axin.plot(xx, np.exp(-xx ** 2 / (2 * TARGET ** 2)) / (TARGET * np.sqrt(2 * np.pi)),
                  color="k", lw=1.6, ls="--", alpha=0.8)
        axin.axvline(0, color="0.4", lw=0.8)
        axin.set_title(inset_title, fontsize=11.5); axin.set_xlabel(inset_xlabel, fontsize=11)
        axin.set_xticks([-0.2, 0, 0.2]); axin.set_yticks([]); axin.tick_params(labelsize=10)
        axin.set_xlim(-0.42, 0.42)
        for sp in ("top", "right", "left"): axin.spines[sp].set_visible(False)
    fig.savefig(out, dpi=600, bbox_inches="tight"); plt.close(fig)


def panel_crossover(sizes, costs, alpha, sa, logC, out, model=None):
    d = np.array([s["d"] for s in sizes])
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    dg = np.geomspace(0.30, 1.30, 300)
    ax.plot(dg * 1e3, np.exp(logC) * dg ** (-alpha), color=LF, lw=3.0, zorder=4,
            label=f"fit  $N_{{\\rm L}}\\propto d^{{-{alpha:.2f}}}$")
    if model is not None:
        ax.plot(dg * 1e3, model(dg), color=LF, lw=1.6, ls=":", zorder=3, label="model, Eq. (S14)")
    for cond, colour, ls, lab in [("3", FL, "-", "fluorescence, $\\beta=3$"),
                                  ("9.3", FL, "--", "fluorescence, $\\beta=9.3$")]:
        ax.axhline(costs[cond], color=colour, lw=3.0 if cond == "3" else 2.0, ls=ls, zorder=4, label=lab)
        dx = crossover(alpha, logC, costs[cond])
        ax.plot([dx * 1e3], [costs[cond]], marker="*", linestyle="None", markerfacecolor=STAR_FC,
                markeredgecolor=STAR_EC, markeredgewidth=0.9, ms=19, zorder=9)
        ax.annotate(f"{dx*1e3:.0f} nm", (dx * 1e3, costs[cond]), textcoords="offset points",
                    xytext=(10, 10), fontsize=12.5)
    ax.errorbar(d * 1e3, [s["NF"] for s in sizes], yerr=[s["NF_sd"] for s in sizes], fmt="o",
                color=FL, mec="k", mew=0.8, ms=8.5, capsize=3, zorder=7, label="measured  $N_{\\rm F}$")
    ax.errorbar(d * 1e3, [s["NL"] for s in sizes], yerr=[s["NL_sd"] for s in sizes], fmt="s",
                color=LF, mec="k", mew=0.8, ms=8.0, capsize=3, zorder=7, label="measured  $N_{\\rm L}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(310, 1500); ax.set_ylim(7e2, 1.2e5)
    ax.set_xticks([400, 500, 700, 1000]); ax.set_xticklabels(["400", "500", "700", "1000"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("bead diameter (nm)"); ax.set_ylabel("photons to reach SNR* = 10")
    ax.grid(True, which="both", alpha=0.18)
    
    ax.text(1420, 2.0e3, "label-free\ncheaper", color="#08519c", fontsize=12.5, weight="bold", ha="right", va="center")
    ax.text(330, 9.0e2, "fluorescence\ncheaper", color=FL, fontsize=12.5, weight="bold", ha="left", va="center")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10.5, ncol=1)
    fig.savefig(out, dpi=600, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "photon_series.xlsx"
    series, frames, sizes = read_data(path)

    costs, fits = {}, {}
    for arm, cond, tag in [("F", "3", "F3"), ("F", "9.3", "F9.3"), ("L", None, "L")]:
        N, rel, err, _ = pick(series, arm, cond)
        cost, slope, sslope, scost = fit_cost(N, rel)
        fits[tag] = (N, rel, err, cost, slope, sslope, scost)
        costs[cond if cond else "L"] = cost
        print(f"{tag:5s}: slope {slope:+.3f} ± {sslope:.3f}   cost at target = {cost:.0f} ± {scost:.0f}")

    # ratio of measured precision to the read-noise-corrected bound
    for tag, bfun in [("F3", lambda N: bound_fluor(N, BETA["3"])),
                      ("F9.3", lambda N: bound_fluor(N, BETA["9.3"])), ("L", bound_labelfree)]:
        N, rel = fits[tag][0], fits[tag][1]
        r = bfun(N) / rel
        print(f"{tag:5s}: efficiency ratio r per level " + " ".join(f"{v:.2f}" for v in r) +
              f"   mean {r.mean():.2f}")

    d = np.array([s["d"] for s in sizes]); NL = np.array([s["NL"] for s in sizes])
    NL_sd = np.array([s["NL_sd"] for s in sizes])
    alpha, sa, logC = fit_power(d, NL, NL_sd)
    alpha4, sa4, _ = fit_power(d[:4], NL[:4], NL_sd[:4])
    print(f"label-free size exponent: alpha = {alpha:.2f} ± {sa:.2f} (5 beads), "
          f"{alpha4:.2f} ± {sa4:.2f} (without the 0.66 um bead)")
    for cond in ("3", "9.3"):
        dx = crossover(alpha, logC, costs[cond])
        print(f"crossover at beta = {cond}: d = {dx*1e3:.0f} nm")

    panel_series(*fits["F3"][:3], lambda N: bound_fluor(N, BETA["3"]), FL,
                 "relative precision  $\\sigma_A/A$", fits["F3"][3],
                 f"$N_{{\\rm F}}\\simeq{fits['F3'][3]:.0f}$ photons",
                 "Fig_3d_fluor_series.png", frames=frames["F"],
                 inset_title="frames at $N_{\\rm F}$", inset_xlabel="$(\\hat A-A)/A$",
                 bound_label="Cram\u00e9r\u2013Rao bound")
    panel_series(*fits["L"][:3], bound_labelfree, LF,
                 "relative precision  $\\sigma_G/G$", fits["L"][3],
                 f"$N_{{\\rm L}}\\simeq{fits['L'][3]:.0f}$ photons",
                 "Fig_3e_labelfree_series.png", frames=frames["L"],
                 inset_title="frames at $N_{\\rm L}$", inset_xlabel="$(\\hat G-G)/G$",
                 bound_label="Cram\u00e9r\u2013Rao bound")
    panel_crossover(sizes, costs, alpha, sa, logC, "Fig_3f_crossover.png")
    print("\nsaved Fig_3d_fluor_series.png, Fig_3e_labelfree_series.png, Fig_3f_crossover.png")
