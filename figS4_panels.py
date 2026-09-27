"""
figS4_panels.py  --  Figure S4 of the revised Supporting Information.

    (a) attainment of the bound. Relative precision of the two arms versus
        detected photons for the 0.5 um bead, from the measured photon series in
        photon_series.xlsx, against the Cramer-Rao bounds of Eqs. (S6) and (S12)
        including the read-noise factor (1 + 4 M_cell sigma_r^2 / N)^(1/2).
    (b) bias of the four-step label-free estimator on synthetic Poisson data at
        the measured coupling G = 0.16, versus the target fidelity SNR*.

Panel (b) forward model, Eq. (S9) of the Supporting Information:

    mu_q = R [ 1 + G cos(chi_q - psi) + G^2/4 ],   chi_q = 0, pi/2, pi, 3pi/2

with the four-step quadratures of Eq. (S11). Two conventions for the reference
level are available and give different high-fidelity behaviour:
    KNOWN_R = True   R measured separately; the residual bias comes from taking
                     the magnitude of the two quadratures and falls with fidelity
    KNOWN_R = False  R = sum_q I_q / 4 from the same four frames; the bias then
                     approaches -G^2/4 at high fidelity
Set KNOWN_R to whichever the experimental analysis uses.

Run:  python figS4_panels.py [photon_series.xlsx]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, LogLocator

from fig3_analysis import (read_data, pick, bound_fluor, bound_labelfree, BETA, TARGET,
                           FL, LF, G_MEAS)

# ------------------------------------------------------------------ panel (b) settings
KNOWN_R = True                 # see the note above
PSI = np.pi / 2                # phase of the scattered field, pure-phase object
SNRS = np.array([4, 5, 6.5, 8, 10, 13, 17, 22, 30, 42])
NTRIAL = 200_000
SEED = 20260908

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 15, "axes.labelsize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 11.5,
    "axes.linewidth": 1.2})


def four_step(R, G, psi=PSI, ntrial=NTRIAL, rng=None, chunk=50_000, known_R=KNOWN_R):
    """relative bias, relative spread and Monte Carlo standard error of G_hat."""
    rng = rng or np.random.default_rng(SEED)
    chi = np.arange(4) * np.pi / 2
    mu = R * (1 + G * np.cos(chi - psi) + G ** 2 / 4)
    s1 = s2 = 0.0
    done = 0
    while done < ntrial:
        m = min(chunk, ntrial - done)
        I = np.array([rng.poisson(mu[q], m) for q in range(4)], dtype=float)
        Rn = R if known_R else I.sum(axis=0) / 4.0
        Gh = np.hypot((I[0] - I[2]) / (2 * Rn), (I[1] - I[3]) / (2 * Rn))
        s1 += Gh.sum(); s2 += (Gh ** 2).sum()
        done += m
    mean = s1 / ntrial
    sd = np.sqrt(max(s2 / ntrial - mean ** 2, 0.0))
    return (mean - G) / G, sd / G, sd / (np.sqrt(ntrial) * G)


def panel_a(ax, series):
    NF, relF, errF, _ = pick(series, "F", "3")
    NL, relL, errL, _ = pick(series, "L")
    Ng = np.geomspace(60, 1.6e5, 300)
    ax.plot(Ng, bound_fluor(Ng, BETA["3"]), color=FL, lw=2.2, zorder=4,
            label="Cram\u00e9r\u2013Rao bound, fluorescence")
    ax.plot(Ng, bound_labelfree(Ng), color=LF, lw=2.2, ls="-", zorder=4,
            label="Cram\u00e9r\u2013Rao bound, label-free")
    ax.errorbar(NF, relF, yerr=errF, fmt="o", color=FL, mec="k", mew=0.8, ms=8,
                capsize=2.5, linestyle="None", zorder=6,
                label="measured  $\\sigma_A/A$   ($p=1$)")
    ax.errorbar(NL, relL, yerr=errL, fmt="s", color=LF, mec="k", mew=0.8, ms=7.5,
                capsize=2.5, linestyle="None", zorder=6,
                label="measured  $\\sigma_G/G$   ($p=2$)")
    ax.axhline(TARGET, ls="--", color="0.38", lw=1.4, zorder=3)
    ax.text(72, TARGET * 1.10, "$1/\\mathrm{SNR}^{\\ast}=0.1$", color="0.30",
            fontsize=12.5, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(60, 1.6e5); ax.set_ylim(0.01, 2)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.set_xlabel("detected photons  $N$")
    ax.set_ylabel("relative precision  $\\sigma$ / value")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.set_title("(a)", loc="left", fontsize=16)


def panel_b(ax, rng=None):
    rng = rng or np.random.default_rng(SEED)
    bias, err = [], []
    print(" SNR*      N = 4R     bias (%)    sigma_G/G")
    for snr in SNRS:
        N = 2 * snr ** 2 / G_MEAS ** 2
        b, s, se = four_step(N / 4, G_MEAS, rng=rng)
        bias.append(100 * b); err.append(100 * se)
        print(f" {snr:5.1f}  {N:10.0f}   {100*b:+8.3f}    {s:8.4f}  (target {1/snr:.4f})")
    bias, err = np.array(bias), np.array(err)
    b10 = np.interp(10, SNRS, bias)

    ax.axhline(0, color="0.5", lw=1)
    ax.errorbar(SNRS, bias, yerr=err, fmt="s-", color=LF, mec="k", mew=0.7, ms=7,
                lw=1.8, capsize=3, zorder=5,
                label="four-step estimator,  $G = %.2f$" % G_MEAS)
    if not KNOWN_R:
        ax.axhline(-100 * G_MEAS ** 2 / 4, color="0.55", lw=1.0, ls=":",
                   label="dark-field limit  $-G^{2}/4$")
    ax.axvline(10, color="0.38", lw=1.4, ls="--", zorder=3, label="operating point")
    ax.annotate(f"{b10:+.2f}% at SNR* = 10", (10, b10), textcoords="offset points",
                xytext=(16, 20), fontsize=12.5, color="k",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.1))
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 40]); ax.set_xticklabels(["5", "10", "20", "40"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ymax = np.ceil(max(bias))
    ax.set_ylim(-0.2, ymax * 1.30)
    ax.set_yticks(np.linspace(0, ymax, 5))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("target  SNR*")
    ax.set_ylabel("relative bias of  $G$  (%)", labelpad=2)
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_title("(b)", loc="left", fontsize=16)
    print(f"\n bias at SNR* = 10: {b10:+.3f}%"
          f"   (reference {'measured separately' if KNOWN_R else 'from the same four frames'})")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "photon_series.xlsx"
    series, _, _ = read_data(path)
    fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.6))
    panel_a(axs[0], series)
    panel_b(axs[1])
    fig.tight_layout(w_pad=3.0)
    fig.savefig("Figure_S4.png", dpi=600, bbox_inches="tight")
    fig.savefig("Figure_S4.pdf", bbox_inches="tight")
    print(" saved Figure_S4.png / .pdf")
