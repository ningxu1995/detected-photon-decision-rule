"""
figS5_panels.py  --  Figure S5, delivered-dose sensitivity analysis.

    (a) dose-referenced decision map over bead diameter and the fluorescence
        detected fraction eta_F, at the low-background condition, with the dose
        boundary G(1+beta) = sqrt(2 eta_F / eta_L)
    (b) crossover diameter versus eta_F for the two background conditions, with
        the detected-photon limits (eta_F = eta_L = 1) as references

Updated from the submitted version in three ways:
    * G(d) is the finite-object coupling of Eq. (S14), taken from fig1_panels.py,
      instead of the small-object closed form G = (d/d0)^3. The two agree only
      below the resolution cell, and the beads of the experiment are above it.
    * the background conditions are the measured beta = 3.0 and beta = 9.3
    * the reference lines are the detected-photon crossovers of the revision

Run:  python figS5_panels.py        (needs fig1_panels.py in the same folder)
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from fig1_panels import Optics, DEPS

# ------------------------------------------------------------------ parameters
BETA_LO, BETA_HI = 3.0, 9.3        # measured background ratios
ETA_L = 1.0                        # probe utilization of the label-free arm
ETA_F0 = 0.04                      # baseline fluorescence detected fraction
DGRID = np.geomspace(0.08, 3.0, 600)

mpl.rcParams.update({                     # set explicitly: fig1_panels sets larger sizes
    "font.size": 9, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "stixsans",
    "axes.labelsize": 9, "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 6.5, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3})

opt = Optics(NA=1.4, lam=0.5)
GCURVE = opt.G_curve(DGRID, "thin")          # coupling of a sphere, Eq. (S14)


def G_of_d(d):
    """finite-object coupling, interpolated on the precomputed curve."""
    return np.exp(np.interp(np.log(np.asarray(d, float)), np.log(DGRID), np.log(GCURVE)))


def d_cross(beta, etaF, etaL=ETA_L):
    """diameter at which G (1 + beta) = sqrt(2 eta_F / eta_L)."""
    t = np.sqrt(2.0 * np.asarray(etaF, float) / etaL) / (1.0 + beta)
    return np.exp(np.interp(np.log(t), np.log(GCURVE), np.log(DGRID)))


if __name__ == "__main__":
    d_det_lo, d_det_hi = d_cross(BETA_LO, ETA_L), d_cross(BETA_HI, ETA_L)
    print(f" detected-photon crossovers: beta = {BETA_LO}: {d_det_lo:.3f} um, "
          f"beta = {BETA_HI}: {d_det_hi:.3f} um")
    for e in (0.01, ETA_F0, 0.1):
        print(f" eta_F = {e:4.2f}:  d_x = {d_cross(BETA_LO, e):.3f} um (beta = {BETA_LO}), "
              f"{d_cross(BETA_HI, e):.3f} um (beta = {BETA_HI})")

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # ---- panel (a) -----------------------------------------------------------
    d = np.linspace(120, 1400, 500) / 1e3
    etaF = np.logspace(-2, 0, 500)
    D, E = np.meshgrid(d, etaF)
    ratio = np.log10(((1 + BETA_LO) ** 2) * (G_of_d(D) ** 2) * ETA_L / (2.0 * E))
    pc = ax[0].pcolormesh(D, E, ratio, cmap="RdBu_r", vmin=-2, vmax=2, shading="auto",
                          rasterized=True)
    eF = np.logspace(-2, 0, 300)
    ax[0].plot(d_cross(BETA_LO, eF), eF, color="gold", lw=1.6)
    ax[0].set_yscale("log")
    ax[0].axhline(ETA_F0, ls="--", color="w", lw=1.0)
    ax[0].text(1.36, ETA_F0 * 1.12, r"$\eta_F=0.04$", fontsize=7, color="w", ha="right",
               va="bottom")
    ax[0].text(0.95, 0.018, "label-free\nfavored", fontsize=7.0, ha="center", color="w")
    ax[0].text(0.30, 0.30, "fluorescence\nfavored", fontsize=7.0, ha="center", color="w")
    ax[0].set_xlabel(r"bead diameter ($\mu$m)")
    ax[0].set_ylabel(r"fluorescence detected fraction $\eta_F$")
    ax[0].set_title(r"dose-referenced decision map ($\beta=3$)", fontsize=8.5)
    ax[0].set_xlim(0.12, 1.4); ax[0].set_ylim(0.01, 1.0)
    cb = fig.colorbar(pc, ax=ax[0], fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_{10}(D_F/D_L)$", fontsize=8)

    # ---- panel (b) -----------------------------------------------------------
    eF = np.logspace(-2, -1, 300)
    ax[1].plot(eF, d_cross(BETA_LO, eF), color="#08519c", lw=1.8, label=r"$\beta=3$")
    ax[1].plot(eF, d_cross(BETA_HI, eF), color="#6baed6", lw=1.8, ls="--", label=r"$\beta=9.3$")
    ax[1].axhline(d_det_lo, color="#08519c", lw=0.8, ls=":")
    ax[1].axhline(d_det_hi, color="#6baed6", lw=0.8, ls=":")
    ax[1].text(0.0101, d_det_lo * 1.02,
               rf"detected limit, $\beta=3$ ({d_det_lo:.2f} $\mu$m)", fontsize=6.3,
               color="#08519c")
    ax[1].text(0.0101, d_det_hi * 1.02,
               rf"detected limit, $\beta=9.3$ ({d_det_hi:.2f} $\mu$m)", fontsize=6.3,
               color="#3182bd")
    ax[1].axvline(ETA_F0, ls="--", color="0.4", lw=0.9)
    ax[1].text(ETA_F0 * 1.08, 0.55, r"$\eta_F=0.04$", fontsize=7, color="0.3", rotation=90,
               va="bottom")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"fluorescence detected fraction $\eta_F$")
    ax[1].set_ylabel(r"crossover diameter $d_\times$ ($\mu$m)")
    ax[1].set_title("crossover vs detected fraction", fontsize=8.5)
    ax[1].legend(fontsize=6.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax[1].set_ylim(0.10, 1.10)

    for a in ax:
        a.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig("Figure_S5.png", dpi=600, bbox_inches="tight")
    plt.savefig("Figure_S5.pdf", bbox_inches="tight")
    print(" saved Figure_S5.png / .pdf")
