"""
figS2_panels.py  --  Figure S2: crossover diameter versus numerical aperture.

Crossover diameter at beta = 3 for permittivity contrasts 0.02, 0.05, 0.1 and 0.3
at 500 nm, and 0.05 at 400 nm, computed from the finite-object modulation
coefficient of Eq. (S14) (projection model). G is linear in delta_eps under the
first Born approximation, so one field per (NA, lambda) serves all contrasts.

The optical model and the budgets come from fig1_panels.py.

Run:  python figS2_panels.py        (needs fig1_panels.py in the same folder)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig1_panels import Optics, N_F, N_L, crossover

def figS2(out="Fig_S2_accessibility.png"):
    """crossover diameter versus NA at beta = 3 (finite-sphere model, markers)."""
    NAs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.45])
    cases = [(0.02, 0.5, "δε = 0.02"), (0.05, 0.5, "δε = 0.05"), (0.1, 0.5, "δε = 0.1"),
             (0.3, 0.5, "δε = 0.3"), (0.05, 0.4, "δε = 0.05, λ = 400 nm")]
    markers = ["o", "s", "^", "D", "x"]
    dgrid = np.geomspace(0.1, 6.0, 60)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    res, cache = {}, {}
    for lam in (0.5, 0.4):                      # G is linear in deps: compute once per (NA, lam) at deps = 1
        for NA in NAs:
            opt = Optics(NA=NA, lam=lam, px=0.05, L=10.0)
            cache[(NA, lam)] = np.array([opt.G_cell(opt.E_thin(d, 1.0)) for d in dgrid])
    for (deps, lam, lab), mk in zip(cases, markers):
        dx = np.array([crossover(dgrid, cache[(NA, lam)] * deps, 3.0) for NA in NAs])
        res[lab] = dx
        ax.plot(NAs, dx * 1e3, marker=mk, ms=6, lw=1.6, label=lab)
    ax.axhspan(100, 1000, color="#fff3c4", alpha=0.7, zorder=0)
    ax.text(0.25, 130, "available bead range (100 nm – 1 μm)", fontsize=11, color="#8a6d00")
    ax.set_yscale("log"); ax.set_xlim(0.15, 1.5)
    ax.set_xlabel("NA"); ax.set_ylabel("crossover diameter (nm)")
    ax.legend(title="β = 3, λ = 500 nm unless noted", fontsize=10)
    ax.grid(True, which="both", alpha=0.2)
    fig.savefig(out, dpi=600, bbox_inches="tight"); plt.close(fig)
    return NAs, res


if __name__ == "__main__":
    NAs, res = figS2()
    print(" Figure S2 crossover diameters (nm) at beta = 3:")
    for lab, dx in res.items():
        print(f"  {lab:26s}: " + " ".join(f"{v*1e3:5.0f}" if np.isfinite(v) else "  nan" for v in dx))
    print(" saved Fig_S2_accessibility.png")
