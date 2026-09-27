"""
figS7_panels.py  --  Figure S7: modulation coefficient of a sphere.

    (a) G versus diameter for delta_eps = 0.051, NA = 1.4, lambda = 500 nm:
        the small-object closed form of Eq. (S15), the projection model of
        Eq. (S14), and the three-dimensional first-Born field of the sphere
    (b) the corresponding label-free photon cost with the fluorescence costs at
        beta = 3 and 10; the shaded band marks the bead diameters of the experiment

The optical model, the coupling G(d) and the budgets come from fig1_panels.py.

Run:  python figS7_panels.py        (needs fig1_panels.py in the same folder)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig1_panels import Optics, N_F, N_L, DEPS, FL, LF

def figS7(opt, dgrid, Gth, G3, out="Fig_S7_coupling.png"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axs[0]
    ax.plot(dgrid * 1e3, opt.G_point(dgrid), ":", color="0.3", lw=2, label="closed form, k₀δεV/(n A_eff)")
    ax.plot(dgrid * 1e3, Gth, "-", color=LF, lw=2.5, label="projection model")
    ax.plot(dgrid * 1e3, G3, "--", color="#e65100", lw=1.8, label="3D first-Born field")
    ax.axhline(1, color="0.6", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(120, 3000); ax.set_ylim(1e-3, 3)
    ax.set_xlabel("object diameter (nm)"); ax.set_ylabel("coupling G")
    ax.legend(fontsize=10); ax.grid(True, which="both", alpha=0.2); ax.set_title("(a)", loc="left")
    ax = axs[1]
    ax.plot(dgrid * 1e3, N_L(opt.G_point(dgrid)), ":", color="0.3", lw=2, label="closed form (∝ d⁻⁶)")
    ax.plot(dgrid * 1e3, N_L(Gth), "-", color=LF, lw=2.5, label="projection model")
    ax.plot(dgrid * 1e3, N_L(G3), "--", color="#e65100", lw=1.8, label="3D first-Born field")
    ax.axhline(N_F(3), color=FL, lw=2, label="fluorescence, β = 3")
    ax.axhline(N_F(10), color=FL, lw=1.6, ls="--", label="fluorescence, β = 10")
    ax.axvspan(350, 660, color="0.9", zorder=0); ax.text(360, 1.5e2, "measured\nbeads", fontsize=10, color="0.4")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(120, 3000); ax.set_ylim(1e2, 1e6)
    ax.set_xlabel("object diameter (nm)"); ax.set_ylabel("photons to reach SNR* = 10")
    ax.legend(fontsize=10); ax.grid(True, which="both", alpha=0.2); ax.set_title("(b)", loc="left")
    fig.savefig(out, dpi=600, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    opt = Optics(NA=1.4, lam=0.5)
    dgrid = np.geomspace(0.12, 3.0, 160)
    Gth = opt.G_curve(dgrid, "thin")
    G3 = opt.G_curve(dgrid, "3d")
    print(" d(um)   G_closed  G_proj   G_3D")
    for d in (0.20, 0.35, 0.42, 0.50, 0.58, 0.66, 1.00, 1.50):
        i = np.argmin(np.abs(dgrid - d))
        print(f" {d:4.2f}    {opt.G_point(d):6.3f}   {Gth[i]:6.3f}   {G3[i]:6.3f}")
    sel = (dgrid >= 0.35) & (dgrid <= 0.66)
    for name, Garr in [("closed form", opt.G_point(dgrid)), ("projection", Gth), ("3D Born", G3)]:
        s = np.polyfit(np.log(dgrid[sel]), np.log(N_L(Garr[sel])), 1)[0]
        print(f" effective exponent of N_L vs d over 0.35-0.66 um, {name:11s}: {s:6.2f}")
    figS7(opt, dgrid, Gth, G3)
    print(" saved Fig_S7_coupling.png")
