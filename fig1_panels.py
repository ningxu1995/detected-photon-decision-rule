"""
fig1_panels.py  --  Figure 1(b) and Figure 1(c) of the revised manuscript.

    Fig_1b_budget.png   single-object photon budget versus diameter, with the
                        finite-sphere coupling (solid) and its small-object
                        closed form (dotted)
    Fig_1c_map.png      decision map at d = 0.5 um over (delta_eps, beta)

Extracted from figures_theory.py, which also produces Figures S2 and S7; the
physics, the conventions and the numbers are identical in the two scripts.

Conventions (Supporting Information, revised):
  N      detected photons inside the object's image aperture, all frames
         (signal + background in fluorescence; reference + scattered in label-free)
  beta   background photons / signal photons in that aperture
  G      fringe-modulation coefficient of the object, G = 2|E_sc/E_R| (rms over the
         aperture for an extended object); mu_q = R[1 + G cos(chi_q - theta) + G^2/4]
  N_F    = SNR*^2 (1+beta)^2            N_L = p SNR*^2 / G^2  (p = 2, four-step)
  rule   G (1+beta) = sqrt(2)
  small-object closed form  G = k0 deps V / (n_m A_eff),  A_eff = lam^2/(pi NA^2)
  extended object           G^2 = 4 (int|E|^2)^2 / (int|E|)^2  with E = E_sc/E_R the
                            first-Born image field of the sphere (thin-object or 3D)

Run:  python figures_theory.py      (numpy, scipy, matplotlib)
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# ------------------------------------------------------------------ parameters
SNR = 10.0
n_m = 1.584          # mounting medium; bead 1.600
DEPS = 0.051         # n_b^2 - n_m^2
FL, LF, GOLD = "#6a1b9a", "#1565c0", "#B48A30"

mpl_style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
             "mathtext.fontset": "dejavusans", "font.size": 15, "axes.labelsize": 18,
             "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12, "axes.linewidth": 1.2}
matplotlib.rcParams.update(mpl_style)


class Optics:
    """Coherent imaging grid for one (NA, lambda)."""

    def __init__(self, NA=1.4, lam=0.5, px=0.010, L=3.0):
        self.NA, self.lam, self.px = NA, lam, px
        self.k0, self.k = 2 * np.pi / lam, n_m * 2 * np.pi / lam
        x = np.arange(-L, L, px)
        X, Y = np.meshgrid(x, x)
        self.r = np.hypot(X, Y)
        kx = 2 * np.pi * fftfreq(x.size, d=px)
        KX, KY = np.meshgrid(kx, kx)
        Kp = np.hypot(KX, KY)
        self.P = (Kp <= 2 * np.pi * NA / lam).astype(float)
        self.KZ = np.sqrt(np.clip(self.k**2 - Kp**2, 0, None))
        self.q = np.sqrt(Kp**2 + (self.KZ - self.k)**2)
        self.dA = px**2
        u = np.real(ifft2(self.P))
        self.A_eff = u.sum()**2 / (u**2).sum() * self.dA      # = 1/u(0) = lam^2/(pi NA^2) for a binary pupil
        h = u**2
        self.h = np.fft.fftshift(h / (h.sum() * self.dA))     # photon weight of the cell, int h dA = 1, centred
        self.Pfull = (Kp <= 0.999 * self.k).astype(float)      # all propagating waves (3D check)

    # ---- scattered field in the object plane, relative to the reference ----
    def E_thin(self, d, deps=DEPS):
        """projection (thin-object) approximation: E_sc/E_R = i phi, phi = k0 deps t/(2 n_m)."""
        a = d / 2
        t = np.where(self.r < a, 2 * np.sqrt(np.clip(a**2 - self.r**2, 0, None)), 0.0)
        return 1j * self.k0 * deps * t / (2 * n_m)

    def E_born3d(self, d, deps=DEPS):
        """first-Born field of the sphere in its mid-plane, all propagating waves (3D check):
        E(k_perp) = i k0^2/(2 k_z) F_deps(k_perp, k_z - k)."""
        a = d / 2
        qa = np.where(self.q == 0, 1e-12, self.q * a)
        F = deps * 4 * np.pi * (np.sin(qa) - qa * np.cos(qa)) / (qa / a)**3 * self.Pfull
        with np.errstate(divide="ignore", invalid="ignore"):
            pref = np.where(self.Pfull > 0, 1j * self.k0**2 / (2 * self.KZ), 0.0)
        return np.fft.fftshift(ifft2(pref * F)) / self.dA

    # ---- coupling coefficients ----
    def G_point(self, d, deps=DEPS):
        return self.k0 * deps * (np.pi * d**3 / 6) / (n_m * self.A_eff)

    def G_cell(self, E):
        """modulation coefficient of the resolution cell centred on the object:
        twice the photon-weighted mean scattered amplitude over the cell, G = 2|int h (E_sc/E_R) dA|."""
        return 2 * np.abs((self.h * E).sum() * self.dA)

    def G_curve(self, ds, model="3d", deps=DEPS):
        f = self.E_born3d if model == "3d" else self.E_thin
        return np.array([self.G_cell(f(d, deps)) for d in ds])


def N_F(beta):
    return SNR**2 * (1 + beta)**2


def N_L(G, p=2):
    return p * SNR**2 / G**2


def crossover(dgrid, Ggrid, beta):
    """diameter where N_L(G) = N_F(beta), log-interpolated (nan if none)."""
    lg = np.log(N_L(Ggrid)) - np.log(N_F(beta))
    i = np.where(np.diff(np.sign(lg)) != 0)[0]
    if i.size == 0:
        return np.nan
    i = i[0]
    return np.exp(np.interp(0, [lg[i + 1], lg[i]], [np.log(dgrid[i + 1]), np.log(dgrid[i])]))


def crossover_closed(opt, beta, deps=DEPS):
    V = np.sqrt(2) * n_m * opt.A_eff / (opt.k0 * deps * (1 + beta))
    return (6 * V / np.pi)**(1 / 3)


# ------------------------------------------------------------------ figures
def fig1b(opt, dgrid, G3, out="Fig_1b_budget.png"):
    """G3: coupling curve used for the plotted finite-sphere line (projection model)."""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    d_strong = crossover(dgrid, G3, 0) if False else dgrid[np.argmin(np.abs(G3 - 1.0))]
    d_valid = dgrid[np.argmin(np.abs(G3 - 0.6))]
    ax.axvspan(d_strong * 1e3, 3000, color="0.88", alpha=0.7, zorder=0)
    ax.text(2650, 1.6e2, "G > 1", fontsize=12, color="0.45", ha="left")
    ax.axhline(N_F(3), color=FL, lw=3, zorder=4, label="fluorescence, β = 3")
    ax.axhline(N_F(10), color=FL, lw=2, ls="--", zorder=4, label="fluorescence, β = 10")
    ax.plot(dgrid * 1e3, N_L(opt.G_point(dgrid)), color=LF, lw=1.6, ls=":", zorder=3,
            label="label-free, small-object limit (∝ d⁻⁶)")
    sel = dgrid <= d_valid
    ax.plot(dgrid[sel] * 1e3, N_L(G3[sel]), color=LF, lw=3, zorder=5, label="label-free, finite sphere, δε = 0.051 (projection model)")
    ax.plot(dgrid[~sel] * 1e3, N_L(G3[~sel]), color=LF, lw=1.8, ls="--", alpha=0.6, zorder=3)
    for beta, lab in [(3, "β = 3"), (10, "β = 10")]:
        dx = crossover(dgrid, G3, beta)
        if np.isfinite(dx):
            ax.plot(dx * 1e3, N_F(beta), "*", markerfacecolor=GOLD, markeredgecolor="#2F2A20", ms=17, zorder=8,
                    label="predicted crossover" if beta == 3 else None)
            ax.annotate(f"{dx*1e3:.0f} nm ({lab})", (dx * 1e3, N_F(beta)), textcoords="offset points",
                        xytext=(10, 8), fontsize=12.5)
    ax.text(140, 2.6e2, "fluorescence\ncheaper", color=FL, fontsize=15, weight="bold", va="center")
    ax.text(1250, 2.6e2, "label-free\ncheaper", color="#08519c", fontsize=15, weight="bold", ha="center", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(120, 3000); ax.set_ylim(1e2, 3e5)
    ax.set_xticks([200, 500, 1000, 2000]); ax.set_xticklabels(["200", "500", "1000", "2000"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("object diameter (nm)"); ax.set_ylabel("detected photons to reach SNR* = 10")
    ax.grid(True, which="both", alpha=0.18); ax.legend(loc="upper right", framealpha=0.95)
    fig.savefig(out, dpi=600, bbox_inches="tight"); plt.close(fig)


def fig1c(opt, G05, out="Fig_1c_map.png"):
    deps = np.geomspace(0.01, 0.3, 300)
    beta = np.linspace(0, 12, 300)
    D, B = np.meshgrid(deps, beta)
    G = G05 * D / DEPS                                   # Born: G linear in deps
    ratio = np.log10(N_F(B) / N_L(G))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.pcolormesh(D, B, ratio, cmap="RdBu_r", vmin=-2, vmax=2, shading="auto")
    b_line = np.sqrt(2) / (G05 * deps / DEPS) - 1
    ok = (b_line >= 0) & (b_line <= 12)
    ax.plot(deps[ok], b_line[ok], "k", lw=2.5)
    d_strong = DEPS / G05                                  # deps at which G = 1 for d = 0.5 um
    if d_strong < 0.3:
        ax.axvspan(d_strong, 0.3, facecolor="none", hatch="///", edgecolor="0.4", zorder=3)
        ax.text(d_strong * 1.05, 11.2, "G > 1", fontsize=11, color="0.3")
    ax.plot(DEPS, 3, "*", markerfacecolor=GOLD, markeredgecolor="#2F2A20", ms=16, zorder=6)
    ax.annotate("bead, β = 3", (DEPS, 3), textcoords="offset points", xytext=(-10, 10), fontsize=12, ha="right")
    ax.plot(DEPS, 10, "o", markerfacecolor="w", markeredgecolor="k", ms=8, zorder=6)
    ax.annotate("same bead, β = 10", (DEPS, 10), textcoords="offset points", xytext=(-10, 6), fontsize=12, ha="right")
    ax.set_xscale("log"); ax.set_xlim(0.01, 0.3); ax.set_ylim(0, 12)
    ax.set_xlabel(r"permittivity contrast $\delta\epsilon$"); ax.set_ylabel(r"background ratio $\beta = b/A$")
    ax.set_title("decision map, d = 0.5 μm", fontsize=14)
    cb = fig.colorbar(im, ax=ax); cb.set_label(r"$\log_{10}(N_\mathrm{F}/N_\mathrm{L})$")
    fig.savefig(out, dpi=600, bbox_inches="tight"); plt.close(fig)




# ------------------------------------------------------------------ main
if __name__ == "__main__":
    opt = Optics(NA=1.4, lam=0.5)
    print(f"A_eff = {opt.A_eff:.4f} um^2  (lam^2/(pi NA^2) = {opt.lam**2/(np.pi*opt.NA**2):.4f});"
          f"  A_coh = n_m A_eff = {n_m*opt.A_eff:.4f} um^2;  Airy FWHM = {0.51*opt.lam/opt.NA*1e3:.0f} nm")

    dgrid = np.geomspace(0.12, 3.0, 160)
    Gth = opt.G_curve(dgrid, "thin")        # projection model, plotted
    G3 = opt.G_curve(dgrid, "3d")           # 3D first-Born check

    print("\n d(um)  phi0    G_closed  G_proj  G_3D    N_L(proj)")
    for d in [0.10, 0.20, 0.35, 0.42, 0.50, 0.58, 0.66, 0.80, 1.00, 1.20, 1.50]:
        print(f" {d:4.2f}  {opt.k0*DEPS*d/(2*n_m):5.3f}   {opt.G_point(d):6.3f}   "
              f"{opt.G_cell(opt.E_thin(d)):5.3f}  {opt.G_cell(opt.E_born3d(d)):5.3f}"
              f"   {N_L(opt.G_cell(opt.E_thin(d))):8.0f}")
    sel = (dgrid >= 0.35) & (dgrid <= 0.66)
    for name, Garr in [("closed form", opt.G_point(dgrid)), ("projection", Gth), ("3D Born", G3)]:
        s = np.polyfit(np.log(dgrid[sel]), np.log(N_L(Garr[sel])), 1)[0]
        print(f" effective exponent of N_L vs d, 0.35-0.66 um, {name:12s}: {s:6.2f}")

    print("\n crossover (um)        beta=3    beta=10")
    print(f" closed form           {crossover_closed(opt, 3):6.3f}    {crossover_closed(opt, 10):6.3f}")
    print(f" projection (plotted)  {crossover(dgrid, Gth, 3):6.3f}    {crossover(dgrid, Gth, 10):6.3f}")
    print(f" 3D Born check         {crossover(dgrid, G3, 3):6.3f}    {crossover(dgrid, G3, 10):6.3f}")
    for gl in (0.6, 1.0):
        print(f" projection: G = {gl} at d = {dgrid[np.argmin(np.abs(Gth-gl))]*1e3:.0f} nm")

    G05 = opt.G_cell(opt.E_thin(0.5))
    print(f" G(0.5 um) = {G05:.3f} (3D Born {opt.G_cell(opt.E_born3d(0.5)):.3f});"
          f"  boundary at d = 0.5 um reached at beta = {np.sqrt(2)/G05-1:.1f} for deps = {DEPS}")
    for beta in (3, 10):
        print(f" N_F/N_L at d = 0.5 um, beta = {beta}: {N_F(beta)/N_L(G05):.2f}")
    print(f" validity in beta: G<=0.6 on the boundary needs beta >= {np.sqrt(2)/0.6-1:.2f};"
          f"  G<=1 needs beta >= {np.sqrt(2)-1:.2f}")

    fig1b(opt, dgrid, Gth)
    fig1c(opt, G05)
    print("\n saved Fig_1b_budget.png, Fig_1c_map.png")
