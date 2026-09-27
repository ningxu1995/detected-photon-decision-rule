"""
fig2_panels.py  --  Figure 2 of the revised manuscript.

    (a) photon cost per resolution cell versus fill factor rho, at beta = 10:
        fluorescence pays only for occupied cells and grows with rho, the
        interferometric reference floods the field and is nearly flat
    (b) regime map log10(N_F/N_L) over bead diameter and fill factor, with the
        crossover contour
    (c) axial k-space support: the single-view coherent transfer (pupil
        autocorrelation over the Ewald caps) leaves the missing cone open,
        whereas the incoherent confocal transfer fills it
    (d) axial conditioning kappa_axial of the confocal fluorescence response and
        the label-free missing-cone fraction 1 - NA/n, versus NA

The coupling G(d), the budgets N_F and N_L and the optical parameters are taken
from fig1_panels.py (or figures_theory.py) so that the two scripts cannot drift
apart. Conditioning factors follow the definition of the Supporting Information,
kappa = (I^-1)_nn I_nn >= 1, evaluated on the identifiable subspace.

Run:  python fig2_panels.py        (numpy, scipy, matplotlib)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Bbox

try:
    from fig1_panels import Optics, N_F, N_L, DEPS, n_m, SNR, FL, LF
except ImportError:                                   # same names, fuller script
    from figures_theory import Optics, N_F, N_L, DEPS, n_m, SNR, FL, LF

N_IMM = 1.515          # immersion oil, used for the transfer functions
DIAMS = (0.6, 0.7, 0.9)
BETA_MAP = 10.0

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 13, "axes.labelsize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11, "axes.linewidth": 1.1})


# ------------------------------------------------------------ lateral conditioning
def kappa_lat(rho, opt, npix=192, half=1.6):
    """variance inflation of one emitter's brightness from overlap with its
    neighbours on a square lattice of fill factor rho, from the full Fisher matrix."""
    if rho <= 0:
        return 1.0
    cell = np.sqrt(opt.A_eff)                      # cell side
    pitch = cell / np.sqrt(rho)                    # emitter spacing
    px = 2 * half / npix
    x = (np.arange(npix) - npix / 2 + 0.5) * px
    X, Y = np.meshgrid(x, x)
    sig = 0.21 * opt.lam / opt.NA                  # Gaussian equivalent of the Airy core
    pos = [(i * pitch, j * pitch) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    H = np.array([np.exp(-((X - a) ** 2 + (Y - b) ** 2) / (2 * sig ** 2)) for a, b in pos])
    H /= H[0].sum() * px ** 2                      # equal brightness, unit integral
    mu = H.sum(axis=0) + 1e-12
    I = np.einsum("ixy,jxy->ij", H / mu, H) * px ** 2
    c = pos.index((0.0, 0.0))
    return float(np.linalg.inv(I)[c, c] * I[c, c])


def cost_fluor_field(rho, beta, opt):
    """detected photons per resolution cell of the field, fluorescence."""
    return rho * N_F(beta) * kappa_lat(rho, opt)


def cost_labelfree_field(rho, G, opt):
    """detected photons per resolution cell of the field, interferometric."""
    return N_L(G) * np.ones_like(np.asarray(rho, dtype=float))


def rho_crossover(G, beta, opt):
    r = np.geomspace(0.01, 1.0, 400)
    f = np.array([cost_fluor_field(x, beta, opt) for x in r])
    lg = np.log(f) - np.log(N_L(G))
    i = np.where(np.diff(np.sign(lg)) != 0)[0]
    if i.size == 0:
        return np.nan
    i = i[0]
    return float(np.exp(np.interp(0, [lg[i], lg[i + 1]], [np.log(r[i]), np.log(r[i + 1])])))


# ------------------------------------------------------------ axial transfer
def _grid(NA, lam, n, N, need_box):
    """transverse sampling fine enough for the transfer support, box long enough
    for the axial sections."""
    k0 = 2 * np.pi / lam
    dx = min(0.9 * np.pi / (k0 * (2 * NA + 0.9)), lam / 7)
    while N * dx < need_box and dx < 0.30:
        dx *= 1.15
    return dx


def axial_resolution(NA, lam=0.5, n=N_IMM):
    """confocal axial resolution, lambda / [2 n (1 - cos alpha)]."""
    return lam / (2 * n * (1 - np.cos(np.arcsin(NA / n))))


def transfer_3d(NA, lam=0.5, n=N_IMM, N=192, need_box=0.0, slices=True):
    """single-view coherent (= widefield intensity) and confocal intensity transfer.
    Returns the (K_perp, K_z) slices in dB-ready magnitude, the axes in units of k0,
    and the on-axis confocal profile O(0, k_z) with its k_z axis."""
    k0 = 2 * np.pi / lam
    k = n * k0
    dx = _grid(NA, lam, n, N, need_box)
    kax = 2 * np.pi * np.fft.fftfreq(N, d=dx).astype(np.float32)
    KX, KY, KZ = np.meshgrid(kax, kax, kax, indexing="ij")
    Kp = np.hypot(KX, KY)
    dk = abs(kax[1] - kax[0])
    cap = ((np.abs(np.sqrt(Kp ** 2 + KZ ** 2) - k) < dk) & (Kp <= NA * k0) & (KZ > 0)).astype(np.float32)
    del KX, KY, KZ, Kp
    A = np.fft.ifftn(cap)
    del cap
    I2 = (np.abs(A) ** 2).astype(np.float32)
    I4 = (I2 ** 2).astype(np.float32)
    del A
    O_cf = np.abs(np.fft.fftn(I4))
    prof = np.fft.fftshift(O_cf[0, 0, :])
    kz = np.fft.fftshift(kax)
    if not slices:
        return None, None, None, prof / prof.max(), kz
    O_wf = np.abs(np.fft.fftn(I2))
    mid = N // 2
    sl_wf = np.fft.fftshift(O_wf)[:, mid, :]
    sl_cf = np.fft.fftshift(O_cf)[:, mid, :]
    return sl_wf, sl_cf, np.fft.fftshift(kax) / k0, prof / prof.max(), kz


def kappa_axial(NA, lam=0.5, n=N_IMM, nsec=7):
    """variance inflation of one axial section amplitude of the confocal
    fluorescence response, kappa = (I^-1)_zz I_zz, sections spaced by the
    confocal axial resolution."""
    dz = axial_resolution(NA, lam, n)
    _, _, _, prof, kz = transfer_3d(NA, lam, n, need_box=8 * dz, slices=False)
    z = (np.arange(nsec) - nsec // 2) * dz
    W = prof.astype(float) ** 2
    I = np.array([[np.sum(W * np.cos(kz * (zi - zj))) for zj in z] for zi in z])
    c = nsec // 2
    return float(np.linalg.inv(I)[c, c] * I[c, c]), dz


# ------------------------------------------------------------ panels
def panel_a(ax, opt):
    rho = np.geomspace(0.02, 1.0, 60)
    f = np.array([cost_fluor_field(r, BETA_MAP, opt) for r in rho])
    ax.plot(rho, f, color=FL, lw=3, label=f"fluorescence, β = {BETA_MAP:.0f}")
    for d, col in zip(DIAMS, ["#1b7837", "#1565c0", "#d95f02"]):
        G = opt.G_cell(opt.E_thin(d))
        ax.plot(rho, cost_labelfree_field(rho, G, opt), color=col, lw=2,
                label=f"label-free {d*1e3:.0f} nm")
        rx = rho_crossover(G, BETA_MAP, opt)
        if np.isfinite(rx):
            ax.plot([rx], [N_L(G)], "o", color=col, mec="k", mew=0.8, ms=7, zorder=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("fill factor  ρ"); ax.set_ylabel("photons per cell to SNR* = 10")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
    ax.set_title("(a)", loc="left", fontsize=14)


def panel_b(ax, opt):
    d = np.geomspace(0.2, 1.5, 160)
    rho = np.geomspace(0.02, 1.0, 160)
    D, R = np.meshgrid(d, rho, indexing="ij")
    kl = np.interp(rho, rho, [kappa_lat(r, opt) for r in rho])
    G = np.array([opt.G_cell(opt.E_thin(x)) for x in d])
    NFv = (R * N_F(BETA_MAP)) * kl[None, :]
    NLv = N_L(G)[:, None] * np.ones_like(R)
    Z = np.log10(NFv / NLv)
    im = ax.pcolormesh(R, D * 1e3, np.clip(Z, -2, 2), cmap="RdBu_r", vmin=-2, vmax=2,
                       shading="auto")
    ax.contour(R, D * 1e3, Z, levels=[0], colors="k", linewidths=2)
    ax.set_xscale("log")
    ax.set_xlabel("fill factor  ρ"); ax.set_ylabel("bead diameter (nm)")
    ax.set_title("(b)", loc="left", fontsize=14)
    ax.text(0.03, 1350, "label-free\ncheaper", color="#08519c", fontsize=11, weight="bold",
            va="center")
    ax.text(0.03, 300, "fluorescence cheaper", color=FL, fontsize=11, weight="bold", va="center")
    return im


def panel_c(ax, NA=1.4, lam=0.5, n=N_IMM):
    O_wf, O_cf, kx, _, _ = transfer_3d(NA, lam, n)
    kz = kx
    def db(O):
        O = O / O.max()
        return 20 * np.log10(np.maximum(O, 1e-4))
    Kx, Kz = np.meshgrid(kx, kz, indexing="ij")
    left = np.where(Kx <= 0, db(O_wf), np.nan)
    right = np.where(Kx > 0, db(O_cf), np.nan)
    for Z in (left, right):
        ax.pcolormesh(Kx, Kz, Z, cmap="Blues_r", vmin=-30, vmax=0, shading="auto")
    gam = np.pi / 2 - np.arcsin(NA / n)          # missing-cone half-angle from K_z
    zz = np.linspace(-1.5, 1.5, 50)
    for s_ in (-1, 1):
        ax.plot(s_ * np.abs(zz) * np.tan(gam) * -1, zz, color="w", lw=1.1, ls="--", zorder=5)
    ax.set_xlim(-3, 3); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("$K_\\perp$  ($k_0$)"); ax.set_ylabel("$K_z$  ($k_0$)")
    ax.text(-2.85, -1.38, "coherent\nmissing cone", color="w", fontsize=10, va="bottom")
    ax.text(2.85, -1.38, "confocal\nfilled", color="0.25", fontsize=10, va="bottom", ha="right")
    ax.text(0, 1.62, f"NA = {NA}, λ = {lam*1e3:.0f} nm, n = {n}", fontsize=9.5, color="0.3", ha="center")
    ax.set_title("(c)", loc="left", fontsize=14)


def panel_d(ax, NAs=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.45), n=N_IMM):
    ka, dz = [], []
    for NA in NAs:
        k, d = kappa_axial(NA); ka.append(k); dz.append(d)
    ka = np.array(ka)
    ax.plot(NAs, ka, "o-", color="#0f4c5c", lw=2, ms=6, label="fluorescence $\\kappa_{\\rm axial}$")
    ax.set_ylim(0, 4)
    ax.set_xlabel("NA"); ax.set_ylabel("fluorescence  $\\kappa_{\\rm axial}$", color="#0f4c5c")
    ax2 = ax.twinx()
    fmc = 100 * (1 - np.array(NAs) / n)
    ax2.plot(NAs, fmc, "s--", color="#c1272d", lw=2, ms=5, label="missing-cone fraction")
    ax2.set_ylabel("label-free missing cone (%)", color="#c1272d")
    ax2.set_ylim(0, 70)
    ax.grid(True, which="both", alpha=0.18)
    ax.set_title("(d)", loc="left", fontsize=14)
    return ka, dz, ax2


# ------------------------------------------------------------ main
if __name__ == "__main__":
    opt = Optics(NA=1.4, lam=0.5)
    print(f"A_eff = {opt.A_eff:.4f} um^2, cell side = {np.sqrt(opt.A_eff)*1e3:.0f} nm")
    for r in (0.1, 0.25, 0.5, 1.0):
        print(f" kappa_lat(rho = {r:4.2f}) = {kappa_lat(r, opt):.2f}")
    print()
    for d in DIAMS:
        G = opt.G_cell(opt.E_thin(d))
        print(f" d = {d*1e3:3.0f} nm: G = {G:.3f}, N_L = {N_L(G):7.0f}, "
              f"rho_x(beta = {BETA_MAP:.0f}) = {rho_crossover(G, BETA_MAP, opt):.2f}")

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.5))
    panel_a(axs[0, 0], opt)
    im = panel_b(axs[0, 1], opt)
    cb = fig.colorbar(im, ax=axs[0, 1], label="$\\log_{10}(N_{\\rm F}/N_{\\rm L})$")
    panel_c(axs[1, 0])
    ka, dz, ax_d2 = panel_d(axs[1, 1])   # axial resolutions are printed, not drawn
    print("\n NA      kappa_axial   dz (nm)")
    for NA, k, d in zip((0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.45), ka, dz):
        print(f" {NA:4.2f}      {k:6.2f}      {d*1e3:5.0f}")
    fig.tight_layout()
    fig.savefig("Figure_2.png", dpi=600, bbox_inches="tight")
    groups = [[axs[0, 0]], [axs[0, 1], cb.ax], [axs[1, 0]], [axs[1, 1], ax_d2]]
    names = ["Fig_2a_density.png", "Fig_2b_regime.png", "Fig_2c_kspace.png", "Fig_2d_axial.png"]
    rend = fig.canvas.get_renderer()
    for group, name in zip(groups, names):
        bb = Bbox.union([a.get_tightbbox(rend) for a in group]).transformed(
            fig.dpi_scale_trans.inverted())
        fig.savefig(name, dpi=600, bbox_inches=bb.expanded(1.06, 1.08))
    print("\n saved Figure_2.png and the four panels separately")
