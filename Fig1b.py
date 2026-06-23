"""
Fig 1(b)  --  unified photon-budget CROSSOVER, with Monte-Carlo / maximum-
likelihood validation that the Cramer-Rao bound is achievable.

Revision:
  * Black stars are kept and explicitly labelled as the theoretical crossover.
  * The confusing black-dot legend entry for Monte-Carlo ML is removed.
  * The grey arrow between the two crossover points is removed.
  * In Fig. S4, the “±0.3%” text is removed.
  * All font sizes are increased by 1 point relative to the previous version.
  * Output is JPG at 600 dpi.
"""

import numpy as np
rng = np.random.default_rng(3)

# ---------------- physics ----------------
lam = 0.5
NA = 1.4
n_med = 1.515
deps = 0.05
beta = 3.0
SNRst = 10.0

k0 = 2 * np.pi / lam
A_coh = 0.42 * (n_med / NA) * lam**2

def c_int(d_um):
    V = (4 * np.pi / 3) * (d_um / 2)**3
    return k0 * V / A_coh

def G(d_um):
    return c_int(d_um) * deps

# ---------------- fluorescence ----------------
def fluor_relprec(Ntot, beta_):
    return (1.0 + beta_) / np.sqrt(Ntot)

def fluor_N(beta_):
    return (1.0 + beta_)**2 * SNRst**2

def fluor_mc_relprec(beta_, Ntot, Mt):
    sig = Ntot / (1.0 + beta_)
    bg = beta_ * Ntot / (1.0 + beta_)
    T = rng.poisson(sig + bg, Mt).astype(float)
    shat = T - bg
    return shat.std() / sig

# ---------------- label-free ----------------
chis = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
cos_c = np.cos(chis)

def lab_mu(Ni, ci):
    return Ni * (1 + ci * cos_c * deps + (ci**2 / 4) * deps**2)

def lab_crlb_relprec(Ni, d):
    ci = c_int(d)
    mu = lab_mu(Ni, ci)
    dmu = Ni * (ci * cos_c + (ci**2 / 2) * deps)
    I = np.sum(dmu * dmu / mu)
    return np.sqrt(1.0 / I) / deps, mu.sum()

def lab_N_for_SNR(d):
    Ni0 = 1.0
    rp0, _ = lab_crlb_relprec(Ni0, d)
    Ni = Ni0 * (rp0 * SNRst)**2
    rp, N = lab_crlb_relprec(Ni, d)
    return Ni, N, rp

def lab_mle(y, Ni, d, x_init):
    ci = c_int(d)
    x = x_init

    for _ in range(60):
        mu = np.clip(
            Ni * (1 + ci * cos_c * x + (ci**2 / 4) * x**2),
            1e-9,
            None
        )

        dmu = Ni * (ci * cos_c + (ci**2 / 2) * x)
        d2mu = Ni * (ci**2 / 2)

        g = np.sum((1 - y / mu) * dmu)
        H = np.sum((y / mu**2) * dmu * dmu + (1 - y / mu) * d2mu)

        if H <= 0:
            H = abs(H) + 1e-9

        step_ = g / H
        x -= step_

        if abs(step_) < 1e-10:
            break

    return x

# ---------------- self-check ----------------
N_F = fluor_N(3.0)
rp = fluor_relprec(N_F, 3.0)

print(
    f"N_F(beta=3)={N_F:.0f}, "
    f"N_F(beta=10)={fluor_N(10.0):.0f}, "
    f"fluor CRLB relprec at N_F={rp:.4f}"
)

d0 = 0.5
Ni_star, N_L, rpl = lab_N_for_SNR(d0)

print(
    f"N_L(0.5um)={N_L:.0f}, "
    f"cross-check 2 SNR^2/G^2={2 * SNRst**2 / G(d0)**2:.0f}, "
    f"label CRLB relprec={rpl:.4f}"
)

rp_mc = fluor_mc_relprec(3.0, N_F, 20000)
print(f"fluor MC relprec={rp_mc:.4f}  (r = {rp / rp_mc:.3f})")

# ======================= BUILD FIG 1(b) =======================
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

NF_thin = fluor_N(3.0)
NF_thick = fluor_N(10.0)

dgrid = np.geomspace(0.10, 1.5, 2000)

def NL_curve(dd, de):
    ci = k0 * (4 * np.pi / 3) * (dd / 2)**3 / A_coh
    return 2 * SNRst**2 / (ci * de)**2

NL_05 = NL_curve(dgrid, 0.05)

def G_of_d(dd, de=0.05):
    return k0 * (4 * np.pi / 3) * (dd / 2)**3 / A_coh * de

d_valid = dgrid[np.argmin(np.abs(np.array([G_of_d(d) for d in dgrid]) - 0.6))]
d_strong = dgrid[np.argmin(np.abs(np.array([G_of_d(d) for d in dgrid]) - 1.0))]
valid = dgrid <= d_valid

dstar_thin = dgrid[np.argmin(np.abs(NL_05 - NF_thin))]
dstar_thick = dgrid[np.argmin(np.abs(NL_05 - NF_thick))]

# ---- MC-ML validation points ----
def mc_fluor(d, Mt=4000):
    rp_mc = fluor_mc_relprec(beta, NF_thin, Mt)
    Nm = NF_thin * (rp_mc * SNRst)**2
    return Nm, Nm * 2 / np.sqrt(2 * Mt)

def mc_label(d, Mt=2500):
    Ni_star, Nstar, rp = lab_N_for_SNR(d)

    e = np.array([
        lab_mle(
            rng.poisson(lab_mu(Ni_star, c_int(d))),
            Ni_star,
            d,
            deps
        )
        for _ in range(Mt)
    ])

    rp_mc = e.std() / deps

    return (
        Nstar * (rp_mc * SNRst)**2,
        Nstar * (rp_mc * SNRst)**2 * 2 / np.sqrt(2 * Mt),
        (e.mean() - deps) / deps
    )

df_pts = [0.15, 0.3, 0.45, 0.58]
dl_pts = [0.35, 0.45, 0.52, 0.58]

Nf_mc = [mc_fluor(d) for d in df_pts]
Nl_mc = [mc_label(d) for d in dl_pts]

# ---- label-free ML bias vs SNR* ----
snr_list = np.array([4, 6, 8, 10, 14, 20, 30, 45])
Mt_bias = 20000

bias_list = []
bias_se = []

for snr in snr_list:
    Ni0 = 1.0
    rp0, _ = lab_crlb_relprec(Ni0, 0.5)
    Ni = Ni0 * (rp0 * snr)**2

    e = np.array([
        lab_mle(
            rng.poisson(lab_mu(Ni, c_int(0.5))),
            Ni,
            0.5,
            deps
        )
        for _ in range(Mt_bias)
    ])

    bias_list.append((e.mean() - deps) / deps * 100)
    bias_se.append(e.std() / np.sqrt(Mt_bias) / deps * 100)

bias_list = np.array(bias_list)
bias_se = np.array(bias_se)

# ---------------- figure style ----------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 18,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "axes.linewidth": 1.2,
})

FL = "#6a1b9a"
LF = "#1565c0"

fig, ax = plt.subplots(figsize=(8.8, 6.6))

ax.axvspan(
    d_strong * 1000,
    1500,
    color="0.88",
    alpha=0.7,
    zorder=0
)

ax.text(
    1040,
    1.45e2,
    "strong scattering\n(G > 1)",
    fontsize=15.5,
    color="0.45",
    ha="center",
    va="center"
)

ax.axhline(
    NF_thin,
    color=FL,
    lw=3.0,
    zorder=4,
    label="fluorescence, β = 3"
)

ax.axhline(
    NF_thick,
    color=FL,
    lw=2.0,
    ls="--",
    zorder=4,
    label="fluorescence, β = 10"
)

ax.plot(
    dgrid[valid] * 1000,
    NL_05[valid],
    color=LF,
    lw=3.0,
    zorder=5,
    label="label-free, δε = 0.05"
)

ax.plot(
    dgrid[~valid] * 1000,
    NL_05[~valid],
    color=LF,
    lw=1.8,
    ls="--",
    alpha=0.6,
    zorder=3
)

# Black stars = theoretical crossover points.
ax.plot(
    dstar_thin * 1000,
    NF_thin,
    "*",
    #color="k",
    markerfacecolor= '#B48A30', markeredgecolor= '#2F2A20', markeredgewidth=0.9,
    ms=17,
    zorder=8,
    label="predicted crossover"
)

ax.plot(
    dstar_thick * 1000,
    NF_thick,
    "*",
    #color="k",
    markerfacecolor='#B48A30', markeredgecolor='#2F2A20', markeredgewidth=0.9,
    ms=17,
    zorder=8
)

ax.annotate(
    "354 nm ($\\beta$=10)",
    (dstar_thick * 1000, NF_thick),
    textcoords="offset points",
    xytext=(10, 6),
    fontsize=13.5,
    ha="left",
    va="bottom",
    color="k"
)

ax.annotate(
    "497 nm ($\\beta$=3)",
    (dstar_thin * 985, NF_thin * 1.65),
    textcoords="offset points",
    xytext=(11, -6),
    fontsize=13.5,
    ha="left",
    va="top",
    color="k"
)

# Grey arrow removed.
# Removed the confusing black-dot proxy legend entry for Monte-Carlo ML.

ax.text(
    122,
    2.6e2,
    "fluorescence\ncheaper",
    color=FL,
    fontsize=17,
    weight="bold",
    va="center"
)

ax.text(
    520,
    2.6e2,
    "label-free\ncheaper",
    color="#08519c",
    fontsize=17,
    weight="bold",
    ha="center",
    va="center"
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(110, 1500)
ax.set_ylim(1e2, 1.6e5)

ax.set_xticks([200, 500, 1000])
ax.set_xticklabels(["200", "500", "1000"])

ax.set_yticks([1e2, 1e3, 1e4, 1e5])

ax.set_xlabel("object diameter (nm)")
ax.set_ylabel("photons to reach SNR* = 10")

ax.grid(True, which="both", alpha=0.18)
ax.legend(loc="upper right", framealpha=0.95)

fig.savefig("Figure 1b.png", dpi=600, bbox_inches="tight")

print(
    f"saved Figure 1b.png; "
    f"d*(b=3)={dstar_thin * 1000:.0f}nm, "
    f"d*(b=10)={dstar_thick * 1000:.0f}nm"
)

# ======================= FIG. S4 =======================
fig2, ax2 = plt.subplots(figsize=(6.4, 4.6))

ax2.axhline(0, color="k", lw=0.8)
ax2.axhline(0.3, color="0.55", lw=1.0, ls=":")
ax2.axhline(-0.3, color="0.55", lw=1.0, ls=":")

# Removed “±0.3%” text.

ax2.errorbar(
    snr_list,
    bias_list,
    yerr=bias_se,
    fmt="o-",
    color=LF,
    ms=6,
    lw=1.8,
    capsize=3
)

ax2.axvline(
    SNRst,
    color="r",
    ls="--",
    lw=1.2
)

ax2.text(
    SNRst * 1.07,
    0.37,
    "SNR$^*$ = 10",
    color="r",
    fontsize=14,
    ha="left",
    va="center"
)

ax2.set_xscale("log")

ax2.set_xticks([5, 10, 20, 40])
ax2.set_xticklabels(["5", "10", "20", "40"])
ax2.xaxis.set_minor_formatter(NullFormatter())

ax2.set_xlim(3.4, 52)
ax2.set_ylim(-0.45, 0.45)
ax2.set_yticks([-0.3, 0, 0.3])

ax2.set_xlabel("target SNR$^*$")
ax2.set_ylabel("relative bias of $\\delta\\epsilon$ (%)")

ax2.grid(True, which="both", alpha=0.2)

fig2.savefig("Figure S4.png", dpi=600, bbox_inches="tight")

print(
    "saved Figure S4.png; bias at SNR*: "
    + ", ".join(
        f"{s}: {b:+.3f}±{se:.3f}%"
        for s, b, se in zip(snr_list, bias_list, bias_se)
    )
)