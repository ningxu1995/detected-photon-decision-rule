"""
Fig. S1 (Supplemental Material) -- experimental accessibility of the crossover.

Where does the coherent-incoherent crossover land as a function of numerical
aperture and index contrast, and is it reachable with standard microspheres?

This uses the same single-object closed forms as the main text,
    N_fluor = SNR*^2 (1+beta)^2            (flat in size; Eq. S7)
    N_label = 2 SNR*^2 / G^2,  G = k_j V / A_b * delta_eps   (Eq. S12)
so the crossover satisfies G = sqrt(2)/(1+beta) and the crossover diameter
scales as delta_eps^{-1/3} (Eq. S17). No fit, no free parameters.

Style: PRL-compliant sans-serif (Arial/Helvetica, DejaVu Sans fallback),
enlarged fonts, sparse ticks. Output PNG is written next to this script.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----- PRL-style typography: sans-serif, larger throughout -----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",   # keep any math in a matching sans face
    "font.size": 15,
    "axes.labelsize": 19,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

# ----- physics (identical to Eq. S7/S12/S17) -----
SNRstar = 10.0
n_m     = 1.515

def A_beam(NA, lam):
    return 0.42 * (n_m / NA) * lam**2          # coherence area of one cell (um^2)

def d_crossover(beta, deps, NA, lam):
    """Crossover bead DIAMETER (nm). N_fluor = N_label  ->  G = sqrt(2)/(1+beta)."""
    Gx   = np.sqrt(2.0) / (1.0 + beta)
    kj   = 2 * np.pi / lam
    coef = kj * (4 * np.pi / 3.0) / A_beam(NA, lam) * deps   # G = coef * a^3
    a    = (Gx / coef) ** (1.0 / 3.0)
    return 2.0 * a * 1000.0

# ----- figure -----
fig, ax = plt.subplots(figsize=(8.2, 6.2))

NAs  = np.linspace(0.20, 1.45, 240)
beta = 3
lam0 = 0.5

# permittivity-contrast family at lambda = 500 nm
deps_list = [0.02, 0.05, 0.10, 0.30]
colors    = ['#0a8f6b', '#1565c0', '#7b3fa0', '#d4540a']   # teal, blue, purple, orange
for de, c in zip(deps_list, colors):
    dc = np.array([d_crossover(beta, de, na, lam0) for na in NAs])
    ax.plot(NAs, dc, color=c, lw=2.7, label=f'\u03b4\u03b5 = {de:g}')

# one shorter-wavelength case (lambda = 400 nm) for delta_eps = 0.05, dashed
dc400 = np.array([d_crossover(beta, 0.05, na, 0.4) for na in NAs])
ax.plot(NAs, dc400, color='#1565c0', lw=1.9, ls='--',
        label='\u03b4\u03b5 = 0.05,  \u03bb = 400 nm')

# available off-the-shelf microsphere range (100 nm - 1 um), label moved lower-left
ax.axhspan(100, 1000, color='gold', alpha=0.16, zorder=0)
ax.text(0.24, 125, 'available bead range (100 nm \u2013 1 \u00b5m)',
        ha='left', va='bottom', fontsize=15, color='#8a6d00')

# ----- axes, sparse ticks -----
ax.set_xlabel('NA')
ax.set_ylabel('crossover diameter (nm)')
ax.set_xlim(0.20, 1.45)
ax.set_ylim(0, 1600)
ax.set_xticks([0.2, 0.6, 1.0, 1.4])          # sparse
ax.set_yticks([0, 400, 800, 1200, 1600])     # sparse
ax.grid(True, alpha=0.25, linewidth=1.0)

leg = ax.legend(loc='upper right', framealpha=0.92, title='\u03b2 = 3,  \u03bb = 500 nm unless noted')
leg.get_title().set_fontsize(14)

fig.tight_layout()
fig.savefig('figS1_feasibility.png', dpi=600, bbox_inches='tight')
print('saved figS1_feasibility.png')
