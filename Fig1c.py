#!/usr/bin/env python3
"""
fig1c_decision_map.py
---------------------
Figure 1(c) of "Coherent-Incoherent Crossover in the Information Limits of
Optical Microscopy".

The decision plane of the budget rule. Axes are the two object properties that
set the two photon costs: the permittivity contrast delta-epsilon (the
label-free parameter B, x-axis) and the background ratio beta = b/A (y-axis),
with the fluorescence brightness A rising downward. Color is
log10(N_F / N_L) at fixed object diameter d = 0.5 um. Label-free detection is
cheaper above the boundary of Eq. (rule); fluorescence is cheaper below it.

Costs (single resolution cell, SNR* cancels in the ratio):
    N_F = (SNR*)^2 (1 + beta)^2
    N_L = 2 (SNR*)^2 / G^2 ,   G = (k_j V / A_b) delta-epsilon
    N_F / N_L = (1 + beta)^2 G^2 / 2
Boundary:  (1 + beta) G = sqrt(2).

The geometric constant k_j V / A_b is computed from the stated optical
parameters (NA = 1.4, lambda = 500 nm, n = 1.515); it reproduces the
Supplemental-Material normalization G = 1 at d0 = 702 nm for delta-epsilon
= 0.05, and places the boundary at beta ~ 3 for the index-matched bead
(delta-epsilon = 0.05) at d = 0.5 um.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch

# ----------------------------------------------------------------------
# Style: Arial-compatible sans-serif, thin spines, publication defaults
# ----------------------------------------------------------------------
for _p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
           "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
           "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass

# 全局字体大小从 7.5 调大到 9.0
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,   # editable text in the PDF
    "ps.fonttype": 42,
})

# ----------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------
lam = 0.5            # um
NA  = 1.4
n   = 1.515
d0  = 0.5            # um, fixed diameter of this plane

kj  = 2 * np.pi / lam
Ab  = 0.42 * (n / NA) * lam**2          # coherence area of one resolution cell
V   = (4.0 / 3.0) * np.pi * (d0 / 2)**3
kd  = kj * V / Ab                        # = (k_j V / A_b) at d = 0.5 um (~7.24)

G_of_de   = lambda de: kd * de           # interference coefficient at fixed d
de_Gunity = 1.0 / kd                     # delta-epsilon at which G = 1 (~0.138)

# ----------------------------------------------------------------------
# Grid and field
# ----------------------------------------------------------------------
de_min, de_max = 0.01, 0.18
b_min,  b_max  = 0.0, 12.0
NDE, NB = 600, 600

de = np.linspace(de_min, de_max, NDE)
bb = np.linspace(b_min,  b_max,  NB)
DE, BB = np.meshgrid(de, bb)
G = G_of_de(DE)
logratio = np.log10((1 + BB)**2 * G**2 / 2.0)     # log10(N_F / N_L)

# Mask the strong-scattering region G > 1, where the linear (Born) cost fails
born = DE <= de_Gunity
logratio_born = np.ma.masked_where(~born, logratio)

# Boundary  (1 + beta) G = sqrt(2)  ->  beta = sqrt(2)/(kd*de) - 1
beta_bd = np.sqrt(2) / (kd * de) - 1.0
in_frame = (beta_bd >= b_min) & (beta_bd <= b_max)
solid = in_frame & (de <= de_Gunity)              # boundary inside Born regime
dash  = in_frame & (de >  de_Gunity)              # boundary extrapolated, G > 1

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)

vlim = 2.0
cmap = plt.get_cmap("coolwarm").copy()            # blue = fluor, red = label-free
pcm = ax.pcolormesh(DE, BB, logratio_born, cmap=cmap,
                    vmin=-vlim, vmax=vlim, shading="auto", rasterized=True)

# Strong-scattering band (G > 1): gray hatch
ax.axvspan(de_Gunity, de_max, color="0.78", alpha=0.55, lw=0, zorder=2)
ax.fill_betweenx([b_min, b_max], de_Gunity, de_max, facecolor="none",
                 edgecolor="0.45", hatch="////", lw=0.0, zorder=2.1)
ax.axvline(de_Gunity, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=2.2)

# fontsize 6.6 -> 8.1
ax.text(de_Gunity + 0.5 * (de_max - de_Gunity), 11.1, r"$G>1$",
        ha="center", va="center", fontsize=8.1, color="0.15", zorder=3)
# fontsize 5.8 -> 7.3
ax.text(de_Gunity + 0.5 * (de_max - de_Gunity), 9.9, "strong\nscattering",
        ha="center", va="center", fontsize=7.3, color="0.20",
        linespacing=0.95, zorder=3)

# Decision boundary
ax.plot(de[solid], beta_bd[solid], color="k", lw=1.5, zorder=4,
        solid_capstyle="round")
ax.plot(de[dash], beta_bd[dash], color="k", lw=1.5, ls=(0, (3, 2)), zorder=4)

# Region labels
# fontsize 6.9 -> 8.4
ax.text(0.030, 0.85, "fluorescence\ncheaper", ha="left", va="center",
        fontsize=8.4, color="#11335c", linespacing=0.98, zorder=5)
# fontsize 6.9 -> 8.4
ax.text(0.130, 10.5, "label-free\ncheaper", ha="right", va="center",
        fontsize=8.4, color="#FFFFFF", linespacing=0.98, zorder=5)

# Experimental bead
de_exp = 0.05
ax.add_patch(FancyArrowPatch((de_exp, 3.0), (de_exp, 10.0),
             arrowstyle="-|>", mutation_scale=7, lw=0.9,
             color="0.15", zorder=5.5,
             shrinkA=3.2, shrinkB=3.2))
ax.scatter([de_exp], [3.0], s=46, marker="*", facecolor="#ffd23f",
           edgecolor="k", linewidths=0.7, zorder=6)
ax.scatter([de_exp], [10.0], s=24, marker="o", facecolor="w",
           edgecolor="k", linewidths=0.9, zorder=6)
# fontsize 6.2 -> 7.7
ax.annotate(r"bead, $\beta=3$", (de_exp, 3.0), (de_exp + 0.007, 3.2),
            fontsize=7.7, color="k", zorder=6,
            ha="left", va="center")
# fontsize 6.2 -> 7.7
ax.annotate(r"$\beta=10$", (de_exp, 10.0), (de_exp + 0.007, 10.0),
            fontsize=7.7, color="k", zorder=6, ha="left", va="center")

# Axes cosmetics
ax.set_xlim(de_min, de_max)
ax.set_ylim(b_min, b_max)
# fontsize 7.5 -> 9.0
ax.set_xlabel(r"permittivity contrast  $\delta\epsilon\ \ (B)$", fontsize=9.0)
ax.set_ylabel(r"background ratio  $\beta=b/A$", fontsize=9.0)
ax.set_xticks([0.01, 0.05, 0.10, 0.15])
ax.set_yticks([0, 3, 6, 9, 12])
# labelsize 7.0 -> 8.5
ax.tick_params(labelsize=8.5)

# "brighter A" direction
ax.annotate("", xy=(0.0155, 1.0), xytext=(0.0155, 5.0),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.30"),
            zorder=5)
# fontsize 6.0 -> 7.5
ax.text(0.0182, 3.0, "brighter $A$", rotation=90, va="center", ha="left",
        fontsize=7.5, color="0.30", zorder=5)

# Colorbar
cb = fig.colorbar(pcm, ax=ax, fraction=0.052, pad=0.03,
                  ticks=[-2, -1, 0, 1, 2])
# fontsize 7.0 -> 8.5
cb.set_label(r"$\log_{10}(N_{\mathrm{F}}/N_{\mathrm{L}})$", fontsize=8.5)
# labelsize 6.6 -> 8.1
cb.ax.tick_params(labelsize=8.1, width=0.7, size=2.4)
cb.outline.set_linewidth(0.7)

# Fixed-diameter note
# fontsize 6.2 -> 7.7
ax.text(0.985, 0.075, r"$d=0.5\,\mu$m", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7.1, color="0.15",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.7", lw=0.5,
                  alpha=0.85), zorder=7)

for s in ax.spines.values():
    s.set_zorder(8)

fig.savefig("fig1c_decision_map.pdf", dpi=600)
fig.savefig("fig1c_decision_map.png", dpi=600)
print("kjV/Ab at d=0.5um  =", round(kd, 3))
print("delta-epsilon(G=1) =", round(de_Gunity, 3))
print("boundary beta at de=0.05 =", round(np.sqrt(2)/(kd*0.05) - 1, 3))
print("saved fig1c_decision_map.pdf / .png")