"""
Fig 3(b)  --  FORWARD-SIMULATED fluorescence photon series on a calibrated bead.

Relative precision on the brightness, sigma_c/c, versus detected photons N. It
follows the single-mode contrast-limited Cramer-Rao bound of the Letter/SM,
        sigma_c/c = (1+beta)/sqrt(N),   C_F = 1/(1+beta),
reaching the operating target 1/SNR* = 0.1 at the fluorescence cost
        N_F = (1+beta)^2 * SNR*^2.
Maximum-likelihood points (Poisson signal+background, background mean known)
attain the bound across the range; the inset shows the brightness estimate at
N_F is unbiased and Gaussian with width 1/SNR*. This is consistent with the
crossover fix (closed-form N_F): beta=3 -> N_F=1600. SET BETA to match Fig 3(d).

Replace with the real calibrated-bead photon series when available.
Needs numpy, matplotlib.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D

rng = np.random.default_rng(11)

# ---------------- knobs ----------------
BETA    = 3.0          # background ratio (MATCH Fig 3(d); 3 -> N_F=1600, 10 -> 12100)
SNRstar = 10.0
M_TRIAL = 8000         # MC trials per point
COL     = '#6a1b9a'    # fluorescence color (matches Fig 1b)
STAR_FC = '#B48A30'    # muted gold for the target-cost marker
STAR_EC = '#2F2A20'    # dark edge, less harsh than pure black

NF = (1.0+BETA)**2 * SNRstar**2
def crlb_rp(N):
    return (1.0+BETA)/np.sqrt(N)
def mc_fluor(N, M=M_TRIAL):
    sig = N/(1.0+BETA); bg = BETA*N/(1.0+BETA)
    T = rng.poisson(sig+bg, M).astype(float)
    rel = (T - bg - sig)/sig          # (s_hat - s)/s, with background mean known
    return rel.std(), rel.mean(), rel

# ---------------- data ----------------
Ngrid = np.geomspace(40, 3e4, 200)
rp_line = crlb_rp(Ngrid)
Npts = np.array([60, 160, 400, 1000, NF, 4000, 1e4, 2.5e4])
mc = [mc_fluor(n) for n in Npts]
rp_pts = np.array([m[0] for m in mc])
rp_err = rp_pts/np.sqrt(2*M_TRIAL)

# ---------------- figure ----------------
mpl.rcParams.update({
    "font.family":"sans-serif",
    "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "mathtext.fontset":"dejavusans",
    "font.size":15, "axes.labelsize":19,
    "xtick.labelsize":15, "ytick.labelsize":15,
    "legend.fontsize":12, "axes.linewidth":1.2,
})
fig, ax = plt.subplots(figsize=(7.4, 6.0))

# CRLB line + ML points
ax.plot(Ngrid, rp_line, color=COL, lw=2.8, zorder=4,
        label='Cram\u00e9r\u2013Rao bound $(1{+}\\beta)/\\sqrt{N}$')
ax.errorbar(Npts, rp_pts, yerr=rp_err, fmt='o', color=COL, mec='k', mew=0.8,
            ms=8.5, capsize=2.5, zorder=6, label='maximum likelihood')

# target + N_F
ax.axhline(1/SNRstar, ls='--', color='0.38', lw=1.5, zorder=3)
ax.text(46, 0.108, 'target  $1/\\mathrm{SNR}^{\\ast}=0.1$', color='0.30',
        fontsize=13, va='bottom', ha='left')
ax.plot([NF], [1/SNRstar], marker='*', linestyle='None',
        markerfacecolor=STAR_FC, markeredgecolor=STAR_EC, markeredgewidth=0.9,
        color=STAR_EC, ms=18, zorder=8, label='photon cost at target')
ax.annotate(f'$N_F = {NF:.0f}$  photons', (NF, 1/SNRstar),
            textcoords='offset points', xytext=(12,11), fontsize=14, color='k',
            arrowprops=dict(arrowstyle='->', color='0.3', lw=1.3))
ax.plot([NF,NF],[ax.get_ylim()[0],1/SNRstar], ls=':', color=COL, lw=1.5, zorder=2)

# slope guide
ax.text(150, 0.40, '$\\propto N^{-1/2}$', color=COL, fontsize=15, rotation=-26,
        rotation_mode='anchor', ha='left', va='center')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(40, 3e4); ax.set_ylim(0.02, 1.0)
ax.set_xticks([1e2,1e3,1e4]); ax.set_yticks([0.1,1.0])
ax.set_yticklabels(['0.1','1'])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('detected photons  $N$')
ax.set_ylabel('relative precision  $\\sigma_c/c$')
ax.grid(True, which='both', alpha=0.18)
ax.legend(loc='lower left', framealpha=0.95)

# inset: estimate distribution at N_F (unbiased, width 1/SNR*)
_,_,rel = mc_fluor(NF, 60000)
axin = fig.add_axes([0.60, 0.57, 0.30, 0.27])
axin.hist(rel, bins=46, density=True, color=COL, alpha=0.65, edgecolor='none')
xx = np.linspace(-0.45,0.45,300)
axin.plot(xx, np.exp(-xx**2/(2*0.1**2))/(0.1*np.sqrt(2*np.pi)), color='k', lw=1.6)
axin.axvline(0, color='0.4', lw=0.8)
axin.set_title('estimate at $N_F$  (unbiased)', fontsize=11)
axin.set_xlabel('$(\\hat c-c)/c$', fontsize=11)
axin.set_xticks([-0.2,0,0.2]); axin.set_yticks([])
axin.tick_params(labelsize=10); axin.set_xlim(-0.42,0.42)
for s in ('top','right','left'): axin.spines[s].set_visible(False)

for ext in ('png','pdf','svg'):
    fig.savefig(f'fig3b_fluor_series.{ext}', dpi=600, bbox_inches='tight')
print(f"saved fig3b_fluor_series.* ; N_F={NF:.0f}, MC r at N_F = {crlb_rp(NF)/mc_fluor(NF,60000)[0]:.3f}")
