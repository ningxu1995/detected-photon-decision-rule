"""
fig3abc_panels.py  --  Figure 3(a), (b) and (c) from the measured images.

    (a) fluorescence image of the calibrated bead
    (b) recovered phase of the same bead, in radians, with the value at the
        resolution cell that sets the coupling G = 2 <phi>_cell
    (c) the same fluorescence image with the signal cell and the background
        annulus used for beta = b/A

Conventions (Supporting Information of the revision)
    signal cell      one resolution cell, area A_eff = lambda^2/(pi NA^2),
                     M_CELL = 10 pixels of 65 nm
    annulus          M_ANN = 196 pixels, concentric with the cell
    A, b             background-subtracted signal and background photons in the
                     cell;  beta = b / A
    G                2 |<phi>|, the photon-weighted mean of the recovered phase
                     over the cell, weight h = |u|^2 of the coherent pupil

The script prints A, b, beta, the peak and cell-averaged phase and G, so that the
numbers in Table S4 and in the text can be checked against the images.

Usage
    python fig3abc_panels.py --fluor fluor.npy --phase phase.npy
    python fig3abc_panels.py --preview          # layout only, clearly watermarked

Accepted image formats: .npy, .npz, .tif, .png (any 2D array of detected
photoelectrons for the fluorescence panel and radians for the phase panel).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import j1

# ------------------------------------------------------------------ constants
PIXEL_NM = 65.0
LAM, NA = 0.500, 1.4          # um
M_CELL, M_ANN = 10, 196       # pixels
CROP_PX = 96
SCALE_BAR_NM = 500
CMAP_F, CMAP_P = "viridis", "magma"

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans"})


# ------------------------------------------------------------------ helpers
def load_image(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        d = np.load(path); arr = d[list(d.keys())[0]]
    else:
        try:
            import imageio.v2 as imageio
            arr = imageio.imread(path)
        except Exception:
            from PIL import Image
            arr = np.array(Image.open(path))
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1) if arr.shape[-1] in (3, 4) else arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D image, got {arr.shape}")
    return arr


def radii():
    """cell and annulus radii in pixels for the stated pixel counts."""
    r_cell = np.sqrt(M_CELL / np.pi)
    r_in = 3.0 * r_cell
    r_out = np.sqrt(r_in ** 2 + M_ANN / np.pi)
    return r_cell, r_in, r_out


def psf_weight(shape, centre, pixel_nm=PIXEL_NM):
    """normalized intensity point-spread function h = |u|^2, unit integral."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - centre[0], yy - centre[1]) * pixel_nm / 1e3
    v = 2 * np.pi * NA * r / LAM
    u = np.where(v < 1e-9, 1.0, 2 * j1(np.where(v < 1e-9, 1e-9, v)) / np.where(v < 1e-9, 1e-9, v))
    h = u ** 2
    return h / h.sum()


def find_centre(img, smooth=2.0):
    from scipy.ndimage import gaussian_filter
    iy, ix = np.unravel_index(np.nanargmax(gaussian_filter(img, smooth)), img.shape)
    return float(ix), float(iy)


def crop(img, centre, size=CROP_PX):
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    half = size // 2
    out = np.full((size, size), np.nan)
    x0, x1 = max(0, cx - half), min(img.shape[1], cx + half)
    y0, y1 = max(0, cy - half), min(img.shape[0], cy + half)
    sub = img[y0:y1, x0:x1]
    oy, ox = (size - sub.shape[0]) // 2, (size - sub.shape[1]) // 2
    out[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
    return out, (size / 2 - 0.5, size / 2 - 0.5)


def extract(fluor, phase, centre):
    """A, b, beta from the fluorescence crop and <phi>, G from the phase crop."""
    ny, nx = fluor.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - centre[0], yy - centre[1])
    r_cell, r_in, r_out = radii()
    cell = rr <= r_cell
    ann = (rr >= r_in) & (rr <= r_out)
    bg_per_px = np.nanmean(fluor[ann])
    b = bg_per_px * cell.sum()
    A = np.nansum(fluor[cell]) - b
    h = psf_weight(fluor.shape, centre)
    hc = h * cell
    phi_cell = float(np.nansum(hc * phase) / hc.sum())
    return dict(A=A, b=b, beta=b / A, n_cell=int(cell.sum()), n_ann=int(ann.sum()),
                bg_per_px=bg_per_px, phi_peak=float(np.nanmax(phase[cell])),
                phi_cell=phi_cell, G=2 * abs(phi_cell), cell=cell, ann=ann,
                radii=(r_cell, r_in, r_out))


def scale_bar(ax, shape, bar_nm=SCALE_BAR_NM, margin=6):
    h, w = shape
    bar = bar_nm / PIXEL_NM
    ax.plot([margin, margin + bar], [h - margin - 2] * 2, color="white", lw=4,
            solid_capstyle="butt")
    ax.text(margin, h - margin - 6, f"{int(bar_nm)} nm", color="white", fontsize=9,
            ha="left", va="bottom")


# ------------------------------------------------------------------ figure
def make_figure(fluor, phase, centre=None, beta_report=None, out="Fig_3abc.png",
                preview=False):
    if centre is None:
        centre = find_centre(fluor)
    fc, c = crop(fluor, centre)
    pc, _ = crop(phase, centre)
    st = extract(fc, pc, c)
    r_cell, r_in, r_out = st["radii"]

    fig = plt.figure(figsize=(11.0, 3.6), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    fv = np.nanpercentile(fc, [0.5, 99.8])
    ax1.imshow(fc, cmap=CMAP_F, vmin=fv[0], vmax=fv[1], interpolation="nearest")
    ax1.set_title("same bead, fluorescence", fontsize=11.5, pad=12)
    scale_bar(ax1, fc.shape)

    pv = np.nanpercentile(pc, [0.5, 99.8])
    im = ax2.imshow(pc, cmap=CMAP_P, vmin=min(0, pv[0]), vmax=pv[1], interpolation="nearest")
    ax2.set_title("same bead, recovered phase", fontsize=11.5, pad=12)
    cax = make_axes_locatable(ax2).append_axes("right", size="6%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("phase (rad)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax2.text(0.04, 0.96, f"$\\langle\\varphi\\rangle_{{\\rm cell}}={st['phi_cell']:.3f}$ rad\n"
                         f"$G=2\\langle\\varphi\\rangle={st['G']:.3f}$",
             transform=ax2.transAxes, ha="left", va="top", fontsize=9.5, color="white")

    ax3.imshow(fc, cmap=CMAP_F, vmin=fv[0], vmax=fv[1], interpolation="nearest")
    ax3.set_title("background extraction", fontsize=11.5, pad=12)
    for r, ls in ((r_cell, "-"), (r_in, "--"), (r_out, "--")):
        ax3.add_patch(Circle(c, r, edgecolor="white", facecolor="none", lw=2.0, ls=ls))
    beta_txt = (f"$\\beta=b/A={beta_report}$" if beta_report is not None
                else f"$\\beta=b/A={st['beta']:.2f}$")
    ax3.text(0.05, 0.94, beta_txt, transform=ax3.transAxes, ha="left", va="top", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8", alpha=0.92))
    ax3.annotate("signal cell", xy=(c[0] + r_cell * 0.7, c[1] - r_cell * 0.7),
                 xytext=(0.13, 0.16), textcoords="axes fraction", color="white", fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.6))
    ax3.annotate("background\nannulus", xy=(c[0] + r_out * 0.75, c[1] + r_out * 0.55),
                 xytext=(0.60, 0.06), textcoords="axes fraction", color="white", fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.6))

    for ax, tag in zip((ax1, ax2, ax3), ("(a)", "(b)", "(c)")):
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.0, 1.20, tag, transform=ax.transAxes, fontsize=14, fontweight="bold",
                ha="left", va="top")
    if preview:
        for yy in (0.28, 0.62):
            fig.text(0.5, yy, "LAYOUT PREVIEW - SYNTHETIC DATA", fontsize=18, color="red",
                     alpha=0.28, ha="center", va="center", rotation=12, weight="bold")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f" cell {st['n_cell']} px (r = {r_cell:.2f} px), annulus {st['n_ann']} px "
          f"(r = {r_in:.2f} to {r_out:.2f} px)")
    print(f" A = {st['A']:.0f}   b = {st['b']:.0f}   beta = {st['beta']:.2f}"
          f"   background {st['bg_per_px']:.1f} per pixel")
    print(f" phase: peak {st['phi_peak']:.3f} rad, cell average {st['phi_cell']:.3f} rad"
          f"  ->  G = {st['G']:.3f},  N_L = 2 SNR*^2/G^2 = {2*100/st['G']**2:.0f} at SNR* = 10")
    print(f" saved {out} and {out.replace('.png', '.pdf')}")
    return st


def preview_data(n=168, seed=7):
    """synthetic images for checking the layout only."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    cx, cy = 84.0, 83.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    fluor = rng.poisson(np.clip(900 * np.exp(-r2 / (2 * 4.2 ** 2)) + 40, 0, None)).astype(float)
    phase = 0.10 * np.exp(-r2 / (2 * 5.0 ** 2)) + rng.normal(0, 0.002, (n, n))
    return fluor, phase


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 3(a)(b)(c) from the measured images.")
    p.add_argument("--fluor"); p.add_argument("--phase")
    p.add_argument("--centre-x", type=float); p.add_argument("--centre-y", type=float)
    p.add_argument("--beta-report", type=str, default=None,
                   help="text printed in panel (c), e.g. '3.0\\\\pm0.3'")
    p.add_argument("--out", default="Fig_3abc.png")
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()

    if a.preview or (a.fluor is None and a.phase is None):
        if not a.preview:
            raise SystemExit("give --fluor and --phase, or --preview for a layout check")
        f, ph = preview_data()
        make_figure(f, ph, out=a.out, preview=True)
    else:
        if a.fluor is None or a.phase is None:
            raise SystemExit("both --fluor and --phase are required")
        centre = (a.centre_x, a.centre_y) if a.centre_x is not None else None
        make_figure(load_image(a.fluor), load_image(a.phase), centre=centre,
                    beta_report=a.beta_report, out=a.out)
