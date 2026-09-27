# Detected-photon decision rule for fluorescence and interferometric label-free microscopy

Code and processed data for

> N. Xu and Q. Tan, *Detected-photon decision rule for fluorescence and interferometric label-free microscopy*, ACS Photonics (under review, manuscript ph-2026-01937).

The repository reproduces every quantitative figure of the main text and the Supporting Information from the equations of the paper and from the processed measurements, and it contains the source data of every panel.

## Contents

| File | What it does |
|---|---|
| `fig1_panels.py` | Figure 1(b) single-object photon budget versus diameter, Figure 1(c) decision map at *d* = 0.5 µm. Defines the optical model (`Optics`), the coupling *G*(*d*) of Eq. (S14) and the budgets *N*<sub>F</sub>, *N*<sub>L</sub>; the other theory scripts import from it. |
| `fig2_panels.py` | Figure 2: density crossover, regime map, axial *k*-space support, axial conditioning versus NA. |
| `fig3abc_panels.py` | Figure 3(a)–(c) from the measured fluorescence and recovered-phase images; extracts *A*, *b*, β, ⟨φ⟩ and *G* with the resolution-cell convention. |
| `fig3_analysis.py` | Figure 3(d)–(f) from `photon_series.xlsx`: free-slope fits of the photon series, photon costs at the target precision, size exponent, crossover diameters, efficiency ratios. |
| `figS4_panels.py` | Figure S4: attainment of the bound in both arms (a) and Monte Carlo bias of the four-step estimator (b). |
| `figS5_panels.py` | Figure S5: delivered-dose sensitivity analysis. |
| `figS2_panels.py` | Figure S2: crossover diameter versus numerical aperture for several permittivity contrasts. |
| `figS7_panels.py` | Figure S7: modulation coefficient of a sphere, closed form, projection model and three-dimensional first-Born field. |
| `FigS1.py` | Figure S1, lateral-resolution encoding (unchanged from the original submission). |
| `photon_series.xlsx` | Processed measurements of the 0.5 µm bead and of the bead-size series (three sheets, described below). |
| `source_data.xlsx` | Source data of every figure panel and table, one sheet per panel, with the fitted quantities in the sheet `Fits`. |
| `requirements.txt` | Python dependencies. |

Figures S3 (optical layout) and S6 (three-dimensional fluorescence volume) are images without a script.

## Conventions

All scripts use the conventions of the Supporting Information.

* A resolution cell is the effective area of the coherent point-spread function, *A*<sub>eff</sub> = λ²/(π NA²) = 0.041 µm² at NA = 1.4 and λ = 500 nm, which is *M*<sub>cell</sub> = 10 pixels of 65 nm. The background annulus has *M*<sub>ann</sub> = 196 pixels.
* *N* counts every photon detected in that cell over the frames of one estimate: signal plus background in fluorescence, reference photons summed over the four phase steps in the label-free arm.
* *G* = 2|*E*<sub>sc</sub>/*E*<sub>R</sub>| is the fringe-modulation coefficient of the cell, the photon-weighted mean of the scattered field over the cell (Eq. S14). For a sphere it is computed from the first-Born field with the projection (thin-object) model; `figS7_panels.py` also evaluates the three-dimensional first-Born field as a check.
* Photon costs are *N*<sub>F</sub> = SNR*² (1 + β)² and *N*<sub>L</sub> = 2 SNR*² / *G*² at the target precision 1/SNR* = 0.1, and the decision rule is *G* (1 + β) = √2.
* Cramér–Rao bounds drawn against measured series include the read-noise factor (1 + 4 *M*<sub>cell</sub> σ<sub>r</sub>² / *N*)<sup>1/2</sup> with σ<sub>r</sub> = 1.3 e⁻ (fluorescence camera) and 1.4 e⁻ (interferometric camera).

## Reproducing the figures

```
pip install -r requirements.txt

python fig1_panels.py                      # Fig_1b_budget.png, Fig_1c_map.png
python fig2_panels.py                      # Figure_2.png and the four panels
python fig3abc_panels.py --fluor fluor.npy --phase phase.npy --beta-report "3.0\pm0.3"
python fig3_analysis.py photon_series.xlsx # Fig_3d, Fig_3e, Fig_3f
python figS4_panels.py photon_series.xlsx  # Figure_S4.png
python figS5_panels.py                     # Figure_S5.png
python figS2_panels.py                     # Fig_S2_accessibility.png
python figS7_panels.py                     # Fig_S7_coupling.png
```

Every script prints the numbers it produces (couplings, crossovers, exponents, efficiency ratios, conditioning factors) so that the values quoted in the paper can be checked directly. `fig2_panels.py`, `figS2_panels.py`, `figS5_panels.py` and `figS7_panels.py` import from `fig1_panels.py`, and `figS4_panels.py` from `fig3_analysis.py`, so all scripts must be run from this folder. `fig3abc_panels.py` needs the measured images as 2D arrays (`.npy`, `.npz`, `.tif` or `.png`): detected photoelectrons for the fluorescence image and radians for the recovered phase; `--preview` draws a watermarked layout check from synthetic data and is not used for any figure of the paper.

## `photon_series.xlsx`

| Sheet | Content |
|---|---|
| 1 | Photon series of the 0.5 µm bead: for each arm (F = fluorescence at β = 3.0 and 9.3, L = label-free), each calibrated photon level *N*, the number of frames, the estimator mean, its standard deviation over frames, and the relative precision. |
| 2 | Per-frame normalized estimates at the operating points (300 fluorescence frames at *N*<sub>F</sub>, 400 label-free frames at *N*<sub>L</sub>), used for the insets of Figure 3(d) and 3(e). |
| 3 | Bead-size series: for each diameter the bead-to-bead means and standard deviations of *N*<sub>F</sub>, *N*<sub>L</sub> and *G*, and the number of beads. |

Photon numbers are detected photoelectrons from the photon-transfer-curve calibration of each camera (fluorescence arm 0.46 e⁻/ADU, 1.3 e⁻ rms; interferometric arm 0.47 e⁻/ADU, 1.4 e⁻ rms).

## Requirements

Python 3.10 or later with numpy, scipy, matplotlib and openpyxl (see `requirements.txt`). Arial is used for the figures when available; the scripts fall back to DejaVu Sans.

## Contact

Ning Xu, ningxuoptics@gmail.com
