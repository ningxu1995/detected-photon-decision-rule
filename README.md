# Photon-budget code for fluorescence and label-free microscopy comparison

This repository contains the Python scripts used to generate selected figures for the manuscript:

**A detected-photon decision rule for choosing between fluorescence and interferometric label-free microscopy**

The code reproduces the photon-budget calculations, decision maps, Cramér–Rao scaling curves, and selected simulation/estimation results shown in the manuscript.

## Included figure scripts

| Figure  | Script     | Description                                                                                                                                                                   |
| ------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fig. 1b | `fig1b.py` | Single-object photon budget versus bead diameter. Shows the fluorescence photon cost, label-free photon cost, predicted crossover points, and weak-scattering validity range. |
| Fig. 1c | `fig1c.py` | Decision map in fluorescence signal-to-background ratio and permittivity contrast. Shows the boundary separating fluorescence-favored and label-free-favored regimes.         |
| Fig. 2  | `fig2.py`  | Photon-budget consequences across object density, axial conditioning, and resolution encoding.                                                                                |
| Fig. 3b | `fig3b.py` | Fluorescence photon series. Shows relative brightness precision versus detected photon number and comparison with the Cramér–Rao bound.                                       |
| Fig. S1 | `figS1.py` | Predicted crossover diameter versus numerical aperture and permittivity contrast.                                                                                             |

## Requirements

The scripts were written in Python and require the following packages:

```bash
python >= 3.9
numpy
scipy
matplotlib
```

A minimal environment can be created with:

```bash
conda create -n photon-budget python=3.10 numpy scipy matplotlib
conda activate photon-budget
```

or with `pip`:

```bash
pip install numpy scipy matplotlib
```

## Running the scripts

Each script can be run independently. For example:

```bash
python fig1b.py
python fig1c.py
python fig2.py
python fig3b.py
python figS1.py
```

The scripts save figure files in common formats such as `.png`, `.pdf`, and/or `.svg`, depending on the script.

## Physical model

The calculations use the detected-photon budget derived from the Cramér–Rao bound,

```math
N \gtrsim N_{\rm vox} p \kappa \frac{({\rm SNR}^*)^2}{C^2},
```

where (N) is the number of detected photons, (p) is the number of real parameters per voxel, (\kappa) is the Fisher-matrix conditioning factor, and (C) is the per-photon contrast.

For a single subresolution object, the fluorescence and interferometric label-free costs are

```math
N_{\rm F} \simeq ({\rm SNR}^*)^2(1+\beta)^2,
```

and

```math
N_{\rm L} \simeq \frac{2({\rm SNR}^*)^2}{G^2},
\qquad
G=\frac{k_0 V}{A_{\rm coh}}\Delta\epsilon .
```

The single-object decision boundary is

```math
G(1+\beta)=\sqrt{2}.
```

Equivalently, label-free interferometry requires fewer detected photons when

```math
\frac{k_0V}{A_{\rm coh}}\Delta\epsilon(1+\beta)>\sqrt{2},
```

and fluorescence requires fewer detected photons otherwise.

## Parameters

Unless otherwise stated, the scripts use the representative parameters from the manuscript:

```text
NA = 1.4
lambda = 500 nm
n = 1.515
SNR* = 10
Delta epsilon = 0.05
beta = 3 or 10
```

The exact parameter values used for each panel are defined at the beginning of the corresponding script.

## Data and reproducibility

The scripts reproduce the plotted photon-budget curves and model-based estimates used in the manuscript figures. Panels that include experimental summary values use the calibrated photon-budget values reported in the manuscript.

Raw camera frames are not included in this repository. The code is intended to document the figure-generation workflow, photon-budget calculations, and estimator simulations used for the reported analysis.

## Citation

If you use this code, please cite the associated manuscript:

```text
N. Xu, "A detected-photon decision rule for choosing between fluorescence and interferometric label-free microscopy," manuscript in preparation.
```

The citation will be updated after publication.

## License

Please see the `LICENSE` file for reuse terms.
