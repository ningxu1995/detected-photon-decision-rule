# Photon-budget code and experimental source data for fluorescence and coherent-scattering localization

This repository contains the Python scripts and processed experimental source-data tables used to generate selected figures and supplementary tables for the manuscript:

**Photon supply sets the mass crossover between fluorescence and coherent-scattering localization**

The code reproduces the photon-budget calculations, Cramér–Rao scaling curves, mass-dependent crossover predictions, residual-speckle analysis, and selected experimental figure panels reported in the manuscript.

## Included experimental data files

| File                             | Contents                                                                                                                                                                                                                                                                                                                                                                                                     | Used for                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| `experimental_tables-Final.xlsx` | Processed experimental source data for all Figure 4 panels, experimental supplementary-table values, and Extended Data Fig. 2a–c. The workbook contains the calibrated photon-count data, object-level summaries, block-level localization results, bleaching-floor measurements, stable-scatterer accumulation data, residual-speckle measurements, and quality-control information used in the manuscript. | Fig. 4, Supplementary Tables, Extended Data Fig. 2a–c |
| `experimental_ExtDataFig2d.xlsx` | Processed experimental source data for Extended Data Fig. 2d. This file is kept separate because the panel uses its own experimental run and panel-specific analysis branch.                                                                                                                                                                                                                                 | Extended Data Fig. 2d                                 |

These workbooks contain the processed experimental values used to generate the reported plots and tables. Raw camera frames are not included in this repository unless separately deposited.

## Included figure scripts

| Figure                  | Script                         | Description                                                                                                                                                                                                 | Data source                                                 |
| ----------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Fig. 1                  | `fig1.py`                      | Per-signal-photon localization equivalence between fluorescence and coherent scattering. Includes numerical checks of the single-photon localization width across numerical aperture and pupil apodization. | Model calculation                                           |
| Fig. 2                  | `fig2.py`                      | Photon-supply-limited mass crossover under dose-referenced and detected-photon budget conventions.                                                                                                          | Model calculation                                           |
| Fig. 3                  | `fig3.py`                      | Achievable crossover under residual-speckle floors, including speckle-limited and dose-limited branches.                                                                                                    | Model calculation with experimentally calibrated parameters |
| Fig. 4                  | `fig4.py`                      | Same-particle experimental validation: photon-calibrated localization, mass-dependent precision ratio, stable-scatterer accumulation, and residual-speckle-dependent crossover.                             | `experimental_tables-Final.xlsx`                            |
| Extended Data Fig. 2a–c | `extended_data_fig2abc.py`     | Experimental calibration and control panels associated with the Figure 4 measurements.                                                                                                                      | `experimental_tables-Final.xlsx`                            |
| Extended Data Fig. 2d   | `extended_data_fig2d.py`       | Panel-specific experimental source data and analysis for Extended Data Fig. 2d.                                                                                                                             | `experimental_ExtDataFig2d.xlsx`                            |
| Supplementary Tables    | `make_supplementary_tables.py` | Regenerates the experimental supplementary tables from the processed source-data workbooks.                                                                                                                 | `experimental_tables-Final.xlsx`                            |

If a local copy of the repository uses different script names, the figure-to-data mapping above should be treated as the authoritative reference.

## Requirements

The scripts were written in Python and require the following packages:

```text
python >= 3.9
numpy
scipy
matplotlib
pandas
openpyxl
```

A minimal conda environment can be created with:

```bash
conda create -n photon-budget python=3.10 numpy scipy matplotlib pandas openpyxl
conda activate photon-budget
```

or with pip:

```bash
pip install numpy scipy matplotlib pandas openpyxl
```

## Running the scripts

Each script can be run independently. The experimental Excel files should be placed in the repository root, unless the data path is changed inside the corresponding script.

Examples:

```bash
python fig1.py
python fig2.py
python fig3.py
python fig4.py
python extended_data_fig2abc.py
python extended_data_fig2d.py
python make_supplementary_tables.py
```

The scripts save figure files in common formats such as `.png`, `.pdf`, and/or `.svg`, depending on the script.

## Physical model

The calculations use the common single-photon localization bound for a point-like particle imaged through a diffraction-limited aperture,

```math
\sigma_x \geq \frac{\sigma_0}{\sqrt{N_{\mathrm{eff}}}},
\qquad
\sigma_0 = \frac{\lambda}{2\pi\,\mathrm{NA}} .
```

For lateral localization of an in-focus point source, fluorescence and optimal-quadrature coherent scattering have the same per-detected-signal-photon Fisher information. The distinction between the two modalities therefore enters through photon supply rather than through the information carried by each signal photon.

For fluorescence, the usable photon number is limited by emission brightness, background, and photobleaching. For coherent scattering, the usable signal grows with the elastic scattering strength but is eventually limited by residual coherent background. The interferometric coupling is parameterized as

```math
G = \frac{k_0\,n_m^{2}\,\alpha}{\varepsilon_0\,A_{\mathrm{coh}}},
```

where $\alpha$ is the particle polarizability and $A_{\mathrm{coh}}$ is the effective coherent detection area.

The scripts report both dose-referenced and detected-photon budget conventions where relevant. The delivered-dose convention is used for live or fragile samples where sample exposure is the limiting resource; the detected-photon convention is retained as a detector-load comparison.

## Experimental source-data structure

The processed experimental workbooks contain table- and panel-level source values. Where block-level data are provided, the main columns include:

```text
particle_id
object_class
channel
session_id
block_id
n_frames
N_detected
x_hat_nm
y_hat_nm
sigma_x_nm
background_beta
sres_session
accepted
rejection_reason
nominal_condition
roi_id
psd_class
mass_MDa
```

Additional columns may be included for panel-specific quantities such as numerical aperture, single-photon localization width, residual-speckle spectrum, object diameter, polarizability-equivalent mass, or quality-control flags.

## Data and reproducibility

The repository is intended to document the figure-generation workflow, photon-budget calculations, and processed experimental values used for the reported analysis. The experimental workbooks contain the values used in the plotted figure panels and supplementary tables.

Raw camera frames are not included in this repository. Reanalysis from raw frames requires the original image stacks, calibration files, registration transforms, and particle-detection outputs, which are not part of this code release unless separately deposited.

## Citation

If you use this code or source-data structure, please cite the associated manuscript:

N. Xu and H. Zhang, "Photon supply sets the mass crossover between fluorescence and coherent-scattering localization," manuscript in preparation.

The citation will be updated after publication.

## License

Please see the `LICENSE` file for reuse terms.
