[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![BayesFlow](https://img.shields.io/badge/BayesFlow-1.1.4-blue)](https://bayesflow.org/)
[![lab.js](https://img.shields.io/badge/lab.js-20.2.4-blue)](https://labjs.felixhenninger.com/)

---

## Project Overview

This repository contains all materials, data, scripts, and analysis pipelines associated with the manuscript
**“Time to Jump: Exploring the Distribution of Noise in Evidence Accumulation as a Function of Time Pressure.”**

This project combines simulation and experimental approaches to investigate how the stability parameter α in the Lévy-flight model (Voss et al., 2019) shapes response time distributions, responds to different levels of time pressure, and demonstrates the model’s potential to capture decision-making dynamics under (deadline-based) time pressure. 

---

## Repository Structure

### 1. Introduction
- `density_sample_paths.ipynb`: Visualizes the assumptions of the Lévy-flight model.
- `density_sample_paths.png`: Output image from the visualization notebook.
- `alpha_behavior.ipynb`: Simulates α-values and shows how they shape RT distributions.
- Simulation-related output images: `sim_behav_dense.png`, `sim_skew_combined.png`, `sim_skew_mod.png`.
- `simulate_alpha_behavior.pkl`: Pregenerated simulation data.
- Supporting files necessary to run the simulations.

### 2. Experiment Script
- `lnt.json`: Script for running the experiment in [lab.js](https://labjs.felixhenninger.com/).

### 3. Simulators
- Model comparison and parameter estimation notebooks:
  - `model_comp_no_deadline.ipynb`
  - `model_comp_deadline.ipynb`
  - `param_estim_no_deadline.ipynb`
  - `param_estim_deadline.ipynb`

- Helper functions: `functions_model_comp.py`, `functions_param_estim.py`.
- Supporting files necessary to run the simulations.

	#### 3.1 Neural Network Checkpoints
	- Saved network checkpoints for different model training runs (`checkpoint_ttj_mod_comp_*` and `checkpoint_ttj_presimulation_*`).

	#### 3.2 Data
	- `ttj_raw_data.csv`: Raw behavioral data.
	- `ttj_raw_data_variables_explained.xlsx`: Codebook for raw data variables.
	- `ttj_result_empirical_osy.csv`: Preprocessed behavioral data.
	- Subfolder `individual_data`: Individual participant datasets.

	#### 3.3 Model Estimates
	- `estimates_model_comp/`: Posterior estimates from model comparison runs.
	- `estimates_param_estim/`: Posterior estimates from parameter estimation runs.

	#### 3.4 Figures
	- Plots generated during analyses, such as posterior predictive checks, parameter recovery, and calibration plots.

	#### 3.5 Source Code
	- `src/helpers.py`: Helper functions.
	- `src/visualization.py`: Visualization utilities.

	#### 3.6 Validation Files
	- Files used for simulation-based calibration and validation.

### 4. R Script
- R script for frequentist and Bayesian tests (`ttj_analyses.Rmd`, output:`ttj_analyses.html`).

---

## Requirements

- Python 3.10
- Key dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `tensorflow`, `bayesflow`, etc.
- See `dependencies.txt` for full details.
> **Note**: To run the notebooks that simulate or fit the Lévy-flight model, ensure that the Cython module `levy_noise` is properly compiled (instructions in `my_setup.py`).

---

## Citation

If you use (parts of) this repository, please consider citing the associated manuscript (once published) and/or relevant papers, e.g.
- Henninger, F., Shevchenko, Y., Mertens, U. K., Kieslich, P. J., & Hilbig, B. E. (2021). Lab.js: A free, open, online study builder. *Behavior Research Methods*, *54*, 556–573. https://doi.org/10.3758/s13428-019-01283-5
- Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Kothe, U. (2022). BayesFlow: Learning complex stochastic models with invertible neural networks. *IEEE Transactions on Neural Networks and Learning Systems*, *33*(4), 1452–1466. https://doi.org/10.1109/TNNLS.2020.3042395
- Radev, S. T., Schmitt, M., Schumacher, L., Elsemüller, L., Pratz, V., Schälte, Y., Köthe, U., & Bürkner, P.-C. (2023). *BayesFlow: Amortized Bayesian workflows with neural networks*. arXiv. https://doi.org/10.48550/arXiv.2306.16015

---

## Support

This research was supported by the Baden-Württemberg Stiftung through the “Eliteprogramm für Postdoktorandinnen und Postdoktoranden” (Elite Program for Postdoctoral Researchers) and by a grant from the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation; GRK 2277) to the research training group “Statistical Modeling in Psychology” (SMiP). 

---