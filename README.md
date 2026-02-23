[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![BayesFlow](https://img.shields.io/badge/BayesFlow-1.1.4-blue)](https://bayesflow.org/)
[![lab.js](https://img.shields.io/badge/lab.js-20.2.4-blue)](https://labjs.felixhenninger.com/)
[![PsychoPy](https://img.shields.io/badge/PsychoPy-2022.1.4-blue)](https://www.psychopy.org/)

---

## Project Overview

This repository contains all materials, data, scripts, and analysis pipelines associated with the manuscript
“**Time to Jump: Exploring the Distribution of Noise in Evidence Accumulation as a Function of Time Pressure**.”

This project combines simulation and experimental approaches to investigate how the stability parameter α in the Lévy-flight model (Voss et al., 2019) shapes response time distributions and responds to different levels of time pressure.

---

## Repository Structure

Study 2 files typically use the affix `_s2` (or are located in `s2_*` folders), while Study 1 files do not.

### 1. Introduction (`1_introduction/`)
- `density_sample_paths.ipynb`: Visualizes the assumptions of the Lévy-flight model.
- `density_sample_paths.png`: Output image from the visualization notebook.
- `alpha_behavior.ipynb`: Simulates α-values and shows how they shape RT distributions.
- Simulation-related output images: `sim_behav_dense.png`, `sim_skew_combined.png`, `sim_skew_mod.png`.
- `simulate_alpha_behavior.pkl`: Pregenerated simulation data.
- Supporting files necessary to run the simulations.

### 2. Experiment Script (`2_experiment_script/`)
- Study 1
	- `lnt.json`: Script for running the letter-number discrimination task in [lab.js](https://labjs.felixhenninger.com/).
- Study 2
	- `s2_psychopy/`: Scripts and material for running the brightness discrimination task via [PsychoPy/PsychoJS](https://www.psychopy.org/).
	- `s2_sosci/`: Questionnaire materuaks and documentation for [SoSci Survey](https://www.soscisurvey.de/).

### 3. Simulators
- Model comparison and parameter estimation notebooks:
  - `model_comp_no_deadline.ipynb`
  - `model_comp_deadline.ipynb`
  - `param_estim_no_deadline.ipynb`
  - `param_estim_deadline.ipynb`
  - `param_estim_no_deadline_s2.ipynb`

- Helper functions: `functions_model_comp.py`, `functions_param_estim.py`, and `functions_param_estim_s2.py`.
- Supporting files necessary to run the simulations.

	#### 3.1 Neural Network Checkpoints
	- Saved network checkpoints for different model training runs (`checkpoint_ttj_mod_comp_*`, `checkpoint_ttj_presimulation_*`, and `checkpoint_ttj_s2_presimulation_*`).

	#### 3.2 Data
	- Study 1
		- `ttj_raw_data.csv`: Raw behavioral data.
		- `ttj_raw_data_variables_explained.xlsx`: Codebook for raw data variables.
		- `ttj_result_empirical_osy.csv`: Preprocessed behavioral data.
		- `individual_data/`: Individual participant datasets.
	- Study 2
		- `ttj_s2_raw_data.csv`: Raw behavioral data.
		- `ttj_s2_raw_data_variables_explained.xlsx`: Codebook for raw data variables.
		- `ttj_s2_result_empirical_wsy.csv`: Preprocessed behavioral data.
		- `individual_data_s2/`: Individual participant datasets.

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
- R scripts for frequentist and Bayesian tests.

---

## Requirements

- Python 3.10
- Key dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `tensorflow`, `bayesflow`, etc.
- See `dependencies.txt` for full details.
> **Note**: To run the notebooks that simulate or fit the Lévy-flight model, ensure that the Cython module `levy_noise` is properly compiled (instructions in `my_setup.py`).

---

## Citation

If you use (parts of) this repository, please consider citing the associated manuscript (once published) and/or relevant papers, e.g.,

- Henninger, F., Shevchenko, Y., Mertens, U. K., Kieslich, P. J., & Hilbig, B. E. (2021). Lab.js: A free, open, online study builder. *Behavior Research Methods*, *54*, 556–573. https://doi.org/10.3758/s13428-019-01283-5
- Peirce, J., Gray, J. R., Simpson, S., MacAskill, M., Höchenberger, R., Sogo, H., Kastman, E., & Lindeløv, J. K. (2019). PsychoPy2: Experiments in behavior made easy. *Behavior Research Methods*, *51*(1), 195–203. https://doi.org/10.3758/s13428-018-01193-y
- Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Kothe, U. (2022). BayesFlow: Learning complex stochastic models with invertible neural networks. *IEEE Transactions on Neural Networks and Learning Systems*, *33*(4), 1452–1466. https://doi.org/10.1109/TNNLS.2020.3042395
- Radev, S. T., Schmitt, M., Schumacher, L., Elsemüller, L., Pratz, V., Schälte, Y., Köthe, U., & Bürkner, P.-C. (2023). *BayesFlow: Amortized Bayesian workflows with neural networks*. arXiv. https://doi.org/10.48550/arXiv.2306.16015

---