# Frame stiffness damage identification

Code and synthetic data for **Stiffness Damage Identification of Frame Structures Based on Modal Features and Deep Learning**, by Zhenyu Zhao.

This repository contains the original experiment archive and the additional analyses for the November 2026 manuscript revision. Original results are preserved in `reference_results/`; new analyses are recorded separately in `revision_results/`. No external benchmark validation is included.

## Contents

- `src/`: Python simulation, validation, analysis and plotting modules; Chinese explanatory comments are preserved. The generator now optionally records actual damage values without changing its random draws or default output.
- `config/experiment_config.json`: structural, uncertainty, generation and validation settings.
- `data/synthetic_shm_dataset.csv`: 1,560 synthetic cases, with four label fields and 28 features.
- `reference_results/`: archived results from the original full run, including its environment record. These files are kept separate from newly generated results.
- `scripts/verify_archive.py`: validates dataset structure, archived result consistency and file checksums without training models.
- `scripts/verify_regeneration.py`: regenerates the data in memory and compares against the archived CSV without overwriting it.
- `revision_results/`: matched per-split predictions, factorized intervals, paired comparison, actual damage audit and print-readable figures.
- `scripts/revision_analysis.py` and `scripts/revision_figures.py`: reproduce these additional analyses.
- `docs/REVISION_STATUS.md`: verification results and remaining limitations.

This is a four-degree-of-freedom simulated shear-frame study, not the IASC–ASCE benchmark. The 10%, 20%, and 30% damage classes are nominal values; the generator samples actual damage ratios with Gaussian SD 0.012 and clips them to [0.04, 0.42].

## Environment

The archived full experiment records Windows 10, Python 3.8.20, NumPy 1.24.3, pandas 1.3.4, SciPy 1.10.1, scikit-learn 1.3.2, matplotlib 3.5.0 and joblib 1.4.2. `requirements.txt` pins those library versions; it is not a complete operating-system or transitive-dependency lockfile.

Using Anaconda Prompt, create the environment, activate it, and change to this repository folder before running commands:

```bash
conda create -n frame-shm python=3.8.20 -y
conda activate frame-shm
python -m pip install -r requirements.txt
```

Installing these historical package versions into a newer Python environment may fail. Reproducing model fitting also depends on numerical-library/platform behavior; match the recorded environment where possible and compare the outputs.

## Check the supplied archive

```bash
python scripts/verify_archive.py
python scripts/verify_regeneration.py
```

The regeneration check uses the original generator and supplied configuration. A prior full comparison reproduced all 1,560 rows in the same order, with matching labels and maximum absolute numeric difference about 5e-12 before CSV rounding. This validates data regeneration, not independent external generalization.

## Reproduce the analysis using the provided data

Run from the repository root. Do not run individual files inside `src/` directly; their relative imports require module execution.

```bash
python -m src.check_dataset
python -m src.run_validation
python -m src.analyze_features
python -m src.plot_outputs
```

Outputs go to `results/`, which is ignored by Git. Archived `reference_results/` remains unchanged. For a one-seed pipeline check, use `--quick` with the last three commands; quick settings are not the ten-run paper results.

To regenerate the CSV and run the whole pipeline:

```bash
python -m src.run_all
```

This last command **overwrites `data/synthetic_shm_dataset.csv`** and creates fresh results. Use the non-overwriting regeneration check above when comparing with the original archive. The simulator computes acceleration histories transiently but the archive retains summary features, not raw time-series files.

## Reproduce the additional reviewer analyses

```bash
python scripts/revision_analysis.py stats
python scripts/revision_analysis.py train
python scripts/revision_analysis.py generate
python scripts/revision_figures.py
```

These commands write to `revision_results/` without changing the original dataset or archived results. The training command refits only the full-feature MLP on the ten original splits; it does not rerun every baseline. The paired test uses the archived matched modal-MLP and random-forest results. Its corrected resampled standard error is `s(d) * sqrt(1/10 + 312/1248)`, with 9 degrees of freedom, following the [scikit-learn statistical comparison example](https://scikit-learn.org/1.3/auto_examples/model_selection/plot_grid_search_stats.html). This approximation accounts for resampling dependence; it does not establish equivalence or external validity.

The new audit ran with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0. All ten archived split accuracies, macro-F1 scores, stopping epochs, per-class F1 scores and the pooled confusion matrix were reproduced. See `revision_results/rerun_validation.json`. Matching these outputs does not imply that every library/platform combination will reproduce them.

The actual damage audit is in generation order before the original dataset shuffle; its row numbers must not be joined directly to shuffled dataset row numbers. Numeric regeneration matches the original 1,560 rows within `rtol=1e-9, atol=1e-12`, with maximum absolute error about `5e-12`.

## Results and limits

The archived full results report mean test accuracy of approximately 85.0641% for full-feature MLP, 88.9744% for 12-feature modal MLP, and 88.3333% for full-feature random forest. Use the CSV tables rather than the older manuscript's random-forest value. Repeated splits reuse one simulation population; they are not independent experimental structures. Do not interpret quick-mode outputs as the paper's main results.

## Availability and licensing

The code, synthetic dataset, configuration and archived reference results are publicly available at [zzy15541531566-jpg/frame-shm-reproducibility](https://github.com/zzy15541531566-jpg/frame-shm-reproducibility).

The software code in `src/` and `scripts/`, together with its configuration and associated software documentation, is licensed under the [MIT License](LICENSE). Copyright (c) 2026 Zhenyu Zhao. Third-party dependencies retain their own licenses.

This software license does not assign a separate reuse license to the dataset and reference-result assets in `data/`, `reference_results/` and `revision_results/`; their licensing remains to be specified.

Use the exact Git commit identifier in the revised manuscript to retrieve the corresponding code/data snapshot. No DOI or release tag is claimed. The retained `frame-shm-reproducibility-draft.zip` predates the license addition; use the current repository files and license notice.
