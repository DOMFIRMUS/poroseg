# poroseg — segmentation-induced uncertainty of micro-CT porosity

Code for the article *"How much of the porosity measured on micro-CT slices is the analyst's choice? A factorial
uncertainty analysis with synthetic ground truth and laboratory comparison"* (Cardoso et al., submitted to
*Computers & Geosciences*). One command regenerates every number, table and figure of the article
from the raw micro-CT slices.

## What it does

`poroseg` segments 2-D micro-CT slices with eight thresholding algorithms (Otsu, Li, Yen, Mean,
Triangle, Isodata, Sauvola, Niblack) × four post-processing levels (none, median, opening–closing,
skeleton) × two region-of-interest designs, and then quantifies:

| Stage | Question |
|---|---|
| `segmentation` | the full factorial design (576 segmentations for the nine slices) |
| `descriptive` | histograms, bimodality, failures, pixel agreement between algorithms, ROI-design effect |
| `spatial` | two-point correlation, integral range, representative window size, block bootstrap vs pixel bootstrap, radial profiles |
| `sensitivity` | porosity–threshold curves, Threshold Sensitivity Matrix, dimensionless Pore Stability Index, local-parameter sweep, response of local methods to homogeneous fields |
| `perturbation` | robustness to controlled noise, cupping (beam hardening) and blur applied to the real slices |
| `morphology` | effect of post-processing on boundary length, Euler number, pore number density and pore-size distribution; grey-scale (partial-volume) porosity; unresolved porosity |
| `synthetic` | benchmark on 270 synthetic images with known ground truth |
| `ml_baseline` | random-forest (Weka/ilastik-type) and Gaussian-mixture baselines |
| `inference` | Friedman / RM-ANOVA, variance decomposition (material vs analyst choices), agreement with laboratory porosity |
| `legacy` | faithful re-implementation and audit of the original analysis code |
| `report` | catalogue of figures and tables, `key_numbers.json`, `RESULTS_DIGEST.md`, `tables_all.xlsx` |

## Installation

Python >= 3.10 (tested with Python 3.14 on Windows 11; Linux and macOS should work).

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run every command from the repository root (the folder that contains `config.yaml`), after placing the
input data as described in the next section.

```bash
python -m pytest -q tests                      # 106 unit tests (statistics validated against pingouin/scipy,
                                               # estimators against synthetic data with known answers); no data needed
python -m poroseg.run --stages all --quick     # fast smoke run (fewer replicates), ~15 min, written to outputs_quick/
python -m poroseg.run --stages all             # the full analysis of the article, 20–30 min on a 12-core laptop, < 1 GB RAM per process
python -m poroseg.run --stages spatial,inference,report   # selected stages (they read earlier outputs from disk)
python -m poroseg.run --stages descriptive --samples DS23_CF5,AG24_VN04    # a subset of images
```

## Input data

The micro-CT slices are not stored in this repository. Download `poroseg-data-v1.0.0.zip` from the data
record of the article ([DATA-DOI]) and extract it into the folder that **contains** this code folder:

```
any-folder/
  poroseg/                                   <- this repository
  imagens/                                   <- the nine 16-bit micro-CT slices (from the data archive)
  results/analysis_20250723_145220/          <- stored results of the original code, read by the `legacy` stage
```

These are the default paths of `config.yaml` (`paths.images_dir`, `paths.legacy_results_dir`); edit them
if you keep the data elsewhere. Sample metadata (`data/samples.csv`) and laboratory porosity
(`data/lab_porosity.csv`) are already in this repository and are the inputs the pipeline reads;
`scripts/build_lab_table.py` only documents how `lab_porosity.csv` was derived from the laboratory
spreadsheet (`paths.lab_xlsx`, not distributed), and the pipeline does not need that file.

The parameters of the segmentation, normalisation, post-processing, statistics, spatial analysis,
sensitivity, perturbations, synthetic benchmark and figures, and the master seed, are in `config.yaml`;
change them and run again. The random-forest settings of the machine-learning baselines are set in
`poroseg/stages/ml_baseline.py`, and the reduced replicate counts of `--quick` in the stage modules and
`poroseg/context.py`.

## Outputs (`outputs/`)

| Path | Content |
|---|---|
| `figures/` | every figure as vector PDF and 600-dpi PNG |
| `tables/` | every table as CSV, LaTeX and Markdown; `tables_all.xlsx` |
| `data/` | tidy full-precision data behind each figure/table |
| `key_numbers.json`, `RESULTS_DIGEST.md` | every number quoted in the article, with description and producing stage |
| `FIGURES_AND_TABLES.md` | catalogue of figures/tables with draft captions |
| `LEGACY_AUDIT.md` | audit of the original code: each defect, where it was and how much it changed the results |
| `logs/provenance.json` | package versions, SHA-256 of the inputs, configuration and seed |

## Code layout

```
config.yaml            all parameters
data/samples.csv       sample metadata (group, holder, pixel size, optional manual ROI)
data/lab_porosity.csv  laboratory porosity (long format)
poroseg/
  io.py, roi.py        TIFF reading, automatic detection of the usable disc ROI
  thresholds.py        the eight algorithms on ROI-normalised images (masked local statistics)
  postprocess.py       post-processing of the pore phase
  stats.py             statistics library (repeated measures, exact tests, variance decomposition, agreement)
  spatial_tools.py     two-point statistics, integral range, tiling, block bootstrap
  morpho_tools.py      Minkowski functionals, edge-corrected object statistics, granulometry, grey-scale porosity
  synthetic.py         phantoms with known truth and the micro-CT degradation model
  context.py, style.py shared run context, writers and figure style
  stages/              one module per analysis (see table above)
tests/                 pytest suite
```

## Reproducibility

All random numbers derive from the master seed in `config.yaml` through per-purpose streams, the
inputs are checksummed in `outputs/logs/provenance.json`, and the article's text reads its numbers
from `outputs/key_numbers.json`.

## Citation and licence

Please cite the article and the version of the code you used, archived on Zenodo (its DOI is in
`CITATION.cff` and on the Zenodo record linked from this repository). Code released under the MIT
licence (`LICENSE`); the data record ([DATA-DOI]) is released under CC BY 4.0.
