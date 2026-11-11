# poroseg user guide

`poroseg` measures how much of a porosity computed from 2-D X-ray micro-CT slices depends on the analyst's
choices (thresholding algorithm, post-processing and region of interest), how uncertain a slice porosity is
because of the spatial structure of the pore phase, and how segmentations compare with synthetic images of
known truth and with laboratory porosity. This guide describes the inputs, the options, the outputs and the
behaviour you should expect. Step-by-step examples are in [`docs/tutorials/`](tutorials/).

Contents: [1 Requirements](#1-requirements-and-installation) ·
[2 Repository layout](#2-repository-layout) · [3 Inputs](#3-inputs) · [4 Configuration](#4-configuration-configyaml) ·
[5 Running the pipeline](#5-running-the-pipeline) · [6 Stages](#6-stages) · [7 Outputs](#7-outputs) ·
[8 Expected behaviour](#8-expected-behaviour) · [9 Python API](#9-python-api) ·
[10 Troubleshooting](#10-troubleshooting)

## 1 Requirements and installation

| Item | Requirement |
|---|---|
| Python | 3.10 or later (developed and tested with 3.12 and 3.14) |
| Operating system | Windows, Linux or macOS (pure Python; no compiled extensions of its own) |
| Packages | NumPy, SciPy, pandas, scikit-image, scikit-learn, Matplotlib, statsmodels, pingouin, tifffile, openpyxl, PyYAML, Jinja2, pytest (minimum versions in `requirements.txt`) |
| Memory | below 1 GB per process for the slices of the article (up to 1800 × 1800 px) |
| Disk | about 45 MB for the repository with its data; about 250 MB for a full `outputs/` folder |
| Time | full analysis of the article 20–30 min on a 12-core laptop; `--quick` about 15 min; tests under 1 min |

```bash
git clone https://github.com/DOMFIRMUS/poroseg.git
cd poroseg
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Every command in this guide is run from the repository root (the folder that contains `config.yaml`). The
package is used from the source tree; no installation step other than the requirements is needed.

## 2 Repository layout

```
config.yaml                 every parameter of the analysis (section 4)
data/images/                the nine 16-bit micro-CT slices analysed in the article
data/samples.csv            sample metadata (section 3.2)
data/lab_porosity.csv       laboratory porosity, long format (section 3.3)
data/legacy_results/        porosities stored by the original analysis code (read by the 'legacy' stage)
poroseg/                    the package: library modules and one module per analysis stage (poroseg/stages/)
examples/                   runnable examples and a synthetic test dataset with known truth (examples/data/)
docs/                       this guide and the tutorials
tests/                      automated tests (pytest)
scripts/build_lab_table.py  how data/lab_porosity.csv was derived from the laboratory spreadsheet
```

## 3 Inputs

### 3.1 Images

* One 2-D grey-scale slice per specimen, as a single-page TIFF of any bit depth (8- or 16-bit integer, or
  float). Multi-page TIFF stacks are rejected: the pipeline analyses slices.
* Pores must be darker than the solid (low X-ray attenuation). Images with the opposite convention must be
  inverted first.
* The specimen is a cylinder seen in cross-section: either a free-standing core surrounded by air
  (`holder = free`) or loose material inside a cylindrical holder (`holder = container`). The usable disc is
  detected automatically (RANSAC circle fit; see Section S1 of the article's supplementary material) or
  given explicitly in `samples.csv`.
* The secondary region of interest `center500` is a 500 × 500 px window at the image centre; it needs images
  of at least 500 px on each side (remove it from `roi.designs` otherwise).

### 3.2 `samples.csv`

One row per slice.

| Column | Required | Meaning |
|---|---|---|
| `sample_id` | yes | unique identifier used in every output table |
| `label` | yes | short name used in figures |
| `file` | yes | file name inside `paths.images_dir` |
| `group` | yes | free text used to group results; the article uses `granular_pack` and `consolidated` |
| `holder` | yes | `free` (core in air) or `container` (material in a cylindrical holder) |
| `pixel_size_um` | yes | pixel size in µm; every length, area and density is scaled with it |
| `grain_size_mm` | no | nominal grain size (metadata only) |
| `pixel_size_source` | no | how the pixel size was established (metadata only) |
| `roi_cx`, `roi_cy`, `roi_r` | no | centre and radius (px) of the disc ROI; when all three are given, automatic detection is skipped |
| `notes` | no | free text |

### 3.3 `lab_porosity.csv`

Long format, one row per specimen and laboratory method: `sample_id`, `specimen`, `method`, `porosity_pct`,
and optionally `has_image` and `source`. Images without a laboratory value are allowed (TQ06 and GT01 in the
article have none); the comparison with the laboratory uses the images that have one. Specimens without an
image (`has_image = False`) enter only the comparison between laboratory methods. The reference porosity of a
specimen is the mean of the methods listed in `lab.reference_methods`; if none of them is present, the mean of
its other values except `lab.uct_method` is used (a single-value reference, as for the limestone cores of the
article). The laboratory's own micro-CT value (`uct`) is reported but never used as the reference. The
synthetic test dataset (`examples/data/lab_porosity.csv`) shows the minimal form.

## 4 Configuration (`config.yaml`)

Paths are relative to the configuration file. Change a value and run again; a different file can be passed
with `--config`.

| Section / key | Default | Meaning |
|---|---|---|
| `project.seed` | 20250923 | master seed; every random stream is derived from it and from the purpose of the stream |
| `paths.images_dir`, `samples_csv`, `lab_csv` | `data/...` | inputs (section 3) |
| `paths.legacy_results_dir` | `data/legacy_results/...` | stored results of the original code, read only by the `legacy` stage |
| `paths.output_dir` | `outputs` | where everything is written (`--quick` writes to `<output_dir>_quick`) |
| `roi.primary` | `disk` | ROI used by every inferential analysis |
| `roi.designs` | `[disk, center500]` | ROI designs crossed in the factorial design |
| `roi.free_margin_frac` | 0.03 | free-standing cores: the disc radius is 97 % of the fitted boundary |
| `roi.container_margin_px` | 6 | holders: the disc ends 6 px inside the inner wall |
| `roi.eval_border_px` | 2 | rim excluded from every metric |
| `normalization.p_low`, `p_high` | 0.1, 99.9 | percentiles inside the ROI mapped to 0 and 1 |
| `thresholds.methods` | eight algorithms | Otsu, Li, Yen, Mean, Triangle, Isodata (global), Sauvola, Niblack (local) |
| `thresholds.nbins` | 256 | histogram bins of the global methods |
| `thresholds.sauvola`, `niblack` | w = 25, k = 0.2, R = 0.5; w = 25, k = 0.8 | window (px, odd) and parameters of the local methods |
| `postprocessing.levels` | none, median, open_close, skeleton | levels applied to the pore phase; `skeleton` is a topological descriptor, never a porosity |
| `postprocessing.median_size`, `morph_se`, `morph_size` | 3, square, 3 | filter and structuring element sizes |
| `statistics.n_boot`, `ci_level`, `p_adjust` | 10 000, 0.95, holm | bootstrap replicates, interval level, multiplicity adjustment |
| `lab.reference_methods`, `uct_method` | direct, gravimetric, indirect, gas; uct | laboratory reference (section 3.3) |
| `spatial.tile_sizes_px` | 16–384 | tile sides for the variance-scaling analysis and the block bootstrap |
| `spatial.block_tile_px` | 64 | smallest block side; the block is also at least three correlation lengths |
| `spatial.target_abs_error_pp` | 1.0 | half-width that defines the representative window L_REV |
| `spatial.n_boot_pixels`, `radial_bin_px` | 1000, 10 | pixel bootstrap replicates; radial-profile bin |
| `sensitivity.psi_delta`, `tsm_deltas` | 0.05; −0.20…0.20 | threshold shifts for the Pore Stability Index and the Threshold Sensitivity Matrix |
| `sensitivity.sweep_*` | windows 15–201 px, k values | parameter sweep of the local methods |
| `perturbation.noise_sd`, `cupping`, `blur_sigma`, `n_rep` | see file | controlled artefacts applied to the real slices |
| `synthetic.*` | 512 px, 5 replicates, 3 phantoms, 3 blur × 3 noise × 2 cupping | synthetic benchmark design |
| `figures.*` | PDF and 600-dpi PNG | figure formats and sizes |

The random-forest settings of the machine-learning baselines are constants at the top of
`poroseg/stages/ml_baseline.py`; the reduced replicate counts of `--quick` are in the stage modules and in
`poroseg/context.py`.

## 5 Running the pipeline

```bash
python -m poroseg.run --stages all                         # the full analysis of the article
python -m poroseg.run --stages all --quick                 # fewer replicates, written to outputs_quick/
python -m poroseg.run --stages segmentation,descriptive    # selected stages
python -m poroseg.run --stages spatial --samples DS23_CF5,AG24_VN04
python -m poroseg.run --config examples/config_example.yaml --quick --stages segmentation,spatial
```

| Option | Meaning |
|---|---|
| `--stages` | comma-separated list of stages (section 6), or `all` |
| `--quick` | reduced bootstrap replicates and repetitions, for a fast check; results go to `<output_dir>_quick` |
| `--samples` | subset of `sample_id`; stages that compare images need at least three |
| `--config` | another configuration file |
| `--keep-going` | continue with the next stage after a failure (the failure is logged and reported in the status line) |

Stages read the outputs of earlier stages from disk, so a later stage can be run again on its own. The
binary segmentations and the detected ROI geometry are cached in `outputs/cache/` together with a signature of
the settings that produced them; a change in `roi`, `normalization`, `thresholds` or `postprocessing`
invalidates the cache automatically.

## 6 Stages

| Stage | What it does | Needs | Time (article) |
|---|---|---|---|
| `segmentation` | factorial design: every image × algorithm × post-processing level × ROI design; ROI detection figure; dataset and parameter tables | images | 0.5–1 min |
| `descriptive` | histograms and thresholds, bimodality, segmentation galleries, pixel agreement between algorithms, effect of the ROI design | segmentation | 4–6 min |
| `spatial` | two-point statistics, correlation length, integral range, L_REV, block bootstrap, integral-range and pixel-bootstrap uncertainty, radial profiles | segmentation | 2 min |
| `sensitivity` | porosity–threshold curves, Threshold Sensitivity Matrix, Pore Stability Index, sweep of the local methods, response of the local methods to homogeneous fields | segmentation | 1.5 min |
| `perturbation` | robustness to controlled noise, cupping and blur applied to the real slices | segmentation | 4–6 min |
| `morphology` | effect of post-processing on boundary length, Euler number, number density and pore-size distribution; grey-scale (partial-volume) porosity; unresolved porosity | segmentation | 8–10 min |
| `synthetic` | benchmark on 270 synthetic images with known truth | nothing (images are generated) | 4–5 min |
| `ml_baseline` | random forests and Gaussian mixture as trainable baselines | segmentation, synthetic phantoms | 5–6 min |
| `inference` | Friedman / repeated-measures tests, exact variance decomposition, agreement with the laboratory | segmentation, spatial, lab table | 2–4 min |
| `legacy` | re-implementation and audit of the original analysis code of the article | stored legacy results | 10 s |
| `report` | key numbers, catalogue of figures and tables, results digest, workbook with every table | all earlier stages | 5 s |

## 7 Outputs

Everything is written under `paths.output_dir` (default `outputs/`).

| Path | Content |
|---|---|
| `figures/<name>.pdf`, `.png` | every figure (vector PDF and 600-dpi PNG) |
| `tables/<name>.csv`, `.tex`, `.md` | every table |
| `data/<name>.csv` | tidy full-precision data behind each figure and table |
| `tables_all.xlsx` | every table in one workbook |
| `key_numbers.json` | every number quoted in the article: value, description and producing stage |
| `RESULTS_DIGEST.md` | the key numbers in readable form |
| `FIGURES_AND_TABLES.md` | catalogue of figures and tables with their captions |
| `LEGACY_AUDIT.md` | defects of the original code and their effect |
| `catalog/<stage>.json`, `keys/<stage>.json` | per-stage catalogue and key numbers |
| `qa/` | quality-control images (for example ROI detection) |
| `logs/run.log`, `logs/provenance.json` | log of the run; package versions, SHA-256 of every input, configuration, seed and status of every stage |

The most used data files are:

| File | One row per | Main columns |
|---|---|---|
| `data/segmentation_factorial.csv` | image × ROI × algorithm × post-processing | `porosity` (fraction), `metric` (porosity or skeleton density), `n_eval_px`, `area_mm2` |
| `data/spatial_summary.csv` | image × ROI × algorithm × post-processing | `lc_um`, `A_mm2`, `ir_converged`, `L_rev_ir_mm`, `L_rev_tiles_mm`, `hw_block_pp`, `hw_ir_pp`, `ci_width_ratio` (block / pixel interval) |
| `data/psi.csv` | image × algorithm | `phi`, `t`, `psi`, `phi_plus`, `phi_minus`, `hist_density` |
| `data/lab_agreement.csv` | image × algorithm × post-processing | image and laboratory porosity and their difference |
| `tables/variance_components.csv` | term of the factorial model | share of the total sum of squares (porosity and logit scales) with cluster-bootstrap intervals |

## 8 Expected behaviour

* With the same inputs, configuration and package versions, two runs give identical results, because every
  random number comes from a stream derived from `project.seed` and the purpose of the stream. `--quick` uses
  fewer replicates, so its intervals differ slightly from those of the full run.
* Segmentations with porosity below 1 % or above 90 % are marked as failures in the tables and kept in every
  analysis.
* When the normalised covariance never crosses zero within the largest lag (large-scale trends such as
  cupping or a wall effect), the integral range is flagged as not converged (`ir_converged = False`) and the
  covariance-based L_REV is a lower bound.
* The block side of the block bootstrap is the smallest configured tile side of at least
  max(`spatial.block_tile_px`, 3 l_c) that still yields 20 complete tiles; `block_rule` in
  `spatial_summary.csv` says when this was not possible.
* The local methods use masked local statistics, so pixels outside the ROI never bias the threshold map.
* The comparison with the laboratory uses only the images that have a laboratory value, and no other analysis
  uses the laboratory values.
* Tests and bootstrap intervals over images need at least three images. With n images, the smallest exact
  p-value a two-sided signed-rank test can reach is 2 / 2^n.
* Warnings, for example a non-converged integral range or a ROI too small for 20 blocks, go to
  `logs/run.log`. A failing stage stops the run unless `--keep-going` is given.

## 9 Python API

The stages are built from small modules that can be used directly; the examples in `examples/` show how.

| Module | Main functions |
|---|---|
| `poroseg.io` | `read_tiff`, `load_samples`, `load_lab`, `lab_reference` |
| `poroseg.roi` | `detect_geometry`, `disk_roi`, `center_square_roi`, `build_rois` |
| `poroseg.thresholds` | `normalize`, `compute_threshold`, `default_params`, `segment`, `sauvola_map`, `niblack_map` |
| `poroseg.postprocess` | `apply` (none, median, open_close, skeleton) |
| `poroseg.metrics` | `porosity`, `dice`, `jaccard` |
| `poroseg.spatial_tools` | `MaskedCorrelator`, `summarize_covariance`, `tile_grid`, `tile_fractions`, `l_rev_theory`, `l_rev_empirical`, `choose_block_size`, `block_bootstrap_mean`, `iid_bootstrap_binomial`, `se_integral_range` |
| `poroseg.morpho_tools` | `crofton_boundary_density`, `euler_quads`, `object_table`, `granulometry`, `phi_pv` |
| `poroseg.synthetic` | `make_phantom`, `degrade`, `generate_case`, `segment_all`, `segmentation_metrics` |
| `poroseg.stats` | variance decomposition, repeated-measures tests, exact tests, agreement statistics |
| `poroseg.context.Context` | the object every stage uses: cached images, ROIs, thresholds and segmentations for a configuration |

## 10 Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `FileNotFoundError: run the 'segmentation' stage first` | a later stage was run alone before the factorial table existed: run `--stages segmentation` first |
| `multi-page TIFF (3-D stack) not supported` | export the slice you want as a single-page TIFF |
| the detected disc is wrong (see `figures/supp_roi_detection.png`) | give `roi_cx`, `roi_cy` and `roi_r` in `samples.csv` |
| `image ... smaller than the 500px ROI` | remove `center500` from `roi.designs` |
| `KeyError: unknown samples` | a `--samples` value is not a `sample_id` of `samples.csv` |
| results differ from the article | check `logs/provenance.json` (input checksums, package versions) and that `--quick` was not used |

For problems or questions open an issue on GitHub or write to the corresponding author
(jean.firmino@ufpe.br).
