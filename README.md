# poroseg: segmentation-induced uncertainty of micro-CT porosity

Code and data for the article *"How much of the porosity measured on micro-CT slices is the analyst's choice?
A factorial uncertainty analysis with synthetic ground truth and laboratory comparison"* (Cardoso et al.,
submitted to *Computers & Geosciences*). One command regenerates every number, table and figure of the article
from the nine raw micro-CT slices, which are included in this repository.

Version 1.1.0 · MIT licence · Python ≥ 3.10 · archived at https://doi.org/10.5281/zenodo.23748708

## Documentation

| Document | Content |
|---|---|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | inputs, every configuration option, stages, outputs, expected behaviour, Python API, troubleshooting |
| [`docs/tutorials/01_quick_start.md`](docs/tutorials/01_quick_start.md) | install, test, and segment a synthetic slice of known truth (no data needed) |
| [`docs/tutorials/02_segment_a_slice.md`](docs/tutorials/02_segment_a_slice.md) | the eight algorithms and three post-processing levels on one real slice, with the Pore Stability Index |
| [`docs/tutorials/03_spatial_uncertainty.md`](docs/tutorials/03_spatial_uncertainty.md) | block bootstrap, integral range and representative window of a slice porosity |
| [`docs/tutorials/04_pipeline_on_your_data.md`](docs/tutorials/04_pipeline_on_your_data.md) | the full pipeline on a synthetic test dataset, and how to run it on your own slices |
| [`docs/tutorials/05_reproduce_the_article.md`](docs/tutorials/05_reproduce_the_article.md) | full reproduction of the article and where each figure and table comes from |

## What it does

`poroseg` segments 2-D micro-CT slices with eight thresholding algorithms (Otsu, Li, Yen, Mean, Triangle,
Isodata, Sauvola, Niblack) × four post-processing levels (none, median, opening–closing, skeleton) × two
region-of-interest designs, and then quantifies:

| Stage | What it computes |
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

## Installation and requirements

Python ≥ 3.10 (tested with Python 3.12 and 3.14 on Windows 11; the code is pure Python and should run on Linux
and macOS). Dependencies, with minimum versions, are in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

A standard computer is enough: the full analysis of the article takes 20–30 min on a 12-core laptop with
less than 1 GB of RAM per process, and writes about 250 MB to `outputs/`.

## Quick start

Run every command from the repository root (the folder that contains `config.yaml`).

```bash
python -m pytest -q tests                                   # 106 automated tests, no data needed (< 1 min)
python examples/01_quick_start_synthetic.py                 # eight thresholds on a synthetic slice of known truth (seconds)
python examples/02_segment_one_slice.py --sample DS23_CF5   # factorial design on one real slice
python examples/03_spatial_uncertainty.py --sample DS23_CF5 # block bootstrap vs pixel bootstrap for one slice
python -m poroseg.run --stages all                          # the full analysis of the article (20–30 min)
python -m poroseg.run --stages all --quick                  # fast smoke run with fewer replicates, written to outputs_quick/
python -m poroseg.run --config examples/config_example.yaml --quick --stages segmentation,descriptive,spatial,sensitivity,perturbation,morphology,inference,report
                                                            # the whole pipeline on the synthetic test dataset (about 2 min)
```

Command-line options (`python -m poroseg.run --help`): `--stages` (comma-separated list, or `all`), `--quick`
(fewer replicates, written to `<output_dir>_quick/`), `--samples` (comma-separated subset of `sample_id`),
`--config` (another configuration file) and `--keep-going` (continue after a failing stage). Stages read the
outputs of earlier stages from disk, so a later stage can be re-run alone.

## Data

| Path | Content |
|---|---|
| `data/images/` | the nine 16-bit micro-CT slices analysed in the article (pixel size 25 µm, or 40 µm for TQ06 and GT01) |
| `data/samples.csv` | sample metadata: group, holder, pixel size, optional manual ROI |
| `data/lab_porosity.csv` | laboratory porosity in long format (four physical methods and the laboratory's micro-CT for the packs; helium porosity of the whole plug for the three limestone cores) |
| `data/legacy_results/` | porosities stored by the original analysis code, read by the `legacy` stage |
| `examples/data/` | synthetic test dataset with known truth (three slices, truth masks, metadata) |

The data are released under CC BY 4.0 and archived, with the derived data of the article, at
https://doi.org/10.5281/zenodo.23748708. `scripts/build_lab_table.py` documents how `lab_porosity.csv` was
derived from the laboratory spreadsheet (not distributed); the pipeline does not need that file.

## Using your own slices

Describe the slices in a `samples.csv` (required columns: `sample_id`, `label`, `file`, `group`, `holder` =
`free` or `container`, `pixel_size_um`; optional `roi_cx`, `roi_cy`, `roi_r` to impose the disc ROI), list the
laboratory porosity in a `lab_porosity.csv`, copy `examples/config_example.yaml` and change its `paths`, then
run the stages you need. [Tutorial 4](docs/tutorials/04_pipeline_on_your_data.md) does this step by step with
the synthetic test dataset, and the [user guide](docs/USER_GUIDE.md) describes every option.

## Outputs (`outputs/`)

| Path | Content |
|---|---|
| `figures/` | every figure as vector PDF and 600-dpi PNG |
| `tables/` | every table as CSV, LaTeX and Markdown |
| `tables_all.xlsx` | every table in one workbook |
| `data/` | tidy full-precision data behind each figure/table |
| `key_numbers.json`, `RESULTS_DIGEST.md` | every number quoted in the article, with description and producing stage |
| `FIGURES_AND_TABLES.md` | catalogue of figures/tables with draft captions |
| `LEGACY_AUDIT.md` | audit of the original code: each defect, where it was and how much it changed the results |
| `logs/provenance.json` | package versions, SHA-256 of the inputs, configuration and seed |

## Code layout

```
config.yaml            all parameters
data/                  the nine slices, sample metadata, laboratory porosity, stored legacy results
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
examples/              runnable examples and the synthetic test dataset
docs/                  user guide and tutorials
tests/                 pytest suite
```

## Reproducibility

All random numbers derive from the master seed in `config.yaml`, through one stream per purpose. The
inputs are checksummed in `outputs/logs/provenance.json`, and the article takes its numbers from
`outputs/key_numbers.json`.

## Changes in version 1.1.0

* Laboratory porosity of the limestone cores VN01, VN02 and VN04: helium porosity of the whole plug from
  Vila Nova (2020); version 1.0.0 had taken these three values from a source that numbers the plugs
  differently. Only the comparison of the cores with the laboratory changes.
* The nine micro-CT slices and the stored results of the original code are now in the repository
  (`data/`), so the article can be reproduced without downloading the data archive.
* User guide, five tutorials, runnable examples and a synthetic test dataset with known truth.
* The `inference` stage skips the laboratory-internal comparison (laboratory micro-CT against physical
  methods) when fewer than three specimens have a laboratory micro-CT value, instead of failing.

## Citation and licence

Please cite the article and the version of the code you used, archived on Zenodo
(https://doi.org/10.5281/zenodo.23748708; `CITATION.cff` gives the metadata). Code released under the MIT
licence (`LICENSE`); data under CC BY 4.0. Questions and problems: GitHub issues or the corresponding author,
Jean Firmino Cardoso (jean.firmino@ufpe.br).
