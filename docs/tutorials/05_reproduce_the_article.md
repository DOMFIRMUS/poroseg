# Tutorial 5: Reproduce the article

**Goal:** regenerate every number, table and figure of the article *"How much of the porosity measured on
micro-CT slices is the analyst's choice? A factorial uncertainty analysis with synthetic ground truth and
laboratory comparison"* from the nine raw slices. **Time:** 20–30 min on a 12-core laptop (about 15 min with
`--quick`). **Data:** `data/` (already in the repository).

## 1. Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m poroseg.run --stages all
```

The run ends with a status line in which every one of the eleven stages reports `ok`. Everything is written to
`outputs/` (about 250 MB); `outputs/logs/provenance.json` records the package versions, the SHA-256 of every
input, the configuration and the seed.

## 2. Where each result of the article comes from

Every number quoted in the article is read from `outputs/key_numbers.json` (value, description and producing
stage); `outputs/RESULTS_DIGEST.md` lists them in words. Figures are in `outputs/figures/` as PDF and PNG:

| Article | Output file | Stage |
|---|---|---|
| Figure 1 | `study_design` | descriptive |
| Figure 2 | `samples_overview` | descriptive |
| Figure 3 | `histograms_thresholds` | descriptive |
| Figure 4 | `segmentation_gallery` | descriptive |
| Figure 5 | `porosity_by_method` | inference |
| Figure 6 | `variance_decomposition` | inference |
| Figure 7 | `rev_tile_variability` | spatial |
| Figure 8 | `uncertainty_block_vs_iid` | spatial |
| Figure 9 | `psi_by_method` | sensitivity |
| Figure 10 | `synthetic_benchmark` | synthetic |
| Figure 11 | `postprocessing_effects` | morphology |
| Figure 12 | `unresolved_porosity` | morphology |
| Table 1 | `tables/dataset.csv` | segmentation |
| Tables S26–S30 | `tables/methods_parameters`, `rev_integral_range`, `lab_agreement`, `unresolved_porosity`, `variance_components` | segmentation, spatial, inference, morphology, inference |

The other supplementary figures and tables have the file names given in brackets in the supplementary material;
`outputs/FIGURES_AND_TABLES.md` lists all of them with their captions, and `outputs/tables_all.xlsx` has every
table in one workbook.

## 3. Checks

* `python examples/02_segment_one_slice.py --sample DS23_CF5` and
  `python examples/03_spatial_uncertainty.py --sample DS23_CF5` reproduce single rows of the factorial and
  spatial tables in seconds (Tutorials 2 and 3).
* With the same package versions the results are identical from run to run; with other versions, differences
  are at the level of floating-point rounding.
* The `legacy` stage re-implements the original analysis code of the conference posters and must report that
  it reproduced the 72 stored porosities exactly; its audit is in `outputs/LEGACY_AUDIT.md`.

## 4. Archived versions

Releases of the code and data are archived on Zenodo; https://doi.org/10.5281/zenodo.23748708 always resolves
to the latest version. Version 1.1.0 (https://doi.org/10.5281/zenodo.24122454) contains the laboratory
porosity of the three limestone cores used in the article (helium porosity of the whole plug, Vila Nova 2020).
Version 1.0.0 had the values of another plug numbering for these three cores, which changes only the comparison
of the cores with the laboratory.
