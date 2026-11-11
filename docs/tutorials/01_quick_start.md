# Tutorial 1 — Quick start with a synthetic slice of known truth

**Goal:** install `poroseg`, check the installation and see, on an image whose true porosity is known, how much
the eight thresholding algorithms disagree. **Time:** 5 minutes. **Data:** none (the image is generated).

## 1. Install and test

```bash
git clone https://github.com/DOMFIRMUS/poroseg.git
cd poroseg
python -m pip install -r requirements.txt
python -m pytest -q tests
```

The test suite (106 tests, under a minute) checks the statistical functions against independent
implementations (SciPy, pingouin) and the estimators against synthetic data with known answers. It needs no
input data.

## 2. Segment a synthetic slice

```bash
python examples/01_quick_start_synthetic.py
```

The script draws a planar packing of grain sections (the analogue of the granular packs of the article) on a
grid four times finer than the image, area-averages it to create partial-volume pixels, applies the micro-CT
degradation model (Gaussian blur of 1.5 px, noise of 5 % of the phase contrast, 15 % cupping), normalises the
image as the pipeline normalises a real slice and segments it with the eight algorithms. Expected output:

```
phantom: grain_pack; blur 1.5 px, noise 0.05, cupping 0.15
true porosity: 36.5 % on the pixel grid, 36.3 % in total
  method  porosity_pct  error_pp  dice
    Otsu         37.07      0.55  0.98
      Li         34.77     -1.75  0.97
     Yen         43.12      6.61  0.92
    Mean         39.24      2.73  0.96
Triangle         39.39      2.87  0.96
 Isodata         37.07      0.55  0.98
 Sauvola         32.60     -3.91  0.93
 Niblack         16.33    -20.18  0.51
smallest absolute error without post-processing: Otsu (+0.55 pp)
```

`examples/outputs/quick_start_synthetic.csv` also holds the median-filtered and opened–closed versions, and
`quick_start_synthetic.png` shows the image, the truth and two segmentations.

## 3. Try other conditions

```bash
python examples/01_quick_start_synthetic.py --kind boolean_pores --blur 3 --noise 0.10
python examples/01_quick_start_synthetic.py --kind multiscale
```

* `boolean_pores`: isolated pores with log-normal radii (porosity 0.11–0.19).
* `multiscale`: resolved pores plus micropores of 0.5–1.5 px. Every algorithm except Mean then underestimates
  the total porosity, because part of it lies below the resolution (`error_pp` is against the pixel grid; the
  total porosity is printed on the second line).

## What to take from it

The same image gives porosities that differ by tens of percentage points depending on the algorithm. With
known truth you can tell which one is right; with a real image you cannot, which is why the pipeline
quantifies the spread (Tutorial 4) and validates algorithms on synthetic images (stage `synthetic`).

Next: [Tutorial 2 — segment a real slice](02_segment_a_slice.md).
