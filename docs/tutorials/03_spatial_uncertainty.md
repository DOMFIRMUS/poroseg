# Tutorial 3: How uncertain is the porosity of one slice?

**Goal:** quantify the sampling uncertainty of a slice porosity with a spatial block bootstrap and the integral
range, estimate the representative window, and see why resampling pixels understates the uncertainty.
**Time:** about 10 s per segmentation. **Data:** the slices in `data/images/`.

## 1. Run it

```bash
python examples/03_spatial_uncertainty.py --sample DS23_CF5 --method Otsu
```

Expected output (the same numbers as the `spatial` stage of the article):

```
CF5, Otsu, post-processing 'none': porosity 29.74 % over 824754 pixels (515 mm2)
correlation length l_c = 702 um; integral range A = 1.746 mm2
window for +/-1 pp: L_REV = 118 mm (integral range), 72 mm (tiles, extrapolated); ROI diameter 25.7 mm
95 % half-width of the ROI porosity: block bootstrap 4.57 pp (68 tiles of 96 px, rule 'ok'), integral range 5.21 pp, pixel bootstrap 0.100 pp
the pixel bootstrap interval is 46 times narrower than the block bootstrap interval
```

## 2. Read the output

The second line comes from the two-point statistics. The normalised covariance of the pore phase,
ρ(h) = (S₂(h) − φ²)/(φ − φ²), is computed inside the disc with a mask-corrected FFT. The correlation length
`l_c` is where ρ falls below 1/e, and the integral range `A` is the integral of ρ up to its first zero
crossing. Here the pores are correlated over about 0.7 mm, roughly a quarter of the 3-mm grain diameter.

The third line gives the representative window. A square window of side L has a porosity variance of about
φ(1 − φ)A/L², so a single window estimates φ within ±1 percentage point (95 %) only when L ≥ L_REV. Both
estimates, from `A` and from the measured standard deviation of tile porosity, are several times larger than
the 25.7-mm slice, so one slice of this pack is not representative at the ±1-point level.

The last two lines compare three uncertainties of the ROI porosity. The block bootstrap resamples tiles at least
three correlation lengths wide (here 96 px = 2.4 mm), and the integral range gives an analytical standard
error. The pixel bootstrap treats the 824 754 pixels as independent replicates and gives an interval 46 times
too narrow.

If the covariance never crosses zero (a large-scale trend such as cupping, common in the cores), the script
says that the integral range did not converge, and the integral-range numbers are lower bounds.

## 3. Variations

```bash
python examples/03_spatial_uncertainty.py --sample AG24_VN04 --method Sauvola --post median
python examples/03_spatial_uncertainty.py --sample DS23_CF10 --method Li --post open_close
```

The tile statistics are written to `examples/outputs/spatial_<sample>_<method>_<post>.csv`; the `spatial`
stage of the pipeline does this for every image, algorithm, post-processing level and ROI and adds radial
porosity profiles (wall and cupping effects).

Next: [Tutorial 4: The full pipeline on your own slices](04_pipeline_on_your_data.md).
