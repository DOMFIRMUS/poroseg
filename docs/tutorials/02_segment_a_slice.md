# Tutorial 2: Segment a real slice and measure threshold stability

**Goal:** apply the factorial design of the article (eight algorithms × three area-preserving post-processing
levels) to one micro-CT slice, and compute the Pore Stability Index of each algorithm. **Time:** under a
minute per slice. **Data:** any slice in `data/images/` or your own 2-D TIFF.

## 1. A slice of the article

```bash
python examples/02_segment_one_slice.py --sample DS23_CF5
```

The script reads the 16-bit slice, detects the usable disc inside the holder (RANSAC circle fit to the inner
wall, ending 6 px inside it), clips the grey levels to the 0.1–99.9 percentiles of the disc and rescales them to
[0, 1], applies every algorithm and post-processing level, and reports the porosity inside the disc (a 2-px rim
excluded). Expected output (identical to the article's factorial table for CF5):

```
DS23_CF5_3mm_25g0280.tif: 1323 x 1324 px, uint16; disc ROI centre (678, 658) px, radius 515 px (auto)
  method family  threshold   psi  porosity_none_pct  porosity_median_pct  porosity_open_close_pct
    Otsu global      0.498 0.983             29.744               29.758                   29.726
      Li global      0.415 0.980             28.850               28.860                   28.808
     Yen global      0.725 0.811             35.221               33.595                   32.637
    Mean global      0.627 0.975             31.092               31.065                   31.033
Triangle global      0.670 0.942             31.905               31.670                   31.584
 Isodata global      0.498 0.983             29.744               29.758                   29.726
 Sauvola  local      0.531 0.671             20.368               20.028                   18.261
 Niblack  local      0.523 0.000             16.649               11.880                    5.013
range across the eight algorithms (no post-processing): 18.6 percentage points
```

`threshold` is in normalised grey level (for the local methods, the mean of the threshold map).
`psi` is the Pore Stability Index, `1 − min(1, |φ(T+δ) − φ(T−δ)| / (2 φ(T)))` with δ = 0.05: 1 means that
shifting the threshold by ±5 % of the grey range hardly changes the porosity. The local methods have the
lowest PSI here. A high PSI does not mean an accurate porosity: Otsu, the most stable algorithm, still differs
from the laboratory porosity of this pack (39.5 %) by about 10 points.

The histogram with the eight thresholds is written to `examples/outputs/segment_DS23_CF5_histogram.png`.

## 2. A core with a unimodal histogram

```bash
python examples/02_segment_one_slice.py --sample AG24_VN02
```

On the limestone cores the histogram has a single peak. The thresholds of the eight algorithms spread across
it, and the porosities they give differ by tens of percentage points.

## 3. Your own slice

```bash
python examples/02_segment_one_slice.py --image path/to/slice.tif --holder free
```

Use `--holder free` for a free-standing core surrounded by air and `--holder container` for loose material in
a cylindrical holder. Pores must be darker than the solid. If the detected disc is wrong (the printed centre
and radius are not plausible), describe the slice in a `samples.csv` with `roi_cx`, `roi_cy` and `roi_r`
(see the [user guide](../USER_GUIDE.md#32-samplescsv)) and use the full pipeline (Tutorial 4).

## Under the hood

The example uses the library functions directly, so it can be copied into your own scripts:

```python
from poroseg import io as pio, roi as proi, thresholds, postprocess, metrics
from poroseg.config import load_config

cfg = load_config("config.yaml")
raw = pio.read_tiff("data/images/DS23_CF5_3mm_25g0280.tif")
geom = proi.detect_geometry(raw.data, "container", cfg, seed=cfg["project"]["seed"])
R = proi.disk_roi(raw.shape, geom, cfg["roi"]["eval_border_px"])
img, lo, hi = thresholds.normalize(raw.data[R.slices], R.mask, 0.1, 99.9)
T = thresholds.compute_threshold("Otsu", img, R.mask, thresholds.default_params("Otsu", cfg))
pore = thresholds.segment(img, T, R.mask)
print(metrics.porosity(postprocess.apply(pore, "median", cfg, R.mask), R.eval_mask))
```

Next: [Tutorial 3: Spatial uncertainty of one slice](03_spatial_uncertainty.md).
