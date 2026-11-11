# Examples

Run every example from the repository root. Outputs go to `examples/outputs/` (not tracked by git).

| Script | What it shows | Data | Time |
|---|---|---|---|
| `01_quick_start_synthetic.py` | the eight thresholds on a synthetic slice of known truth, scored against the truth | none (generated) | seconds |
| `02_segment_one_slice.py` | the factorial design (8 algorithms × 3 post-processing levels) and the Pore Stability Index on one slice; works on any 2-D TIFF with `--image` | `data/images/` or your own | seconds |
| `03_spatial_uncertainty.py` | correlation length, integral range, representative window, block bootstrap versus pixel bootstrap for one segmentation | `data/images/` | seconds |
| `make_example_dataset.py` | rebuilds the synthetic test dataset in `data/` (three slices with truth masks, metadata and reference porosity) | none (generated) | seconds |
| `config_example.yaml` | configuration that runs the whole pipeline on the synthetic test dataset | `examples/data/` | about 2 min with `--quick` |

```bash
python examples/01_quick_start_synthetic.py
python examples/02_segment_one_slice.py --sample DS23_CF5
python examples/03_spatial_uncertainty.py --sample DS23_CF5
python -m poroseg.run --config examples/config_example.yaml --quick --stages segmentation,descriptive,spatial,sensitivity,perturbation,morphology,inference,report
```

The tutorials in [`../docs/tutorials/`](../docs/tutorials/) explain each example and its expected output.
