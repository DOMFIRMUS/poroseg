"""Write the synthetic test dataset in ``examples/data/`` (three 16-bit slices with known truth).

Run from the repository root::

    python examples/make_example_dataset.py

The dataset is already in the repository; this script documents how it was made and rebuilds it
identically (fixed seed). Each slice is a 640 x 640 px image of a free-standing cylindrical specimen in air:
inside a disc of radius 300 px lies a phantom with known structure (a grain pack, Boolean pores, or
resolved plus sub-resolution pores), degraded with the forward model of ``poroseg.synthetic`` (partial
volume, Gaussian blur, cupping, noise) and stored as uint16, like a reconstructed micro-CT slice.

Files
-----
images/SYN_<kind>.tif        the slices (inputs of the pipeline)
truth/SYN_<kind>_truth.tif   pixel-grid truth (255 = pore), for checking segmentations
samples.csv                  sample metadata in the format of data/samples.csv (the disc ROI is given
                             explicitly through roi_cx, roi_cy, roi_r)
lab_porosity.csv             the true total porosity inside the ROI, as a stand-in for a laboratory value
                             (method 'gas', so that it is used as the reference)
../config_example.yaml       configuration that points the pipeline to this dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # run from a source checkout without installing the package

import numpy as np
import pandas as pd
import tifffile

from poroseg import synthetic

SIZE, RADIUS, PIXEL_UM, SEED = 640, 300, 25.0, 20251110
CASES = [  # sample_id, phantom kind, group, blur (px), noise (fraction of contrast), cupping
    ("SYN_PACK", "grain_pack", "granular_pack", 1.5, 0.05, 0.10),
    ("SYN_BOOL", "boolean_pores", "consolidated", 1.5, 0.05, 0.10),
    ("SYN_MULTI", "multiscale", "consolidated", 1.5, 0.08, 0.15),
]
GREY_SCALE = 40000.0                    # grey value of pure solid in the uint16 image


def make_slice(kind: str, blur: float, noise: float, cupping: float, seed: int):
    rng = synthetic.derive_rng(seed, "example", kind)
    ph = synthetic.make_phantom(kind, SIZE, rng)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    c = (SIZE - 1) / 2.0
    inside = (yy - c) ** 2 + (xx - c) ** 2 <= RADIUS ** 2
    frac = np.where(inside, ph["truth_frac"], 1.0)           # air outside the specimen has pore attenuation
    grey = synthetic.degrade(frac, blur, noise, cupping, rng, radius=RADIUS, center=(c, c))
    img = np.clip(grey / synthetic.MU_S * GREY_SCALE, 0, 65535).astype(np.uint16)
    truth = (ph["truth"] & inside).astype(np.uint8) * 255
    return img, truth, ph, inside


def main():
    data = ROOT / "examples" / "data"
    (data / "images").mkdir(parents=True, exist_ok=True)
    (data / "truth").mkdir(parents=True, exist_ok=True)
    samples, lab = [], []
    for sid, kind, group, blur, noise, cupping in CASES:
        img, truth, ph, inside = make_slice(kind, blur, noise, cupping, SEED)
        tifffile.imwrite(data / "images" / f"{sid}.tif", img)
        tifffile.imwrite(data / "truth" / f"{sid}_truth.tif", truth)
        phi_total = float(ph["truth_frac"][inside].mean())
        samples.append(dict(sample_id=sid, label=sid.replace("SYN_", "S-"), file=f"{sid}.tif", group=group,
                            holder="free", grain_size_mm="", pixel_size_um=PIXEL_UM,
                            pixel_size_source="synthetic", roi_cx=(SIZE - 1) / 2, roi_cy=(SIZE - 1) / 2,
                            roi_r=RADIUS - 6, notes=f"synthetic {kind}; blur {blur} px, noise {noise}, cupping {cupping}"))
        lab.append(dict(sample_id=sid, specimen=sid, method="gas", porosity_pct=round(100 * phi_total, 2),
                        has_image=True, source="true total porosity of the phantom inside the disc"))
        print(f"{sid}: {kind}, true porosity {100 * phi_total:.1f} % (pixel grid {100 * ph['phi_pixel']:.1f} %)")
    pd.DataFrame(samples).to_csv(data / "samples.csv", index=False)
    pd.DataFrame(lab).to_csv(data / "lab_porosity.csv", index=False)
    print("wrote", data)


if __name__ == "__main__":
    main()
