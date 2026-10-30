"""poroseg - reproducible analysis of segmentation-induced uncertainty in
X-ray micro-CT porosity.

Package layout
--------------
config        load/validate config.yaml
io            TIFF reading, pixel-size inference, sample/lab tables, hashing
roi           sample/holder geometry detection and ROI definitions
thresholds    the eight thresholding algorithms (masked, on normalised images)
postprocess   binary post-processing of the pore phase
metrics       basic binary-image metrics (porosity, overlap)
style         figure style, palettes, saving helpers
context       shared run context (caches, output writers, key numbers)
stages/       one module per analysis stage, each exposing ``run(ctx)``
run           command-line entry point:  python -m poroseg.run --stages all
"""

__version__ = "1.0.0"
