"""Input/output of raw data: images, sample metadata and laboratory values."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


@dataclass
class RawImage:
    sample_id: str
    path: Path
    data: np.ndarray                 # float32, original grey values (not rescaled)
    dtype: str                       # original dtype, e.g. 'uint16'
    shape: tuple[int, int]
    pixel_size_um_tag: float | None  # inferred from the TIFF resolution tag (None if absent/default)
    meta: dict = field(default_factory=dict)


def _tag_pixel_size_um(page) -> float | None:
    """Pixel size from the TIFF XResolution tag, if it looks physical.

    Many reconstruction packages store the voxel size as a resolution in
    pixels per inch (ResolutionUnit = 2). 72/96/300 dpi are software defaults
    and carry no physical meaning, so they are ignored.
    """
    tags = page.tags
    if "XResolution" not in tags:
        return None
    num, den = tags["XResolution"].value
    if not den:
        return None
    dpi = num / den
    unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 2
    unit = int(getattr(unit, "value", unit))
    if unit != 2 or round(dpi) in (72, 96, 150, 300, 600) or dpi <= 0:
        return None
    return 25400.0 / dpi


def read_tiff(path: str | Path, sample_id: str = "") -> RawImage:
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        if len(tf.pages) > 1:
            raise ValueError(f"{path.name}: multi-page TIFF (3-D stack) not supported by this 2-D pipeline")
        px = _tag_pixel_size_um(page)
        meta = dict(n_pages=len(tf.pages), bits=int(page.bitspersample), compression=str(page.compression))
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2-D grey-scale image, got shape {arr.shape}")
    return RawImage(sample_id=sample_id, path=path, data=arr.astype(np.float32), dtype=str(arr.dtype),
                    shape=arr.shape, pixel_size_um_tag=px, meta=meta)


def filename_pixel_size_um(name: str) -> float | None:
    """Voxel size encoded as a token '_NN_' in the file name (e.g. 'BC23_TQ-06_40_0114')."""
    m = re.search(r"_(\d{2})_\d{4}\.tif$", name, flags=re.I)
    return float(m.group(1)) if m else None


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_samples(csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype={"grain_size_mm": str})
    need = {"sample_id", "label", "file", "group", "holder", "pixel_size_um"}
    if not need.issubset(df.columns):
        raise KeyError(f"{csv}: missing columns {need - set(df.columns)}")
    if df["sample_id"].duplicated().any():
        raise ValueError("duplicated sample_id in samples.csv")
    bad = set(df["holder"]) - {"free", "container"}
    if bad:
        raise ValueError(f"holder must be 'free' or 'container', got {bad}")
    return df.set_index("sample_id", drop=False)


def load_lab(csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    need = {"sample_id", "specimen", "method", "porosity_pct"}
    if not need.issubset(df.columns):
        raise KeyError(f"{csv}: missing columns {need - set(df.columns)}")
    return df


def lab_reference(lab: pd.DataFrame, reference_methods: list[str], uct_method: str = "uct") -> pd.DataFrame:
    """One row per specimen with the laboratory reference porosity (fractions, not %).

    Columns
    -------
    ref          mean of the available reference methods (or the single 'unspecified' value)
    ref_sd       SD across reference methods (NaN when only one method)
    ref_n        number of reference methods
    ref_se       ref_sd / sqrt(ref_n)
    ref_kind     'multi-method' | 'single-value'
    uct          laboratory uCT porosity if available
    mean_all5    mean of the five columns (the value used by the legacy code, includes uCT)
    """
    out = []
    for (sid, spec), g in lab.groupby(["sample_id", "specimen"], sort=False):
        vals = g.set_index("method")["porosity_pct"] / 100.0
        refs = vals.reindex(reference_methods).dropna()
        if len(refs):
            kind = "multi-method"
        else:
            refs = vals.drop(labels=[uct_method], errors="ignore")
            kind = "single-value"
        n = len(refs)
        sd = float(refs.std(ddof=1)) if n > 1 else np.nan
        out.append(dict(sample_id=sid, specimen=spec, ref=float(refs.mean()), ref_sd=sd, ref_n=n,
                        ref_se=sd / np.sqrt(n) if n > 1 else np.nan, ref_kind=kind,
                        ref_methods=",".join(refs.index),
                        uct=float(vals.get(uct_method, np.nan)),
                        mean_all5=float(vals.mean()) if kind == "multi-method" else np.nan,
                        has_image=bool(g["has_image"].iloc[0]) if "has_image" in g else True))
    return pd.DataFrame(out)
