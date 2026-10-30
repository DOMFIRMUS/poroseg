"""Shared run context.

A :class:`Context` is created once per run and passed to every stage's
``run(ctx)``. It owns

* configuration, sample metadata and laboratory reference values;
* cached access to raw images, geometry, ROIs, normalised images, thresholds
  and binary pore images (``ctx.pore(sid, method, post)``);
* writers for figures, tables, tidy data, key numbers and the figure/table
  catalogue, so that every stage produces outputs in the same way.

Outputs (all under ``paths.output_dir``)::

    figures/   <name>.pdf + <name>.png          (main + supplementary figures)
    tables/    <name>.csv / .tex / .md          (formatted tables)
    data/      <name>.csv                       (tidy, full-precision results)
    qa/        quality-assurance images
    cache/     binary pore images, geometry
    catalog/   <stage>.json                     (figures/tables + draft captions)
    keys/      <stage>.json                     (key numbers quoted in the paper)
    logs/      run log, provenance
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import io as pio
from . import postprocess, roi as proi, style, thresholds
from .metrics import porosity as _porosity

LOG = logging.getLogger("poroseg")


@dataclass
class NormImage:
    img: np.ndarray    # bbox-shaped float32 in [0, 1]
    lo: float          # raw grey level mapped to 0
    hi: float          # raw grey level mapped to 1


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class Context:
    def __init__(self, cfg: dict, quick: bool = False, samples: list[str] | None = None,
                 stage: str = "adhoc"):
        self.cfg = cfg
        self.quick = quick
        self.stage = stage
        self.root = Path(cfg["_root"])
        self.out = Path(cfg["paths"]["output_dir"])
        self.dirs = {k: self.out / k for k in
                     ("figures", "tables", "data", "qa", "cache", "catalog", "keys", "logs")}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.seed = int(cfg["project"]["seed"])
        self.samples = pio.load_samples(cfg["paths"]["samples_csv"])
        if samples:
            unknown = set(samples) - set(self.samples.index)
            if unknown:
                raise KeyError(f"unknown samples {unknown}")
            self.samples = self.samples.loc[samples]
        self.lab = pio.load_lab(cfg["paths"]["lab_csv"])
        self.lab_ref = pio.lab_reference(self.lab, cfg["lab"]["reference_methods"], cfg["lab"]["uct_method"])
        self.methods: list[str] = list(cfg["thresholds"]["methods"])
        self.posts: list[str] = list(cfg["postprocessing"]["levels"])
        self.primary_roi: str = cfg["roi"]["primary"]
        self._raw: dict[str, pio.RawImage] = {}
        self._geom: dict[str, proi.Geometry] = {}
        self._rois: dict[tuple, proi.ROI] = {}
        self._norm: dict[tuple, NormImage] = {}
        self._thr: dict[tuple, thresholds.ThresholdResult] = {}
        self._pore: dict[tuple, np.ndarray] = {}
        self._pore_order: list[tuple] = []
        self._pore_max = 96
        self._seg: pd.DataFrame | None = None
        self._catalog: list[dict] = []
        self._keys: dict[str, dict] = {}
        style.apply_style(float(cfg["figures"].get("font_size", 7.5)))

    # ------------------------------------------------------------------ basics
    @property
    def sample_ids(self) -> list[str]:
        return list(self.samples.index)

    def label(self, sid: str) -> str:
        return str(self.samples.at[sid, "label"])

    def group(self, sid: str) -> str:
        return str(self.samples.at[sid, "group"])

    def rng(self, *tags: Any) -> np.random.Generator:
        """Independent, reproducible stream for a given purpose (tags)."""
        h = hashlib.sha256(("|".join(map(str, (self.seed,) + tags))).encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    @property
    def n_boot(self) -> int:
        return 500 if self.quick else int(self.cfg["statistics"]["n_boot"])

    def log(self, msg: str, *args) -> None:
        LOG.info(f"[{self.stage}] " + msg, *args)

    # ------------------------------------------------------------------ images
    def raw(self, sid: str) -> pio.RawImage:
        if sid not in self._raw:
            f = Path(self.cfg["paths"]["images_dir"]) / self.samples.at[sid, "file"]
            self._raw[sid] = pio.read_tiff(f, sid)
        return self._raw[sid]

    def pixel_size_um(self, sid: str) -> float:
        return float(self.samples.at[sid, "pixel_size_um"])

    def geometry(self, sid: str) -> proi.Geometry:
        if sid in self._geom:
            return self._geom[sid]
        cache = self.dirs["cache"] / f"geometry_{sid}.json"
        row = self.samples.loc[sid]
        override = {k: float(row[k]) if pd.notna(row.get(k)) and str(row.get(k)).strip() != "" else np.nan
                    for k in ("roi_cx", "roi_cy", "roi_r")}
        sig = hashlib.sha256(json.dumps(dict(override=override, roi=self.cfg["roi"]),
                                        sort_keys=True, default=_json_default).encode()).hexdigest()[:16]
        if cache.exists():
            d = json.loads(cache.read_text())
            if d.get("_sig") == sig:
                d.pop("_sig")
                g = proi.Geometry(**d)
                self._geom[sid] = g
                return g
        g = proi.detect_geometry(self.raw(sid).data, row["holder"], self.cfg, override, seed=self.seed)
        d = asdict(g)
        d["info"] = {k: v for k, v in g.info.items() if k != "edge_points"}
        d["_sig"] = sig
        cache.write_text(json.dumps(d, default=_json_default, indent=1))
        self._geom[sid] = g
        return g

    def roi(self, sid: str, name: str | None = None) -> proi.ROI:
        name = name or self.primary_roi
        key = (sid, name)
        if key not in self._rois:
            rois = proi.build_rois(self.raw(sid).shape, self.geometry(sid), self.cfg)
            for n, r in rois.items():
                self._rois[(sid, n)] = r
        return self._rois[key]

    def norm(self, sid: str, roi: str | None = None) -> NormImage:
        roi = roi or self.primary_roi
        key = (sid, roi)
        if key not in self._norm:
            R = self.roi(sid, roi)
            sub = self.raw(sid).data[R.slices]
            n = self.cfg["normalization"]
            img, lo, hi = thresholds.normalize(sub, R.mask, n["p_low"], n["p_high"])
            img[~R.mask] = 0.0
            self._norm[key] = NormImage(img, lo, hi)
        return self._norm[key]

    # --------------------------------------------------------- segmentation
    def threshold(self, sid: str, method: str, roi: str | None = None,
                  params: dict | None = None) -> thresholds.ThresholdResult:
        roi = roi or self.primary_roi
        p = thresholds.default_params(method, self.cfg)
        if params:
            p.update(params)
        key = (sid, roi, method, json.dumps(p, sort_keys=True))
        if key not in self._thr:
            R = self.roi(sid, roi)
            self._thr[key] = thresholds.compute_threshold(method, self.norm(sid, roi).img, R.mask, p)
        return self._thr[key]

    def pore(self, sid: str, method: str, post: str = "none", roi: str | None = None,
             params: dict | None = None) -> np.ndarray:
        """Binary pore image (bbox-shaped bool; False outside the ROI)."""
        roi = roi or self.primary_roi
        key = (sid, roi, method, post, json.dumps(params or {}, sort_keys=True))
        if key in self._pore:
            return self._pore[key]
        arr = None
        if not params:
            arr = self._load_cached_pore(sid, roi, method, post)
        if arr is None:
            R = self.roi(sid, roi)
            if post == "none":
                arr = thresholds.segment(self.norm(sid, roi).img, self.threshold(sid, method, roi, params), R.mask)
            else:
                base = self.pore(sid, method, "none", roi, params)
                arr = postprocess.apply(base, post, self.cfg, R.mask)
        self._pore[key] = arr
        self._pore_order.append(key)
        if len(self._pore_order) > self._pore_max:
            old = self._pore_order.pop(0)
            self._pore.pop(old, None)
        return arr

    def porosity(self, sid: str, method: str, post: str = "none", roi: str | None = None) -> float:
        R = self.roi(sid, roi)
        return _porosity(self.pore(sid, method, post, roi), R.eval_mask)

    def _cache_file(self, sid: str, roi: str) -> Path:
        return self.dirs["cache"] / f"pores_{sid}_{roi}.npz"

    def _load_cached_pore(self, sid, roi, method, post):
        f = self._cache_file(sid, roi)
        if not f.exists():
            return None
        key = f"{method}__{post}"
        with np.load(f) as z:
            if key not in z.files or "_shape" not in z.files or "_sig" not in z.files:
                return None
            if str(z["_sig"]) != self.segmentation_signature():
                return None
            shape = tuple(z["_shape"])
            return np.unpackbits(z[key], count=shape[0] * shape[1]).reshape(shape).astype(bool)

    def write_pore_cache(self, sid: str, roi: str) -> None:
        arrays = {}
        shape = None
        for m in self.methods:
            for p in self.posts:
                a = self.pore(sid, m, p, roi)
                shape = a.shape
                arrays[f"{m}__{p}"] = np.packbits(a.ravel())
        np.savez_compressed(self._cache_file(sid, roi), _shape=np.array(shape),
                            _sig=np.array(self.segmentation_signature()), **arrays)

    def segmentation_signature(self) -> str:
        """Hash of every config entry that changes the binary images."""
        keys = {k: self.cfg[k] for k in ("roi", "normalization", "thresholds", "postprocessing")}
        return hashlib.sha256(json.dumps(keys, sort_keys=True, default=_json_default).encode()).hexdigest()[:16]

    # ---------------------------------------------------- factorial results
    @property
    def seg(self) -> pd.DataFrame:
        """Tidy factorial table produced by the 'segmentation' stage."""
        if self._seg is None:
            f = self.dirs["data"] / "segmentation_factorial.csv"
            if not f.exists():
                raise FileNotFoundError("run the 'segmentation' stage first (python -m poroseg.run --stages segmentation)")
            self._seg = pd.read_csv(f)
        return self._seg

    def seg_primary(self, area_preserving_only: bool = True) -> pd.DataFrame:
        d = self.seg[self.seg["roi"] == self.primary_roi]
        if area_preserving_only:
            d = d[d["post"].isin(postprocess.AREA_PRESERVING)]
        return d.copy()

    def read_data(self, name: str) -> pd.DataFrame:
        f = self.dirs["data"] / f"{name}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found - run the stage that produces it first")
        return pd.read_csv(f)

    # ------------------------------------------------------------- writers
    def save_fig(self, fig, name: str, caption: str, kind: str = "main") -> list[Path]:
        fmts = self.cfg["figures"]["formats"]
        paths = style.save(fig, self.dirs["figures"] / name, fmts, int(self.cfg["figures"]["dpi"]))
        self._catalog.append(dict(type="figure", name=name, kind=kind, caption=caption, stage=self.stage,
                                  files=[p.name for p in paths]))
        self.log("figure %s", name)
        return paths

    def save_qa(self, fig, name: str) -> Path:
        p = self.dirs["qa"] / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return p

    def save_data(self, df: pd.DataFrame, name: str) -> Path:
        p = self.dirs["data"] / f"{name}.csv"
        df.to_csv(p, index=False)
        return p

    def save_table(self, df: pd.DataFrame, name: str, caption: str, kind: str = "main",
                   index: bool = False, float_format: str = "{:.2f}", notes: str = "") -> list[Path]:
        """Save a formatted table as CSV (as given), LaTeX (booktabs) and Markdown."""
        paths = []
        p = self.dirs["tables"] / f"{name}.csv"
        df.to_csv(p, index=index)
        paths.append(p)
        shown = df.reset_index() if index else df.copy()
        fmt = shown.copy()
        for c in fmt.columns:
            if pd.api.types.is_float_dtype(fmt[c]):
                fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else float_format.format(v))
            else:
                fmt[c] = fmt[c].astype(str).replace({"nan": "", "None": ""})
        p = self.dirs["tables"] / f"{name}.tex"
        p.write_text(_to_latex(fmt, caption, name, notes), encoding="utf-8")
        paths.append(p)
        p = self.dirs["tables"] / f"{name}.md"
        md = "| " + " | ".join(map(str, fmt.columns)) + " |\n|" + "---|" * len(fmt.columns) + "\n"
        md += "\n".join("| " + " | ".join(map(str, r)) + " |" for r in fmt.itertuples(index=False))
        p.write_text(f"**{name}.** {caption}\n\n{md}\n\n{notes}\n", encoding="utf-8")
        paths.append(p)
        self._catalog.append(dict(type="table", name=name, kind=kind, caption=caption, notes=notes,
                                  stage=self.stage, files=[q.name for q in paths]))
        self.log("table %s", name)
        return paths

    def add_key(self, name: str, value: Any, desc: str, unit: str = "") -> None:
        """Register a number that the manuscript will quote."""
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        self._keys[name] = dict(value=value, unit=unit, desc=desc, stage=self.stage)

    def finalize(self) -> None:
        """Write this stage's catalogue and key numbers (one file per stage,
        so stages can run in separate processes)."""
        if self._catalog:
            (self.dirs["catalog"] / f"{self.stage}.json").write_text(
                json.dumps(self._catalog, indent=1, default=_json_default), encoding="utf-8")
        if self._keys:
            (self.dirs["keys"] / f"{self.stage}.json").write_text(
                json.dumps(self._keys, indent=1, default=_json_default), encoding="utf-8")

    # --------------------------------------------------------- provenance
    def provenance(self, stages: list[str]) -> dict:
        import importlib
        pkgs = {}
        for m in ("numpy", "scipy", "pandas", "skimage", "matplotlib", "statsmodels", "tifffile",
                  "pingouin", "sklearn", "yaml"):
            try:
                pkgs[m] = importlib.import_module(m).__version__
            except Exception:
                pkgs[m] = None
        files = {}
        for sid in self.sample_ids:
            f = Path(self.cfg["paths"]["images_dir"]) / self.samples.at[sid, "file"]
            files[f.name] = pio.sha256(f)
        for k in ("samples_csv", "lab_csv"):
            files[Path(self.cfg["paths"][k]).name] = pio.sha256(self.cfg["paths"][k])
        cfg = {k: v for k, v in self.cfg.items() if not k.startswith("_")}
        cfg["paths"] = dict(self.cfg.get("_paths_raw", cfg["paths"]))   # relative, no user-specific folders
        return dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), python=sys.version,
                    platform=platform.platform(), packages=pkgs, stages=stages, quick=self.quick,
                    segmentation_signature=self.segmentation_signature(), input_sha256=files, config=cfg)


def _latex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
           "±": r"$\pm$", "≤": r"$\leq$", "≥": r"$\geq$", "–": "--", "—": "---", "×": r"$\times$",
           "φ": r"$\phi$", "η": r"$\eta$", "χ": r"$\chi$", "²": r"$^2$", "µ": r"$\mu$", "Δ": r"$\Delta$",
           "ρ": r"$\rho$", "σ": r"$\sigma$", "τ": r"$\tau$", "κ": r"$\kappa$", "ε": r"$\varepsilon$"}
    return "".join(rep.get(ch, ch) for ch in str(s))


def _to_latex(df: pd.DataFrame, caption: str, label: str, notes: str = "") -> str:
    cols = "l" + "r" * (len(df.columns) - 1)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{{_latex_escape(caption)}}}", rf"\label{{tab:{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(_latex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(_latex_escape(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\par\smallskip\footnotesize {_latex_escape(notes)}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"
