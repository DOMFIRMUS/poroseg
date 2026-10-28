"""Figure style shared by every stage.

Rules (applied everywhere)
--------------------------
* One fixed colour per thresholding method, in a fixed order, never cycled or
  re-assigned (colour follows the entity). Colour is always redundant with
  position, marker or a label - never the only carrier of identity.
* Global methods use filled circles, local methods filled squares.
* Sequential maps: a single blue hue, light -> dark. Diverging maps: blue <->
  red with a neutral grey midpoint. No rainbow maps; 'gray' for CT images.
* Hairline, solid, recessive grid and axes; no dual y-axes.
* Journal widths: 90 mm (single), 140 mm (1.5), 190 mm (double column).
* PDF output keeps text editable (TrueType, fonttype 42).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4

# Categorical palette (reference instance, documented order, validated for CVD
# on adjacent pairs). Mapping to methods is fixed.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_ORDER = ["Otsu", "Li", "Yen", "Mean", "Triangle", "Isodata", "Sauvola", "Niblack"]
METHOD_COLORS = dict(zip(METHOD_ORDER, PALETTE))
METHOD_MARKERS = {m: ("s" if m in ("Sauvola", "Niblack") else "o") for m in METHOD_ORDER}

POST_ORDER = ["none", "median", "open_close", "skeleton"]
POST_LABELS = {"none": "None", "median": "Median 3×3", "open_close": "Open–close 3×3", "skeleton": "Skeleton"}
# post-processing levels use the first three slots (validated all-pairs) + a neutral
POST_COLORS = {"none": "#0b0b0b", "median": PALETTE[0], "open_close": PALETTE[1], "skeleton": PALETTE[2]}
POST_MARKERS = {"none": "o", "median": "s", "open_close": "^", "skeleton": "D"}

GROUP_LABELS = {"granular_pack": "Granular pack", "consolidated": "Consolidated"}
GROUP_MARKERS = {"granular_pack": "o", "consolidated": "D"}

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
REF_BAND = "#d9d8d3"       # laboratory reference band
SURFACE = "#ffffff"

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_SEQ_W = LinearSegmentedColormap.from_list("seq_blue_w", ["#ffffff"] + SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f4b3ae", "#e34948", "#8c1f1e"])
PORE_OVERLAY = "#eb6834"   # colour used to paint pores over CT images


def apply_style(font_size: float = 7.5) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size + 0.5,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "legend.fontsize": font_size - 0.5,
        "legend.title_fontsize": font_size - 0.5,
        "figure.titlesize": font_size + 1,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titleweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "image.cmap": "gray",
        "image.interpolation": "nearest",
    })


def figsize(width: str | float = "double", height_mm: float = 80.0) -> tuple[float, float]:
    widths = {"single": 90.0, "onehalf": 140.0, "double": 190.0}
    w = widths[width] if isinstance(width, str) else float(width)
    return w * MM, height_mm * MM


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02, **kw) -> None:
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=mpl.rcParams["font.size"] + 1,
            fontweight="bold", va="bottom", ha="right", color=INK, **kw)


def hgrid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = mpl.ticker.PercentFormatter(xmax=1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def scalebar(ax, pixel_size_um: float, length_um: float | None = None, loc: str = "lower right",
             color: str = "white", img_width_px: int | None = None, pad_frac: float = 0.04) -> None:
    """Draw a physical scale bar on an image axis (data coordinates = pixels)."""
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    w_px = x1 - x0
    if length_um is None:
        target = 0.2 * w_px * pixel_size_um
        nice = np.array([50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000])
        length_um = float(nice[np.argmin(np.abs(nice - target))])
    L = length_um / pixel_size_um
    label = f"{length_um / 1000:g} mm" if length_um >= 1000 else f"{length_um:g} µm"
    bar = AnchoredSizeBar(ax.transData, L, label, loc, pad=0.4, sep=2, color=color, frameon=False,
                          size_vertical=max(0.008 * (y1 - y0), 1.0), label_top=True,
                          fontproperties=FontProperties(size=mpl.rcParams["font.size"] - 0.5))
    ax.add_artist(bar)


def method_handles(methods=METHOD_ORDER, **kw):
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="none", marker=METHOD_MARKERS[m], color=METHOD_COLORS[m],
                   markersize=kw.get("ms", 4), label=m) for m in methods]


def save(fig, stem: Path, formats=("pdf", "png"), dpi: int = 600) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        kw = dict(bbox_inches="tight", pad_inches=0.02)
        if fmt in ("png", "tif", "tiff"):
            kw["dpi"] = dpi
        if fmt in ("tif", "tiff"):
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, **kw)
        out.append(p)
    plt.close(fig)
    return out
