"""Convert 'dados reais de porosidade.xlsx' into a tidy, auditable CSV.

The spreadsheet has two blocks:

* granular packs (columns H-M): one row per specimen, one column per laboratory
  method (direct porosimetry, gravimetry, indirect porosimetry, gas porosimetry,
  X-ray uCT) plus a pre-computed mean of the five columns;
* consolidated samples (rows 15-21, columns H-I): the helium porosity of seven Tambaba
  limestone plugs numbered as in Cardoso (2020), "Amostra 1-7"; only Amostra 1 is an imaged
  plug (VN02). VN01 and VN04 are not in the file: their helium porosity comes from Vila Nova (2020).

Output: data/lab_porosity.csv with columns
    sample_id, specimen, method, porosity_pct, has_image, source

Usage:  python scripts/build_lab_table.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Laboratory column header (as typed in the spreadsheet) -> method code
METHOD_COLUMNS = {
    "Porosimetria direta (%)": "direct",
    "Gravimetria (%)": "gravimetric",
    "Porosimetria indireta (%)": "indirect",
    "Porosimetria a g": "gas",          # header contains a non-ASCII character; matched by prefix
    "TC-RX (%)": "uct",                 # idem ('µTC-RX (%)')
}

# Specimen code in the spreadsheet -> sample_id used by the pipeline (None = no image)
PACK_IDS = {
    "CF4_3mm": "DS23_CF4", "CF5_3mm": "DS23_CF5", "CF7_4mm": "DS23_CF7", "CF10_M": "DS23_CF10",
}
# Cardoso (2020) numbers the plugs differently from Vila Nova (2020): "Amostra 1" is VN02, and "Amostra 2"
# and "Amostra 4" are other plugs (corrected 2026-11-01; the first version mapped Amostra 1, 2, 4 to VN01, 02, 04)
CONSOLIDATED_IDS = {"Amostra 1": "AG24_VN02"}
CONSOLIDATED_NAMES = {"Amostra 1": "VN02"}
NOT_IMAGED_NOTES = {"Amostra 2": " (not VN02)", "Amostra 4": " (not VN04)"}
# imaged plugs that are not in the spreadsheet: helium porosity of the whole plug, Vila Nova (2020)
VILA_NOVA = {"AG24_VN01": ("VN01", 16.55), "AG24_VN04": ("VN04", 6.67)}
IMAGED_ORDER = ["AG24_VN01", "AG24_VN02", "AG24_VN04"]


def _match_method(header: str) -> str | None:
    header = str(header)
    for key, code in METHOD_COLUMNS.items():
        if key in header:
            return code
    return None


def build(xlsx: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, header=None)
    rows = []

    # ---- block 1: granular packs -------------------------------------------
    hdr_row = raw.index[raw[7].astype(str).str.strip().eq("Colunas ID")][0]
    headers = raw.loc[hdr_row]
    method_cols = {c: _match_method(h) for c, h in headers.items() if _match_method(h)}
    r = hdr_row + 1
    while r < len(raw) and isinstance(raw.at[r, 7], str) and raw.at[r, 7].strip():
        specimen = raw.at[r, 7].strip()
        for c, method in method_cols.items():
            val = raw.at[r, c]
            if pd.notna(val):
                rows.append(dict(sample_id=PACK_IDS.get(specimen, f"PACK_{specimen}"),
                                 specimen=specimen, method=method, porosity_pct=float(val),
                                 has_image=specimen in PACK_IDS,
                                 source=f"{xlsx.name}: row {r + 1}"))
        r += 1

    # ---- block 2: consolidated samples ---------------------------------------
    hdr2 = raw.index[raw[7].astype(str).str.strip().eq("Amostras")][0]
    cons = [dict(sample_id=sid, specimen=name, method="helium", porosity_pct=phi, has_image=True,
                 source=f"Vila Nova (2020): helium porosity of plug {name} (corrected 2026-11-01)")
            for sid, (name, phi) in VILA_NOVA.items()]
    r = hdr2 + 1
    while r < len(raw) and isinstance(raw.at[r, 7], str) and raw.at[r, 7].strip():
        specimen = raw.at[r, 7].strip()
        if specimen in CONSOLIDATED_IDS:
            name = CONSOLIDATED_NAMES[specimen]
            src = (f"Vila Nova (2020): helium porosity of plug {name} = {specimen} of Cardoso (2020); "
                   f"{xlsx.name}: row {r + 1}")
        else:
            name = specimen
            src = f"Cardoso (2020) Table 7{NOT_IMAGED_NOTES.get(specimen, '')}; {xlsx.name}: row {r + 1}"
        cons.append(dict(sample_id=CONSOLIDATED_IDS.get(specimen, f"CONS_{specimen.replace(' ', '')}"),
                         specimen=name, method="helium", porosity_pct=float(raw.at[r, 8]),
                         has_image=specimen in CONSOLIDATED_IDS, source=src))
        r += 1
    # imaged plugs first, in sample order, then the plugs without an image in the order of the file
    cons.sort(key=lambda d: IMAGED_ORDER.index(d["sample_id"]) if d["sample_id"] in IMAGED_ORDER else len(IMAGED_ORDER))
    rows += cons

    return pd.DataFrame(rows)


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    xlsx = (cfg_path.parent / cfg["paths"]["lab_xlsx"]).resolve()
    out = (cfg_path.parent / cfg["paths"]["lab_csv"]).resolve()
    df = build(xlsx)
    df.to_csv(out, index=False)
    print(f"{len(df)} lab records -> {out}")
    print(df.pivot_table(index="specimen", columns="method", values="porosity_pct", sort=False).round(2))


if __name__ == "__main__":
    main()
