# Data

Input data of the article, read by the pipeline through the paths in `../config.yaml`. Released under the
Creative Commons Attribution 4.0 International licence (CC BY 4.0); archived with the derived data at
https://doi.org/10.5281/zenodo.23748708.

| Path | Content |
|---|---|
| `images/` | one reconstructed 16-bit X-ray micro-CT slice per specimen (Nikon XT H 225 ST, X-ray Computed Tomography Laboratory, Department of Nuclear Energy, Federal University of Pernambuco) |
| `samples.csv` | sample metadata: group (granular pack or consolidated core), holder, pixel size (25 µm, or 40 µm for TQ06 and GT01), notes |
| `lab_porosity.csv` | laboratory porosity in long format: four physical methods and the laboratory's own micro-CT value for the granular packs (nine specimens, four imaged); helium porosity of the whole plug for the limestone cores VN01, VN02 and VN04 (Vila Nova, 2020) |
| `legacy_results/analysis_20250723_145220/global_analysis/` | porosities (`pivoted_data.csv`) and ICC table stored by the original analysis code, read by the `legacy` stage |

Specimens: CF4, CF5, CF7 and CF10 are packs of spheroidal grains 3–4 mm in diameter (CF10 with mixed sizes) in
a cylindrical holder; VN01, VN02 and VN04 are reef limestones of the Tambaba Formation; TQ06 is a trachyte and
GT01 a coquina. The acquisition and reconstruction settings of the scans are not available.

SHA-256 of the slices (also recorded in `outputs/logs/provenance.json` by every run):

```
129a9c636a61fb318839872e7ab6cd8264aa91dfbaf9cb973bf8356f7cbce935  AG24_VN01_01_0207.tif
9a0d99e147534193b047daf777b3d82fb6d39e899fb844d18722a764de9b66dd  AG24_VN02_250425.tif
ee1ff182608aa6c734d8f43ecb358027028e096c278adf83cc83418bf8184978  AG24_VN04_01_0049.tif
7815384b5766a09a9d1b6cc97fdf587fe1aea93b7f2e29f7c103fa753500b2fd  BC23_TQ-06_40_0114.tif
119601f363a171216b456fcba92fb6081502a218b88cee08fef8a4e4747923e2  DS23_CF10_M_0245.tif
05d41fb4ac73a813e66262cae92d16ac6a5150f3ac32d2180d373467760d440c  DS23_CF4_3mm_25g_0244.tif
0adc98bc2c36ae0f4185f11520630f0d7da8ca4317ea320755515064d70a10ff  DS23_CF5_3mm_25g0280.tif
0d0041c25397b0ded16a6a9ecd6c7340085c5408ee8500f6052168373f34ed04  DS23_CF7_4mm_25_0340.tif
04991022c360c3610ce19df704fb914948c225eb212553a1243da8edd3320f3c  GT24_01_40_0220.tif
```

`../scripts/build_lab_table.py` documents how `lab_porosity.csv` was derived from the laboratory spreadsheet.
