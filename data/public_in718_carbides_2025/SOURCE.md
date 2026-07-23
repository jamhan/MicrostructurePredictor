# IN718 carbide-additive public dataset

- Zenodo record: <https://zenodo.org/records/16603134>
- DOI: <https://doi.org/10.5281/zenodo.16603134>
- Creator: Konrad Gruber
- License: Creative Commons Attribution 4.0
- Local raw directory: `data/public_in718_carbides_2025/raw`
- Fetch command: `uv run python scripts/fetch_priority_zenodo_datasets.py --record 16603134`

The complete record was downloaded on 2026-07-23. All 11 published files
(6,740,710,938 bytes) passed their Zenodo MD5 checks. The API response is
retained as `raw/zenodo_record.json`. No `.part` or failed-checksum files
remain.

## Archive inventory

| archive | compressed size | files | expanded size | useful contents |
|---|---:|---:|---:|---|
| `00_Powders.zip` | 67.7 MB | 68 | 0.09 GiB | 40 powder SEM TIFFs, EDS, PSD, spreadsheets |
| `01_Chemical_composition.zip` | 36 kB | 2 | negligible | XRF and C/S composition workbooks |
| `02_PBF-LB_process.zip` | 5.50 GB | 94 | 8.25 GiB | 80 TIFF process recordings, build files and parameter workbooks |
| `03_CALPHAD.zip` | 4.86 MB | 59 | 0.01 GiB | thermodynamic CSV/XLS data and plots |
| `04_Microscopy.zip` | 583.4 MB | 438 | 0.73 GiB | 401 TIFFs: optical, SEM, EDS and carbide masks |
| `05_Mechanical_properties.zip` | 2.27 MB | 30 | small | 12 raw tensile CSVs, workbooks, figures and HV1 workbook |
| `06_EBSD.zip` | 209.6 MB | 71 | 0.67 GiB | EBSD maps plus grain-size, aspect and misorientation tables |
| `07_Fractures.zip` | 372.3 MB | 172 | 0.48 GiB | 168 fracture SEM TIFFs and four optical images |
| `08_Tensile property map.zip` | 15.6 kB | 2 | negligible | literature/experimental tensile-property map and references |

Material codes used throughout the archive are:

- A: reference IN718;
- B: IN718 + 0.6 wt.% NbC;
- C: IN718 + 0.6 wt.% TiC;
- D: IN718 + 0.2 wt.% micron-sized B4C.

The mechanical archive contains three raw tensile replicates for each of A–D,
plus a Vickers HV1 workbook. The microscopy filenames encode material code,
as-built/heat-treated state, XY/XZ plane and field number. These variables form
the initial record-linkage key.

The 168 fracture images are post-test outcomes and must remain separate from a
pre-test microstructure-to-property model.
