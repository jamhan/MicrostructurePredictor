# LPBF 316L composition/microstructure/hardness public dataset

- Zenodo record: <https://zenodo.org/records/18800251>
- DOI: <https://doi.org/10.5281/zenodo.18800251>
- Lead creator: Jaromír Brůža
- License: Creative Commons Attribution 4.0
- Local raw directory: `data/public_316l_composition_2026/raw`
- Fetch command: `uv run python scripts/fetch_priority_zenodo_datasets.py --record 18800251`

The complete record was downloaded on 2026-07-23. Its one published archive
(806,362,680 bytes) passed the Zenodo MD5 check. The API response is retained
as `raw/zenodo_record.json`. No `.part` or failed-checksum files remain.

`Data-Impact_of_chemical_composition.zip` contains 131 files and expands to
1.24 GiB:

- five raw SEM TIFFs covering EOS, Praxair and SLM 316L powders;
- EBSD raw data and three `.h5oina`/`.oipx` map pairs;
- the original HV1 table with individual measurements, means and standard
  deviations;
- maximum Feret-diameter distributions derived from EBSD;
- the source README with acquisition and filename conventions.

The hardness table reports approximately:

| powder/source | mean HV1 | reported SD |
|---|---:|---:|
| EOS | 229.37 along build direction; 218.38 perpendicular; 224.91 overall | 8.11 overall |
| Praxair | 200.56 | 1.06 |
| SLM | 214.88 along build direction; 226.50 perpendicular; 219.86 overall | 5.71 overall |

Orientation is therefore a real linkage variable for EOS and SLM. It must not
be dropped when the SEM/EBSD record identifies a plane.
