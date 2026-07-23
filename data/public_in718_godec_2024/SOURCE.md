# Godec et al. IN718 public dataset

- Public record: <https://zenodo.org/records/14163786>
- DOI: <https://doi.org/10.5281/zenodo.14163786>
- License: Creative Commons Attribution 4.0
- Local fetch: `python scripts/fetch_zenodo_in718.py`

The fetch script retains the 22 files named `BEI *.tif`, the hardness workbook,
the two mechanical-test workbooks, and the Zenodo API metadata.  It verifies
each downloaded file against the MD5 published by Zenodo.

`hardness.csv` and `tensile.csv` are auditable transcriptions of the supplied
workbooks.  They are kept outside the workbooks so the runtime does not depend
on an Excel parser and so every value used by the matcher has a stable source
locator.

The filename carries state (`AB` or `HT`), heat-treatment temperature, beam
shape (`Gauss` or `Ring`), and field number.  This supports a high-confidence
same-source/same-condition link to the HV1 means.  It does **not** prove that a
field is the exact indent neighbourhood, so these hardness labels are
`distant`, carry training weight 0.85, and are excluded from validation.

The tensile workbook separates H and V orientations, while the BSE-SEM
filenames do not.  Both orientation-specific values therefore remain
medium-confidence alternatives.  The matcher must neither select one nor
average them automatically.

The archive also contains orientation-labelled fracture SEM images. They are
not fetched for the forward microstructure model: a fracture surface is
created by the tensile test and would leak the outcome into the input. They
may support a separate post-mortem fracture-analysis task later.
