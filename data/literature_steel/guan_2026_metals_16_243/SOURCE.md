# Guan et al. 2026

## Citation and reuse

Boyu Guan, Shaobin Bai, Yongqing Zhang, Peimao Fu, Haitao Lu, Hejia Zhu,
Xingchi Chen, Kaikai Guo, Haonan Wang, and Yongan Chen. "Effect of Tempering
on Microstructure, Strength and Toughness Gradient in Quenched Low-Alloy
Medium-Thickness Steel Plate." *Metals* 16 (2026), 243.

- DOI: <https://doi.org/10.3390/met16030243>
- article: <https://www.mdpi.com/2075-4701/16/3/243>
- official PDF: <https://mdpi-res.com/d_attachment/metals/metals-16-00243/article_deploy/metals-16-00243.pdf>
- license: Creative Commons Attribution 4.0,
  <https://creativecommons.org/licenses/by/4.0/>
- accessed and extracted: 2026-07-21

The files `figure3_a` through `figure3_r` are the 18 raster panels embedded in
Figure 3 on PDF page 5. They were extracted from the official PDF and renamed.
No visual content was changed. The original panel letters, scale bars, and
author annotations remain in the images. This file supplies the attribution
required by the source license and describes the extraction.

## Material and processing

The study used one industrially produced 25 mm low-alloy high-strength steel
plate. Table 1 gives the measured composition in wt.%:

| C | Si | Mn | Cr | Mo | Al | Nb | Ti | Fe |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 0.15 | 0.25 | 1.45 | 0.40 | 0.18 | 0.025 | 0.025 | 0.015 | balance |

The plate was hot rolled and online direct quenched by water spray from 900 C
to below 150 C. The two tempered states were held at 530 C or 580 C for 1.5 h,
then immediately water quenched to room temperature. The paper names the three
states DQ, DQ-T530, and DQ-T580.

Metallographic samples came from the upper surface layer, mid-thickness, and
lower surface layer. They were etched in 4 vol.% nital and imaged with a TESCAN
MIRA 3 SEM. The methods are on PDF pages 3 and 4.

## Property measurement

Table 3 on PDF page 11 reports HV1 hardness. The test used a 1 kgf load and a
10 s dwell. Each table cell is the mean of five independent measurements. The
paper does not report a standard deviation or range, so the manifest leaves
scatter blank and says `unreported`.

| state | upper surface | mid-thickness | lower surface |
|:---|---:|---:|---:|
| DQ | 201.25 | 210.17 | 183.77 |
| DQ-T530 | 262.82 | 246.14 | 242.26 |
| DQ-T580 | 270.32 | 253.70 | 230.88 |

Values are recorded exactly as published. There is no hardness-scale
conversion and no invented uncertainty.

## Image-to-property match

Figure 3's caption assigns each panel pair to one treatment and one
through-thickness location. Table 3 uses the same treatment and location
labels. The manifest records this as
`same_study_plate_condition_location`.

| panels | specimen id | table cell |
|:---|:---|:---|
| a, b | `dq_upper` | DQ, upper surface layer |
| c, d | `dq_mid` | DQ, mid-thickness |
| e, f | `dq_lower` | DQ, lower surface layer |
| g, h | `dqt530_upper` | DQ-T530, upper surface layer |
| i, j | `dqt530_mid` | DQ-T530, mid-thickness |
| k, l | `dqt530_lower` | DQ-T530, lower surface layer |
| m, n | `dqt580_upper` | DQ-T580, upper surface layer |
| o, p | `dqt580_mid` | DQ-T580, mid-thickness |
| q, r | `dqt580_lower` | DQ-T580, lower surface layer |

This is a high-confidence metadata match, but it is still a weak property
label. The paper establishes the same plate, treatment, and sampling location;
it does not establish that an SEM field is the exact physical hardness-tested
coupon or indent neighbourhood. `property_source` is therefore `distant`, and
`exact_physical_specimen_confirmed` is `false` for every panel.

Each panel pair is one split group. The two magnifications must not be counted
as independent hardness observations or placed on opposite sides of a model
split.

The sampling location is essential. The three values within a treatment are
different, so these values do not belong in `data/property_lookup.csv`, whose
key is only `(alloy_grade, condition)`. The literature adapter attaches the
value directly from the panel manifest and includes location in `specimen_id`.

## Training caveats

The published panels contain letters, scale bars, arrows, and phase labels.
Those markings can become shortcuts. Any raw-image training run must crop or
mask them mechanically and keep the unmodified files as the provenance copy.
No such cleaned derivative is claimed here.

HV1 is not automatically interchangeable with hardness measured at a
different Vickers load. Training and reporting should retain the 1 kgf load,
even though the current canonical property name is the broader
`hardness_hv`.
