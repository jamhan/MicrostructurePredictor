# Ren et al. 2023

## Citation and reuse

Qichao Ren, Ziming Kou, Juan Wu, Tengyan Hou, and Peng Xu. "Effect of
Tempering Temperature on Microstructure and Mechanical Properties of 35CrMo
Steel." *Metals* 13 (2023), 771.

- DOI: <https://doi.org/10.3390/met13040771>
- article: <https://www.mdpi.com/2075-4701/13/4/771>
- official PDF: <https://mdpi-res.com/d_attachment/metals/metals-13-00771/article_deploy/metals-13-00771-v2.pdf>
- source Figure 2 image: <https://mdpi-res.com/d_attachment/metals/metals-13-00771/article_deploy/html/images/metals-13-00771-g002.png>
- license: Creative Commons Attribution 4.0,
  <https://creativecommons.org/licenses/by/4.0/>
- accessed and extracted: 2026-07-21

`figure2_b.png` is the SEM half of the source's composite Figure 2. The crop
keeps source columns 2072 through 4141 over the full 1430-pixel height. It was
saved losslessly as PNG without resizing or altering scientific content. The
panel letter and 5 um scale bar remain. This is a cropped derivative under the
source's CC BY license; this file supplies attribution and states the change.

## Material and processing

The source material was a 150 mm diameter by 100 mm long forged 35CrMo round
bar. Table 1 reports this experimental composition in wt.%:

| C | Si | Mn | S | P | Cr | Mo |
|---:|---:|---:|---:|---:|---:|---:|
| 0.343 | 0.26 | 0.71 | 0.032 | 0.027 | 0.92 | 0.16 |

The bar was cut into 70 mm by 48 mm by 12 mm specimens, homogenized at 1050 C
for 20 h, heated to 860 C over 3 h, held at 860 C for 2 h, removed, and oil
cooled to room temperature. The exact controlled condition id retains both
thermal steps:

    condition/homogenize_austenitize/oil_quench/t1050c_20h_t860c_2h

The microstructure was ground, polished, etched with 4% nitric acid alcohol
solution and observed using a Zeiss GeminiSEM 500. Figure 2b on PDF page 5 is
the SEM panel for the oil-quenched state.

## Property measurement and match

The methods on PDF page 4 state that Vickers hardness was measured on the
polished sample surface with a 500 gf load and 10 s dwell. Section 3.1 on PDF
page 5 describes Figure 2 as the microstructure after the 860 C for 2 h oil
quench, then reports hardness as 532.1 plus or minus 7.2 HV in the paragraph
immediately above the figure.

The mean and 7.2 spread are transcribed exactly. The paper does not say whether
the plus-or-minus value is a standard deviation, range, standard error or
another uncertainty, so the manifest uses
`reported_plus_minus_unspecified`. It also does not give a hardness indent or
specimen count, so `n_measurements` is blank.

The figure and value have an explicit same-study, same-batch, same-condition
match. The source does not prove that the SEM field is from the exact physical
coupon or indent neighbourhood tested for hardness. The manifest therefore
uses `same_study_batch_condition`, sets
`exact_physical_specimen_confirmed=false`, and tags the property `distant`.

The panel contains its source letter and scale bar. Raw-image training must
mask or crop those markings mechanically while retaining this unmodified
provenance copy.
