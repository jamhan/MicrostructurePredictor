# Prior art, and what to take from it

Papers that shape what this project builds. Each entry ends with the concrete
thing to do about it rather than a summary, because the point of reading them
is to change the code or the data plan.

## Azimi et al. 2018: deep learning can separate the matrix constituents

Azimi, Britz, Engstler, Fritz, Mücklich, "Advanced Steel Microstructural
Classification by Deep Learning Methods," *Scientific Reports* 8, 2128 (2018).
doi:10.1038/s41598-018-20037-5. Preprint: <https://arxiv.org/abs/1706.06480>

They classified steel microstructures into martensite, tempered martensite,
bainite, and pearlite with a fully convolutional network doing pixel-wise
segmentation, then a max-voting scheme to collapse the pixel predictions into
one label per image. They report 93.94% accuracy against a prior state of the
art of 48.89%. That is close to a doubling on a task where trained
metallographers disagree with each other.

What to take:

The four classes they resolve are exactly the ones our segmentation collapses
into a single `ferrous/matrix` node, and exactly the ones that set hardness in
this steel. Two of our samples 200 HV apart look nearly identical to the
current segmenter for precisely this reason. Resolving the matrix into its
constituents is the most direct unblock available for the hardness head, and
this paper is the evidence that a network can do it.

Their aggregation is worth copying independently of the classes. We predict an
image-level label by pooling encoder features and classifying once
(`classify.py`). Predicting per pixel and then voting is better calibrated,
degrades more gracefully on mixed fields, and produces a spatial map for free
rather than a single guess.

The precondition to check before assuming the result transfers: their contrast
came from a specific etching and imaging protocol, and separating bainite from
martensite is difficult without it. Before building a matrix-subclass head, we
should confirm that UHCS SEM images at the magnifications we hold actually
carry that contrast. If they do not, this is a sample-preparation problem
wearing a model-shaped costume.

The blocking ingredient is labels, not architecture. Nothing in the current
benchmark distinguishes matrix constituents, so this is a task for
[DATASET_PLAN.md](DATASET_PLAN.md) before it is a task for `segment.py`.

## Holm et al. 2020: the map of the field

Holm, Cohn, Gao, Kitahara, Matson, Lei, Yarasi, "Overview: Computer Vision and
Machine Learning for Microstructural Characterization and Analysis,"
*Metallurgical and Materials Transactions A* 51(12), 5985–5999 (2020).
Preprint: <https://arxiv.org/abs/2005.14260>

Written by the group that produced the UHCS dataset we run on. It organizes
the field by how a micrograph is numerically encoded (hand-engineered features
versus learned representations) and by which task is being performed:
classification, semantic segmentation, object detection, instance
segmentation.

What to take:

Placed on their map, we do semantic segmentation and image classification, and
we do not do instance segmentation. That omission is the one that costs us.
`region_stats` in `features.py` uses connected components as a stand-in for
per-particle measurement, which merges touching particles and undercounts them
in dense fields. Real instance segmentation would give per-carbide size
distributions, which is both a better morphology feature and the quantity
Hecht's coarsening analysis is actually about.

Use its reference list as the reading graph for anything else in this space.

## Stuckner et al. 2022: microscopy pretraining, and the question we reopened

Stuckner, Harder, Smith, "Microstructure segmentation with deep learning
encoders pre-trained on a large microscopy dataset," *npj Computational
Materials* 8, 200 (2022). Weights:
<https://github.com/nasa/pretrained-microscopy-models>

They pretrained encoders on MicroNet, a large microscopy corpus, showed it
beats ImageNet pretraining for microstructure segmentation, and released the
weights. This is why training a usable segmenter on 24 labeled images is
possible at all.

What to take:

We already run these weights; they are what `checkpoints/backbone.pt` holds.

We also have a result that sits awkwardly against the paper's central claim.
A linear probe on frozen features over the UHCS benchmark, same split and same
head for each backbone, gave DINOv2 ViT-S/14 a mean IoU of 0.635 against
MicroNet resnet50 at 0.436 and plain ImageNet resnet50 at 0.495. On this data,
under that probe, domain pretraining lost to a general self-supervised model.
Treat it as an open question rather than a conclusion: the probe reads only the
deepest feature map, which understates what MicroNet contributes through the
U-Net skip connections, and DINOv2's finer patch stride flatters it on a dense
task. Settle it by building a real DINOv2-backed segmenter and comparing full
models, not probes.

Their release model is also the template for what this project could give back:
publish the trained encoder, not just the paper.

## Two more, specific to this project

DeCost, Lei, Francis, Holm, "High throughput quantitative metallography for
complex microstructures using deep learning: A case study in ultrahigh carbon
steel," *Microscopy and Microanalysis* 25, 21–29 (2019).
doi:10.1017/S1431927618015635. Preprint: <https://arxiv.org/abs/1805.08693>.
This is our segmentation benchmark and the source of the 0.7+ mean IoU we
measure ourselves against. They fully fine-tuned; we train the decoder only and
sit at 0.495. Read it before spending any effort optimizing segmentation.

DeCost and Holm, "A computer vision approach for automated analysis and
classification of microstructural image data," *Computational Materials
Science* (2015). The founding computer-vision-for-microstructure paper,
predating the deep learning shift. Worth knowing for its bag-of-visual-words
representations, which remain a useful interpretable baseline against CNN
features when you need to explain a prediction to a metallurgist.

## What scale these papers actually worked at

Worth knowing before treating our 24-image benchmark as a handicap.

| Work | Training images | Notes |
|---|---|---|
| Azimi 2018 | 11 | 21 total, 11 train / 10 test, 4 classes |
| Stuckner 2022, downstream | 1 to 18 | EBC1 18, EBC3 15, Super1 10, EBC2/Super2/Super4 4, Super3 1 |
| Stuckner 2022, pretraining | 100,000+ | MicroNet, 54 material classes |
| DeCost 2019 (our benchmark) | 24 total | we train on 17, validate on 7 |

The headline result in this literature came from eleven training images, and one
of NASA's benchmarks trained on a single image. Segmentation data volume is not
our problem, and it is not anyone's.

Image count is the wrong unit, though. Azimi's micrographs averaged 7000x8000 px,
which they cropped into 1000x1000 patches for 2831 training objects, then rotated
each by 90, 180 and 270 degrees. Their 11 training images carry roughly 616
megapixels against our 17 images at roughly 5.3, so they had on the order of 100
times more pixel data from fewer files. One of their patches covers about three
times the area of an entire UHCS micrograph. If we ever want more segmentation
signal, the lever is more material area per image (magnification and field of
view), not more images.

Their ablation also ranks the levers, and it is lopsided: training from scratch
gave 55.50%, fine-tuning from pretrained weights 87.98%, adding class balancing
90.97%, adding augmentation 93.94%. Transfer learning was worth about ten times
everything else combined. We already have transfer learning. We do not do class
balancing, and our weakest class by a wide margin is the rare Widmanstätten one,
which is exactly the case balancing addresses.

## What this list says about the gap

Every paper above is about characterization: segmenting, classifying,
representing. None predicts a mechanical property from an image. That
asymmetry runs through the whole literature, and it is not an oversight. The
canon is about seeing microstructure rather than predicting what it will do,
because paired image and property data barely exists.

That absence is the thing [DATASET_PLAN.md](DATASET_PLAN.md) is aimed at, and
it is the reason the distant-supervision work matters more than another
increment of segmentation accuracy.
