# Roadmap: from a working pipeline to microstructure inverse design

This document records where the project is aimed, honestly reconciled with
where it actually stands. It exists because the original goal (generate a
microstructure that achieves a target property) is genuinely exciting and
worth building toward, and because the path there is longer and less flashy
than the destination.

## The original aim, and where we strayed

The project started as microstructure to property prediction with a specific
edge: use topological features of the pore and grain structure to predict
fatigue life or crack propagation, on open micrograph datasets, with the
commercial motivation that physical fatigue certification (aerospace in
particular) is slow and expensive, so anything that substitutes computation
for testing has real value. The frontier version of the same idea is inverse:
given a target property, generate the microstructure and processing route that
produces it.

What we actually built is a clean, multi-material segmentation-to-property
pipeline validated on ultrahigh carbon steel, predicting hardness. That was
the right vehicle to get the infrastructure working, but it drifted from the
original thesis in three ways. The property became hardness, which is governed
by matrix state and not by the pore or network topology that motivated the
topological approach. The topological featurizer we wrote (`topo.py`) is
installed but never wired into the property fit. And the dataset is a single
steel, where the interesting fatigue and porosity story does not live.

None of that was wasted. The adapter and taxonomy design exists precisely so
new datasets and material families plug in without touching task code, which
is the vehicle the original "any alloy or composite micrograph" ambition
needs. The point of this document is to name the path back.

## The constraint that orders everything

The project has two sub-problems with opposite data situations, and almost
every decision follows from telling them apart.

Segmentation is label-scarce but image-rich: 24 pixel-labeled images, 961
unlabeled micrographs available. This is the regime where modern
self-supervised and foundation-model methods pay off, because they turn
unlabeled images into label efficiency.

Property prediction is label-scarce with no cheap fix: 7 hardness
measurements, and they vary along essentially one processing axis (cooling
method), so microstructure signal is confounded with the process log. No
architecture change fixes this. It is a data-acquisition problem wearing a
modeling costume.

The inverse-design endgame sits on top of both. You cannot condition a
generative model on a property you cannot measure or predict reliably, so the
generative dream depends on the property bottleneck being addressed first.
Most of the road back to the cool idea is therefore not generative modeling.

## The dependency ladder to inverse design

Inverse design is the top rung of a ladder, not a standalone project. Each rung
depends on the ones below it actually working.

Rung 0, a trustworthy forward model (microstructure to property with honest
uncertainty). This is where we are, blocked on property labels. Everything
above needs it, because inverse design is guided by the forward model.

Rung 1, an unconditional generative model of microstructure. A diffusion model
trained on the micrographs alone, no property labels required, that can produce
images a metallurgist accepts as real. This is buildable now with images we
already have, and it is a legitimate first piece of the generative goal rather
than a placeholder.

Rung 2, processing-conditioned generation. "Generate the microstructure for
970C / 90 min / water quench." This also needs no property labels, because the
processing metadata (anneal temperature, time, cooling method) already exists
in the sqlite for all samples. This is the most under-appreciated near-term
opportunity: a real, conditional generative model of microstructure that is
label-free to train, and a direct stepping stone toward inverse design.

Rung 3, property-conditioned generation. "Generate a microstructure with
hardness 700 HV." This needs property labels to train the conditioning, and it
needs Rung 0 to guide sampling (classifier or classifier-free guidance using
the forward property model). This is the first rung that is genuinely blocked
on the data problem.

Rung 4, full inverse design over process-structure-property. "Give me the
processing route for a target property." This needs process-structure-property
triples, which is the richest data requirement of all.

The honest reading: Rungs 1 and 2 are buildable today and worth building
because they advance the generative aim without waiting on labels. Rungs 3 and
4, the parts that make it inverse design, wait on property data.

## Near-term improvements

These are the concrete ideas worth pursuing, tagged by which bottleneck they
attack and whether they need new labels. Ordered roughly by value per effort.

DINOv2 or foundation-model features, no new labels. Swap the MicroNet encoder
for DINOv2 (or ensemble them) behind the same frozen-backbone heads and
compare. Foundation features are strongly label-efficient, and this is a
one-afternoon test of whether better representations move the needle at all
before investing in anything heavier. Start here.

Self-supervised pretraining on the 961 unlabeled micrographs, no new labels.
A masked-autoencoder or DINO objective learns steel-specific features that
fine-tune on the 24 labels with less overfitting. This is the same idea that
produced MicroNet at 100k-image scale, and there is prior work applying
self-supervised segmentation to this exact dataset. This is the real
segmentation unlock.

Foundation-model-assisted annotation, produces new labels cheaply. Segment
Anything does not know spheroidite from pearlite, but it proposes clean
boundaries. Human-in-the-loop, that can take labeling 200 of the 961 images
from weeks to days. Growing 24 to 200 labels beats any encoder swap, because
below 100 images data quantity dominates architecture.

Persistent-homology features for the property step, no new labels. Covered in
its own section below; this is the reconnection to the original thesis.

Physics-informed property model, no new labels. Hecht's thesis writes the
rule of mixtures for this system: hardness is the phase-fraction-weighted sum
of phase hardnesses. That is a four-parameter model you can fit on seven points
where a deep network cannot, and it is interpretable. Replace or baseline the
gradient-boosting head with it.

Calibrated uncertainty for the property step, no new labels. A Gaussian process
or conformal regression gives honest error bars instead of a false-confidence
point estimate, which is the correct posture at n=7 and a better scientific
artifact than a single number.

Active learning to prioritize measurements, chooses which new labels to get.
Use the forward model to decide which of the 47 UHCSDB samples to measure next
for maximum information. This is arguably the highest-value item on the whole
list, because it attacks the binding constraint directly.

Two things to deliberately avoid at current data scale. Transformer
segmentation heads (SegFormer, Mask2Former) are data-hungry and overfit 24
images worse than the CNN. Diffusion-generated synthetic training data means
validating a model on images produced by another model trained on the same
corpus, which can inject physically wrong microstructures invisibly. Both are
excellent above ten thousand images and counterproductive below a hundred.

## Persistent homology and the original fatigue thesis

The original project was built around topological featurization, and this is
the cleanest thread back to it. Persistent homology measures connectivity and
holes: H0 counts connected components, H1 counts loops and enclosed regions,
and persistence measures how robust each feature is to threshold. That is
exactly the language for the structures the original thesis cared about. A
pore network's connectivity determines where a fatigue crack nucleates and how
it propagates. A cementite network's connectivity is what Hecht ties to
toughness. These are topological properties that area fractions and
Hall-Petch relations do not capture, and that persistent homology does.

We already have `topo.py`, which computes H0 and H1 summaries per constituent
via Cubical Ripser or giotto-tda. It was written for this and then left
unused because the pipeline drifted toward hardness, where topology is not the
governing variable. The reconnection is straightforward: wire the topological
features into the `FeatureVector` and let the property head consume them, and
they become the natural representation the moment the target property is one
that topology governs.

That points at the real strategic choice. Hardness on UHCS is blocked on
labels and is not topology-driven, so it is a weak home for the original
thesis. The original aim wants a dataset that pairs microstructure or
tomography with fatigue or crack data, where pore and grain topology is the
governing variable, additively manufactured alloys with porosity being the
obvious candidate family. Coming back to the original aim most likely means a
dataset pivot, carried by the infrastructure already built: the adapters and
taxonomy for ingesting a new family, `topo.py` for the featurizer the thesis
was designed around, and the property-head registry for the new target. The
UHCS work becomes the validated reference implementation rather than the final
subject. Availability of specific open fatigue-plus-microstructure datasets
needs scoping before committing; that scoping is itself a high-value next step.

## The inverse-design endgame

For completeness, what the destination actually is. Inverse design is a
conditional generative model, p(microstructure | target property, processing),
that you sample from to obtain microstructures achieving a target. It is built
from three parts: a generative prior over microstructure (the diffusion model),
a conditioning signal (the property and processing), and a guidance mechanism
that steers sampling toward the target using the forward property model
(classifier guidance, or classifier-free guidance trained with the
conditioning). The forward model is not optional scaffolding; it is what makes
the generation goal-directed rather than merely realistic.

This is why the road runs through the data problem. A diffusion model that
generates realistic UHCS is buildable now and worth building (Rungs 1 and 2),
but the step that makes it design rather than generation is the property
conditioning, and that needs the labels. The intellectually honest sequence is
to build the label-free generative rungs in parallel with solving the property
data bottleneck, and to converge them only when both are ready.

## Recommended sequence

A defensible order that respects the constraints:

1. Run the DINOv2 feature-swap experiment. One afternoon, no new labels, tells
   us whether better representations are even a lever here.
2. Scope open fatigue-plus-microstructure datasets, and decide whether the
   project's home stays UHCS or pivots to a topology-governed property. This is
   the strategic fork and it gates everything topological.
3. In parallel, make the property step honest at n=7: physics-informed rule of
   mixtures plus calibrated uncertainty, and wire `topo.py` into the feature
   vector so the topological representation is ready for the pivot.
4. Attack the label bottleneck directly: request the raw hardness data behind
   Hecht's tabulated conditions, or stand up SAM-assisted annotation to grow
   the segmentation labels, or both. Everything above scales with this.
5. Build the label-free generative rungs (unconditional, then
   processing-conditioned diffusion on the micrographs) as the concrete first
   steps toward inverse design that do not wait on property data.
6. Converge: property-conditioned generation once labels and a trustworthy
   forward model exist.

The short version: the exciting generative destination is real and reachable,
but the intelligent path there is mostly data work and small-data method work,
with the genuinely buildable-now generative pieces being the label-free ones.
