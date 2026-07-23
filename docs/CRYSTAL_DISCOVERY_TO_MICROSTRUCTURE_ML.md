# Crystal discovery to microstructure ML

Status: research direction to investigate after the current data and validation
work. This document records the hypothesis; it does not make the work an
immediate implementation commitment.

## Core thesis

Adapt the ideas that made atomistic and crystal-structure machine learning
successful to the microstructure scale:

- geometric and symmetry-aware representations;
- self-supervised pretraining followed by data-efficient task adaptation;
- physics-based generation of training labels and candidate validation;
- calibrated property prediction;
- active-learning loops that choose the next useful calculation or experiment.

The aim is not to claim that these techniques have never been applied to
microstructure. Grain graph networks, microstructure generators and
simulation-trained property surrogates already exist. The opportunity is to
combine the pieces into an experimentally grounded
processing--microstructure--property system that generalizes beyond a single
dataset or imaging protocol.

## Translating the crystal-discovery analogy

The useful analogy is broader than "atoms become grains." A microstructure is
a history-dependent, multiscale field, and the right entities depend on the
material and the measurement.

| Crystal discovery | Microstructure analogue |
|---|---|
| atoms | grains, particles, pores, cracks or connected phase regions |
| bonds or neighbour lists | shared interfaces, contact, proximity, misorientation and phase relationships |
| crystal graph | multiscale graph, ideally combined with image or voxel fields |
| composition and unit cell | composition, process history, specimen state and spatial scale |
| formation-energy/property model | calibrated engineering-property or performance model |
| DFT label generation | phase-field, CALPHAD, crystal-plasticity/FEM and direct experiments |
| DFT relaxation | process-realizability and physics validation, not one universal relaxation step |
| crystal generation | a distribution of microstructures conditional on composition and process |
| active-learning search | selection of the next feasible process condition, characterization and test |

For a clean EBSD polycrystal, grains can be nodes and grain boundaries can be
edges. For an SEM image of multiphase steel, however, the observable entities
may instead be carbide networks, particles and connected phase regions. A
grain-only graph would discard important information and may not be
constructible from the image at all.

A graph can express adjacency and interaction well, but it can lose continuous
shape, texture, interface thickness and within-region fields. The most useful
representation may therefore be hybrid:

1. pixels or voxels for local appearance and continuous morphology;
2. a graph for entities and their physical relationships;
3. global features for composition, processing, test conditions and imaging
   metadata.

## Research strands

### 1. Structured representations

Compare representations rather than assuming that a graph is superior:

- raw microscopy pixels;
- pretrained vision embeddings;
- segmented phase fractions and morphology;
- topology descriptors;
- region- or grain-adjacency graphs;
- hybrid graph-plus-field representations.

The scientific question is whether a representation improves data efficiency,
physical interpretability and out-of-distribution generalization, not merely
whether it lowers random-split test error.

Representations should respect the relevant symmetries. Scalar properties
should normally be invariant to an arbitrary change of image or specimen
reference frame. Directional and tensor properties should transform
equivariantly, while crystallographic orientations must respect their material
symmetries.

### 2. Foundation representations

Do not begin by training a large foundation model from scratch. Start with
existing encoders such as DINOv2, MicroNet or other microscopy models, then
test domain-specific self-supervision on the available unlabeled images.

Any future large pretraining corpus must preserve:

- physical scale and magnification;
- modality, detector and instrument;
- specimen preparation and etching;
- material family, composition and processing provenance;
- relationships among fields of view from the same physical specimen.

Without that context, a model may learn laboratory and acquisition style
rather than material structure.

### 3. Property prediction

The forward model should combine microstructure, composition and process
metadata, but it must also demonstrate that microstructure contributes
information beyond grade and process alone.

Required evaluations include:

- splits by physical specimen;
- held-out processing conditions, batches and material grades;
- metadata-only and composition/process-only baselines;
- tests across instruments, magnifications and preparation routes;
- calibrated uncertainty on direct property measurements;
- ablations that measure the incremental value of microstructure.

The initial target should be a precisely defined property under documented
test conditions. "Engineering performance" is not one label: yield strength,
fatigue life, toughness and conductivity each depend on additional state and
test variables.

### 4. Generative microstructures

Generation should be staged:

1. learn an unconditional distribution of plausible microstructures;
2. learn `composition + process -> distribution of microstructures`;
3. establish a trustworthy `microstructure -> property` forward model;
4. search over feasible processes for a target property;
5. validate the proposed process, resulting structure and measured property.

Directly asking a generator for "the microstructure that gives 900 MPa yield
strength" is under-specified and non-unique. The alloy, test temperature,
strain rate, orientation, specimen state and feasible manufacturing processes
must also be defined. Property-conditioned images are hypotheses, not inverse
design, until a realizable process creates them and testing verifies the
target.

### 5. Physics and process-realizability

There is no single mesoscale equivalent of DFT relaxation. The relevant
physics tools answer different parts of the chain:

- CALPHAD constrains thermodynamics and phase stability;
- phase-field and grain-growth models describe aspects of microstructure
  evolution;
- crystal plasticity and FEM estimate mechanical response;
- direct experiments establish whether the simulated and learned
  relationships transfer to manufactured material.

Physics should be used to generate data, constrain models and reject
unrealizable candidates throughout the workflow, rather than only as a final
cosmetic refinement.

### 6. Active learning

The eventual loop is:

```text
choose process
    -> manufacture
    -> characterize microstructure
    -> test properties
    -> update model and uncertainty
    -> choose the next feasible process
```

The acquisition function should select processing conditions and measurements,
not attractive synthetic images. It should balance expected improvement,
information gain, experimental cost, safety and manufacturability. A
simulation-only loop is a useful precursor, but it does not remove the need
for prospective experimental validation.

## First investigation: a representation benchmark

The current UHCS data is suitable for a processing-information benchmark, but
not yet for a convincing comparison of property models. There are hundreds of
micrographs with processing metadata but only seven directly linked hardness
specimens.

### Question

Do structured microstructure representations improve data efficiency and
generalization across specimens, magnifications and processing conditions
relative to generic visual representations?

### Initial task

Use heat-treatment recovery as the target while property data is scarce. The
existing temperature probe already shows that:

- image features contain some austenitization-temperature signal;
- acquisition metadata is a strong baseline;
- the result depends sharply on magnification;
- pooling features indiscriminately across physical scales destroys signal.

### Representations to compare

1. CNN features from the existing microscopy backbone.
2. DINOv2 or another general vision-transformer representation.
3. Segmented phase fractions, morphology and topology.
4. A graph of segmented microconstituent regions and their interfaces.
5. A hybrid graph-plus-image representation if the graph alone loses
   important texture.

All representations should use the same physical-sample splits and comparable
downstream heads. Magnification, detector and other acquisition fields should
form an explicit metadata-only baseline. Results should be reported by scale,
not only as a pooled score.

### Decision value

This experiment can show whether structured representations are a real lever
before the project invests in a larger graph architecture or pretraining
programme. A negative result is informative: it may show that graph
construction discarded relevant visual information, that segmentation quality
is limiting, or that the available images do not contain the proposed
physical signal.

A true grain-graph comparison requires EBSD or 3D data with defensible grain
boundaries and orientations. The current UHCS SEM segmentation should not be
presented as a grain graph.

## Longer-term sequence

1. Run the processing-information representation benchmark.
2. Scope an EBSD or 3D dataset with processing history and direct property
   measurements.
3. Establish a forward property benchmark with held-out-condition and
   held-out-batch tests.
4. Add physics-generated data only with an explicit simulation-to-experiment
   validation plan.
5. Build process-conditioned microstructure generation.
6. Introduce constrained process optimization and active learning.
7. Close the loop prospectively with manufactured and tested specimens.

## Long-term vision

The credible near-term objective is not a universal world model of all
materials. It is a validated, uncertainty-aware model for a defined material
family and processing envelope:

```text
composition + processing
        -> distribution of microstructures
        -> distribution of measured properties
```

Once the forward chain is trustworthy, it can be searched in reverse:

```text
target property and manufacturing constraints
        -> candidate processing routes
        -> predicted microstructures and uncertainty
        -> experiments
```

A reusable foundation representation may eventually sit beneath several such
family-specific models, but the family-specific data and physics remain
essential.

## Commercial hypothesis

"AI for microstructure" is already a commercial category, so a generic
segmentation or image-generation claim is unlikely to be a durable
differentiator. Potential defensible positions include:

- proprietary, reliably linked process--microstructure--property data;
- performance on small, expensive experimental datasets;
- calibrated uncertainty and auditable provenance;
- process-realizable inverse design;
- a narrow qualification problem, such as fatigue, where testing is slow and
  expensive.

The commercial case should be assessed through a specific user, decision and
avoided cost rather than through model novelty alone.

## Selected prior art

- Dai et al., "Graph neural network for predicting the effective properties of
  polycrystalline materials: A comprehensive analysis," *Computational
  Materials Science* 230, 112461 (2023).
- Pagan et al., "Graph neural network modeling of grain-scale anisotropic
  elastic behavior using simulated and measured microscale data," *npj
  Computational Materials* 8, 259 (2022).
- Qin et al., "GrainGNN: A dynamic graph neural network for predicting 3D
  grain microstructure," *Journal of Computational Physics* 113061 (2024).
- Wei and Chen, "Foundation Model for Polycrystalline Material Informatics,"
  arXiv:2512.06770 (2025).
- Holm et al., "Overview: Computer Vision and Machine Learning for
  Microstructural Characterization and Analysis," *Metallurgical and Materials
  Transactions A* 51, 5985--5999 (2020).
- Barmak and Rickman, "Machine learning for materials microstructures: A
  survey of applications and methodologies," *Acta Materialia* 122041 (2026).
