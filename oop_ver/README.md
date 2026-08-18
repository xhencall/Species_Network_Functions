# Species_Network_Functions

An extended framework built on **PhyloX** for species network manipulation, composite likelihood computation, curvature adjustment matrix calculation, and Bayesian parameter inference under the **Multispecies Network Coalescent (MSNC)** model.

---

## Key Features

* **Object-Oriented Network Architecture**
  * `SpeciesNetwork`: Wraps PhyloX `DiNetwork` graphs with automatic node attribute delegation (`__getattr__`), taxonomy/label mapping, topology pruning, and ladderization routines.
  * Encapsulated topology strategies (`AsymmQuartet`, `SymmQuartet`) using polymorphism to eliminate branch-checking overhead when computing true site pattern probabilities.
  * Extract displayed trees and weighted displayed tree mappings ($\gamma$-IDs) from phylogenetic networks.
* **Alignment & Site Pattern Processing**
  * Parse DNA alignments with full support for IUPAC ambiguity codes and multi-individual-per-species partitions.
  * Vectorized mapping matrices ($E_{\text{mat}}$) linking global site pattern counts ($n^D$) to sub-quartet frequencies ($n^Q$).
* **Composite Likelihood Estimation (MCLE)**
  * Analytic calculation of exact 15-category site pattern probabilities ($p^Q$) under the Jukes-Cantor (JC69) substitution model.
  * `Optimizer`: Supports unconstrained and constrained multi-start BFGS optimization using parameter-space transformations via `ParameterTransformer` ($\boldsymbol{\tau}, \theta, \boldsymbol{\gamma}$).
* **Godambe Information & Curvature Adjustment**
  * `CurvatureAdjustmentCalculator`: Computes score functions ($U_c$), variability matrices ($J$), sensitivity matrices ($H$), and Godambe curvature-adjustment matrices ($C$) to adjust composite likelihood surfaces.
* **Bayesian MCMC Inference**
  * `MCMCSampler`: Metropolis-within-Gibbs sampling for joint posterior inference across speciation times ($\boldsymbol{\tau}$), effective population size ($\theta$), and inheritance probabilities ($\boldsymbol{\gamma}$).
  * `ProposalKernel` & `TauPriorTauConstraint`: Custom proposal distributions (Normal, Uniform, Bactrian) with vectorized boundary reflections and topological ancestor-descendant branch length constraints.

---

## Dependencies

* **Python 3.8+**
* **Core Libraries:** `numpy`, `scipy`, `networkx`, `phylox`, `dendropy`, `tqdm`

Install the main dependencies via pip:

```bash
pip install numpy scipy networkx phylox-bio dendropy tqdm
```

## Class Architecture Overview
```commandline
=================== Data Preprocessing classes relationship diagram ===============

  ┌────────────────┐ (passed to)  ┌────────────────────────┐
  │ SpeciesNetwork │─────────────►│ DisplayedTreeExtractor │
  └───────┬────────┘              └──────────┬─────────────┘
          │                                  │
          │ (network)                        │ (major_tree)
          ▼                                  ▼
  ┌────────────────────────────────────────────────────────┐
  │                QuartetFeatureExtractor                 │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             │ (add seq_data row indices)
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │                  QuartetFeaturePairer                  │
  └──────────────────────────┬─────────────────────────────┘
                             │
  ┌───────────────────────┐  │
  │ SequenceDataProcessor │  │
  └──────────┬────────────┘  │
             │               │
             │ (id_code,n_D) │
             ▼               ▼
  ┌────────────────────────────────────────────────────────┐
  │                   SitePatternCounter                   │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             │ (outputs data-classes: FullSitePatterns, QuartetData)
                             │
============================ │ === Data classes relationship diagram =================================
                             │
                             │    ┌──────────────┐    ┌──────────────┐
                             │    │ AsymmQuartet │    │ SymmQuartet  │
                             │    └──────┬───────┘    └──────┬───────┘
                             │           │  (subclasses of)  │
                             │           └──────────┬────────┘
                             │                      │
                             │                      ▼
                             │            ┌────────────────────┐
                             │            │ QuartetTreeTopology│
                             │            └─────────┬──────────┘
                             │                      │
                             │                 (is part of)
                             │                      │
                             │                      ▼
                             │            ┌────────────────────┐
                             │            │ QuartetTreeFeature │
                             │            └─────────┬──────────┘
                             │                      │
                             │                (subclass of)
                             │                      │
                             │                      ▼
                             │            ┌────────────────────┐
                             │            │   QuartetFeature   │
                             │            └─────────┬──────────┘
                             │                      │
                             │                (subclass of)
                             │                      │
                             │                      ▼
                             │            ┌────────────────────┐
                             │            │PairedQuartetFeature│
                             │            └─────────┬──────────┘
                             │                      │ (Encapsulated by SitePatternCounter)
                             │                      │
                             │                      │
                             │                      │
                             ├────────────────┐     │
                             ▼                ▼     ▼
          ┌─────────────────────┐    ┌──────────────────┐      ┌────────────────┐  ┌────────────────┐
          │  FullSitePatterns   │    │   QuartetData    │      │ TreeParameters │  │ GammaParameters│
          └─────────────┬───────┘    └──┬───┬───────┬───┘      └───────┬────────┘  └───────┬────────┘
                        │               │   │       │                  │  (subclasses of)  │
                        │               │   │       │                  └──────────┬────────┘
                        │               │   │       │                             │
                        │               │   │       │                             ▼
                        │               │   │       │                  ┌────────────────────┐
                        │ ┌─────────────┘   │       │                  │ NetworkParameters  │
                        │ │                 │       │                  └──────────────┬─────┘
                        │ │ ┌───────────────┘       └───────────────────────┐         │
                        │ │ │                                               │         │
======================= │ │ │ ==== Utility classes relationship diagram === │ ======= │ ==========================
                        │ │ │                                               │         │
                        │ │ │  ┌──────────────────────┐                     ▼         ▼
                        │ │ │  │ ParameterTransformer │────┐          ┌────────────────────────┐
                        │ │ │  └──────────────────────┘    │ ┌────────│get_network_comp_log_lik│
                        │ │ │  ┌───────────────────────┐   │ │        └──────────┬─────────────┘
                        │ │ │  │ TauPriorTauConstraint │─┐ │ │                   │
                        │ │ │  └───────────────────────┘ │ │ │                   │
                        │ │ │  ┌──────────────┐          │ │ │                   │
                        │ │ │  │ MOMEstimator │────────┐ │ │ │                   │
                        │ │ │  └──────────────┘        │ │ │ │                   │
                        │ │ │                          ▼ ▼ ▼ ▼                   │
                        │ │ │   ┌───────────────────────────────┐                │
                        │ │ └──►│           Optimizer           │                │
                        │ │     └─────────────┬─────────────────┘                │
                        │ │                   │ (MCLE Parameters)                │
                        ▼ ▼                   ▼                                  │
                  ┌───────────────────────────────────────────────┐              │
                  │       CurvatureAdjustmentCalculator           │              │
                  └───────────────────────────┬───────────────────┘              │
                                              │                                  │
             (Curvature Adjustment Matrix C)  │                                  │
                                              ▼                                  ▼
                ┌──────────────────────────────────────────────────────────────────┐
                │                      MCMC Sampling Pipeline                      │
                │   ┌──────────────────┐  ┌─────────────┐  ┌───────────────────┐   │
                │   │  ProposalKernel  │  │ MCMCPriors  │─►│    MCMCSampler    │   │
                │   └────────┬─────────┘  └─────────────┘  └─────────▲─────────┘   │
                │            └───────────────────────────────────────┘             │
                └──────────────────────────────────────────────────────────────────┘
```

## Quickstart & Object-Oriented Workflow

Below is an end-to-end example demonstrating how to parse alignments, perform maximum composite likelihood estimation (MCLE), calculate curvature adjustment, and run MCMC sampling:

```python
import phylox
import dendropy
from Network_functions_v02 import (
    SpeciesNetwork,
    SitePatternCounter,
    Optimizer,
    CurvatureAdjustmentCalculator,
    MCMCSampler
)

# 1. Initialize species network and major tree wrappers
raw_net = phylox.DiNetwork(newick="((((A,B),(C,D)),(E,F#H1)),(#H1,(((G,H),I#H2),(#H2,J))));")
raw_major = phylox.DiNetwork(newick="((((A,B),(C,D)),E),(F,((I,J),(G,H))));")

network = SpeciesNetwork(raw_net)
major_tree = SpeciesNetwork(raw_major)

# 2. Load sequence alignment (DendroPy DnaCharacterMatrix)
seq_data = dendropy.DnaCharacterMatrix.get(path="sample_n10h2.nex", schema="nexus")

# 3. Parse alignment and compute quartet pattern counts & transformation matrices
counter = SitePatternCounter(
    network=network,
    major_tree=major_tree,
    seq_data=seq_data,
    skip_gap=True,
    skip_missing=True
)

all_quartet_data, full_site_pattern = counter.get_parsed_data_net()
compressed_quartet_data = counter.to_compressed_quartet_data(all_quartet_data)

# 4. Compute Maximum Composite Likelihood Estimates (MCLE)
optimizer = Optimizer(network=network, all_quartet_data=compressed_quartet_data)
mcle_params, max_log_lik = optimizer.get_MCLE_parameters()

print(f"Maximum Composite Log-Likelihood: {max_log_lik:.4f}")
print(f"Estimated Parameters: {mcle_params}")

# 5. Compute Curvature Adjustment Matrix (C)
calc = CurvatureAdjustmentCalculator(
    mcle_net_params=mcle_params,
    all_quartet_data=all_quartet_data,
    full_site_pattern=full_site_pattern
)
curv_adj_matrix = calc.get_curvAdjust_matrix()

# 6. Perform Curvature-Adjusted MCMC Sampling
sampler = MCMCSampler(network=network, compressed_quartet_data=compressed_quartet_data)
mcmc_chain = sampler.run(
    nsample=3000,
    thin=100,
    step_width=[5.2e-5, 4e-4, 6.3e-3],  # [tau, theta, gamma]
    curv_adj=curv_adj_matrix,
    kernel_type="normal",
    mcle=mcle_params
)

print(f"MCMC Chain shape: {mcmc_chain.shape}")
```

---

## Key Classes & Data structures
| Class Name | Primary Responsibility |
| :--- | :--- |
| `SpeciesNetwork` | High-level phylogenetic network wrapper providing topology manipulation and string exports. |
| `QuartetTreeTopology` | Abstract base class defining site-pattern probability and MOM estimation for quartet topologies. |
| `AsymmQuartet` / `SymmQuartet` | Concrete strategy implementations for asymmetric $(A,(B,(C,D)))$ and symmetric $((A,B),(C,D))$ quartets. |
| `NetworkParameters` | Composite container encapsulating `TreeParameters` ($\boldsymbol{\tau}, \theta$) and `GammaParameters` ($\boldsymbol{\gamma}$) with arithmetic operator overloads. |
| `SitePatternCounter` | Projects dataset site patterns onto sub-quartet matrices ($n^Q = E \cdot n^D$). |
| `Optimizer` | Runs unconstrained and constrained multi-start BFGS parameter optimization. |
| `CurvatureAdjustmentCalculator` | Derives Godambe sensitivity ($H$), variability ($J$), and curvature correction ($C$) matrices. |
| `MCMCSampler` | Metropolis-within-Gibbs sampler executing raw or curvature-adjusted MCMC iterations. |

## Citation & References

This toolkit builds upon theoretical foundations and software frameworks from:

* **PhyloX:** Framework for phylogenetic network topology manipulation in Python.
* **Ribatet et al. (2012):** Curvature adjustments for using composite likelihood in MCMC.
* **Chifman & Kubatko (2015):** Analytic site pattern probabilities under the Multispecies Coalescent (MSC) model.
* **Kong et al. (2025):** Parameter transformations, and species network composite likelihood under Multispecies Network Coalescent (MSNC) models.
