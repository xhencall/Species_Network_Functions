# Species_Network_Functions

An extended framework built on **PhyloX** for species network manipulation, composite likelihood computation, curvature adjustment matrix calculation, and Bayesian parameter inference under the **Multispecies Network Coalescent (MSNC)** model.

---

## Key Features

* **Network & Quartet Manipulation**
  * Extract displayed trees and weighted displayed tree mappings ($\gamma$-IDs) from phylogenetic networks.
  * Sub-quartet extraction, ladderization, and classification into symmetric and asymmetric quartet topologies.
* **Alignment & Site Pattern Processing**
  * Parse DNA alignments with full support for IUPAC ambiguity codes and multi-individual-per-species partitions.
  * Vectorized mapping matrices ($E_{\text{mat}}$) linking global site pattern counts ($n^D$) to sub-quartet frequencies ($n^Q$).
* **Composite Likelihood Estimation (MCLE)**
  * Analytic calculation of true 15-category site pattern probabilities ($p^Q$) under the Jukes-Cantor (JC69) substitution model.
  * Unconstrained and constrained multi-start BFGS optimization with parameter-space transformations ($\boldsymbol{\tau}, \theta, \boldsymbol{\gamma}$).
* **Godambe Information & Curvature Adjustment**
  * Derive full Score functions ($U_c$), Variability ($J$), Sensitivity ($H$), and Curvature-Adjustment ($C$) matrices to correct composite likelihood surfaces.
* **Bayesian MCMC Inference**
  * Metropolis-within-Gibbs sampling for joint posterior inference of speciation times ($\boldsymbol{\tau}$), effective population size ($\theta$), and inheritance probabilities ($\boldsymbol{\gamma}$).
  * Supports both raw Composite Likelihood MCMC and **Curvature-Adjusted MCMC** (adjCL).
* **(On going project) Hypothesis Testing & Simulation**
  * Modified Composite Likelihood Ratio Tests (**mCLRT**) for hybridization hypotheses ($H_0: \gamma_i = 0$).
  * Automated NEXUS/PAUP* script generation for multilocus DNA data simulations.

---

## Dependencies

* **Python 3.8+**
* **Core Libraries:** `numpy`, `scipy`, `networkx`, `phylox`, `dendropy`, `tqdm`

Install the main dependencies via pip:

```bash
pip install numpy scipy networkx phylox-bio dendropy tqdm
```

---

## Module Overview

The toolkit provides a suite of utilities organized into six core functional layers:

* **Network Tools** (`get_displayed_trees`, `ladderize_network`, `retain_taxa_with_labels`): Functions for parsing phylogenetic network topologies, generating displayed trees, ladderizing DAGs, and pruning leaf branches.
* **Site Patterns** (`get_n_D_hasAmbiguityCode`, `get_quartet_SitePatternCount_MapMatrix`): Utilities for parsing alignments, handling IUPAC ambiguity codes, and computing transformation mapping matrices ($E_{\text{mat}}$) between dataset pattern counts ($n^D$) and sub-quartet pattern counts ($n^Q$).
* **Composite Likelihood** (`getTrueProbsQuartet`, `SpeciesNetwork_CompLogLik`): Analytic functions for calculating exact 15-category site pattern probabilities for sub-quartets under mutation and coalescent units, as well as evaluating full species network composite log-likelihoods.
* **Optimization** (`get_MCLE_parameters`, `parameter_transform`, `parameter_backtransform`): Algorithms for finding Maximum Composite Likelihood Estimates (**MCLE**) using constrained/unconstrained multi-start BFGS optimization across transformed parameter spaces.
* **Information Matrix** (`get_Vari_Sens_Mat`, `get_curvAdjust_matrix`): Analytic evaluation of the score function ($U_c$), variability matrix ($J$), sensitivity matrix ($H$), and Godambe curvature-adjustment matrix ($C$).
* **MCMC Inference & Testing** (`MCMC_rawCompLik`, `MCMC_curvAdjCompLik`, `modified_CompLik_ratio_stat`): Metropolis-within-Gibbs samplers for raw and curvature-adjusted composite likelihood MCMC, alongside modified Composite Likelihood Ratio Tests (**mCLRT**) for hybridization hypothesis testing.

---

## Quickstart & Basic Usage

Below is a conceptual workflow illustrating how to estimate network parameters and run MCMC sampling from a PhyloX network and alignment dataset:

```python
import phylox
import dendropy
from Network_functions_v01 import (
    get_zippedData_net,
    get_zippedData_net_reduced,
    get_MCLE_parameters,
    get_curvAdjust_matrix,
    MCMC_curvAdjCompLik
)

# 1. Load species network and major tree
net = phylox.DiNetwork(newick="((A,(B)#H1),((#H1,C),D));")
major_tree = phylox.DiNetwork(newick="((A,B),(C,D));")

# 2. Load DNA alignment (Dendropy CharacterMatrix)
alignment = dendropy.DNACharacterMatrix.get(path="alignment.nex", schema="nexus")

# 3. Extract quartet site pattern data and mapping matrices
zipped_data, all_E_mat, n_D = get_zippedData_net(alignment, net, major_tree)
zipped_data_reduce, _, _ = get_zippedData_net_reduced(alignment, net, major_tree)

# 4. Find Maximum Composite Likelihood Estimates (MCLE)
mcle, loglik = get_MCLE_parameters(zipped_data, net, MultiStart=10)
print(f"MCLE Parameters: {mcle}")
print(f"Composite Log-Likelihood: {loglik}")

# 5. Compute Curvature Adjustment Matrix (C)
curv_adj = get_curvAdjust_matrix(mcle, zipped_data, all_E_mat, n_D)

# 6. Run Curvature-Adjusted MCMC Sampling
mcmc_samples = MCMC_curvAdjCompLik(
    zipped_data_net=zipped_data,
    zipped_data_net_reduce=zipped_data_reduce,
    phylox_network=net,
    nsample=5000,
    thin=10,
    step_width=[0.01, 0.001, 0.05], # tau, theta, gamma
    curvAdj=curv_adj,
    MCLE=mcle
)
```

---

## Citation & References

This toolkit builds upon theoretical foundations and software frameworks from:

* **PhyloX:** Framework for phylogenetic network topology manipulation in Python.
* **Ribatet et al. (2012):**Curvature adjustments for using composite likelihood in MCMC.
* **Chifman & Kubatko (2015):** Analytic site pattern probabilities under the Multispecies Coalescent (MSC) model.
* **Chen et al. (2018):** Composite likelihood ratio testing and profile composite likelihood inference on phylogenetic networks.
* **Kong et al. (2025):** Parameter transformations, and species network composite likelihood under Multispecies Network Coalescent (MSNC) models.
