import dendropy, sys
from phylox import DiNetwork
# Replace this with the actual path to your directory
sys.path.append("/Users/shawn/simulation/python/Species_Network_Functions/oop_ver")
from Network_functions_v02 import (SpeciesNetwork, NetworkParameters, SitePatternCounter, Optimizer, CurvatureAdjustmentCalculator,
                                   MCMCSampler, get_network_comp_log_lik, printQuartet)
import numpy as np
np.set_printoptions(suppress=True, linewidth=np.nan)


# 1. Initialize species network and major tree wrappers
network_string = "((((A,B),(C,D)),(E,F#H1)),(#H1,(((G,H),I#H2),(#H2,J))));"
major_tree_string = "((((A,B),(C,D)),E),(F,((I,J),(G,H))));"

network = SpeciesNetwork(DiNetwork.from_newick(network_string))
major_tree = SpeciesNetwork(DiNetwork.from_newick(major_tree_string))

# 2. Load sequence alignment (DendroPy DnaCharacterMatrix)
seq_data = dendropy.DnaCharacterMatrix.get(path="/Users/shawn/simulation/python/Species_Network_Functions/oop_ver/sample_n10h2.nex", schema="nexus")

# 3. Parse alignment and compute quartet pattern counts & transformation matrices
site_pattern_counter = SitePatternCounter(network, major_tree, seq_data)
all_quartet_data, full_site_pattern = site_pattern_counter.get_parsed_data_net()
compressed_quartet_data = site_pattern_counter.to_compressed_quartet_data(all_quartet_data)


# 4. Compute Maximum Composite Likelihood Estimates (MCLE)
optim = Optimizer(network, compressed_quartet_data)
mcle, best_cl = optim.get_MCLE_parameters()
# mcle = NetworkParameters.from_vectors([0.03010305, 0.01985305, 0.01494486, 0.00991589, 0.00991164,
#                                                      0.01424298, 0.0252182 , 0.01993655, 0.01454349, 0.00500243,
#                                                      0.00545873, 0.00500916], [0.18583204, 0.38333605])
# get_network_comp_log_lik(all_quartet_data, mcle)


# 5. Compute Curvature Adjustment Matrix (C)
c_mat_calculate = CurvatureAdjustmentCalculator(mcle, all_quartet_data, full_site_pattern)
c_mat = c_mat_calculate.get_curvAdjust_matrix()

# 6. Perform Curvature-Adjusted MCMC Sampling
sampler = MCMCSampler(
    network=network,
    compressed_quartet_data=compressed_quartet_data
)

curv_samples = sampler.run(
    nsample=3000,
    thin=100,
    step_width=[5.2e-5, 4e-4, 6.3e-3], # [tau, theta, gamma]
    curv_adj=c_mat,
    mcle=mcle
)

