import dendropy
from phylox import DiNetwork
from Network_functions_v02 import (SpeciesNetwork, NetworkParameters, SitePatternCounter, get_network_comp_log_lik,
                                   printQuartet, Optimizer, CurvatureAdjustmentCalculator)
import numpy as np
np.set_printoptions(suppress=True, linewidth=np.nan)

network_string = "((((A,B),(C,D)),(E,F#H1)),(#H1,(((G,H),I#H2),(#H2,J))));"
major_tree_string = "((((A,B),(C,D)),E),(F,((I,J),(G,H))));"

network = SpeciesNetwork(DiNetwork.from_newick(network_string))
major_tree = SpeciesNetwork(DiNetwork.from_newick(major_tree_string))

seq_data = dendropy.DnaCharacterMatrix.get(path="sample_n10h2.nex", schema="nexus")

site_pattern_counter = SitePatternCounter(network, major_tree, seq_data)
all_quartet_data, full_site_pattern = site_pattern_counter.get_parsed_data_net()

optim = Optimizer(network, all_quartet_data)
mcle, best_cl = optim.get_MCLE_parameters()

# mcle = NetworkParameters.from_vectors([0.03010305, 0.01985305, 0.01494486, 0.00991589, 0.00991164,
#                                                      0.01424298, 0.0252182 , 0.01993655, 0.01454349, 0.00500243,
#                                                      0.00545873, 0.00500916], [0.18583204, 0.38333605])

get_network_comp_log_lik(all_quartet_data, mcle)

c_mat_calculate = CurvatureAdjustmentCalculator(mcle, all_quartet_data, full_site_pattern)
c_mat = c_mat_calculate.get_curvAdjust_matrix()