import os, dendropy, time, glob
from phylox import DiNetwork
from Network_functions_v02 import SpeciesNetwork, NetworkParameters



network_string = "((((A,B),(C,D)),(E,F#H1)),(#H1,(((G,H),I#H2),(#H2,J))));"
major_tree_string = "((((A,B),(C,D)),E),(F,((I,J),(G,H))));"

network = SpeciesNetwork(DiNetwork.from_newick(network_string))
major_tree = SpeciesNetwork(DiNetwork.from_newick(major_tree_string))


seq_data = dendropy.DnaCharacterMatrix.get(file=open("sample_n10h2.nex"), schema="nexus")

all_quartet_features = network.get_all_quartet_features(major_tree)

net_params = NetworkParameters.from_vectors([0.03010305, 0.01985305, 0.01494486, 0.00991589, 0.00991164,
                                                     0.01424298, 0.0252182 , 0.01993655, 0.01454349, 0.00500243,
                                                     0.00545873, 0.00500916], [0.18583204, 0.38333605])

all_quartet_features[0].get_true_probs(net_params)