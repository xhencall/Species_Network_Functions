# Press Control + Shift + X to execute it or replace it with your code.
# This version uses object-oriented programing

# To fully display matrix, use this code below
# noinspection PyTypeChecker
import numpy as np
# np.set_printoptions(suppress=True, linewidth=np.nan)

############################################
## Tools for phylox network manipulation  ##
############################################
import re, phylox, dendropy, itertools, warnings
import networkx as nx
from phylox.constants import LABEL_ATTR
from phylox import suppress_node
from dataclasses import dataclass
from itertools import combinations, product
from abc import ABC, abstractmethod

############################################
## Class for phylox network manipulation  ##
############################################

class SpeciesNetwork:
    """Class of species network for manipulating PhyloX networks."""
    def __init__(self, phylox_network: "phylox.dinetwork.DiNetwork", is_labeled: bool = False) -> None:
        self.phylox_network = phylox_network
        self.is_labeled = is_labeled

    @property
    def num_taxa(self):
        """number of taxa"""
        return len(self.leaves)

    @property
    def num_retic(self):
        """number of reticulations"""
        return len(self.reticulations)

    @property
    def num_tau(self):
        """total number of tau parameters"""
        return self.num_taxa + self.num_retic - 1

    def __getattr__(self, name: str):
        """
        If an attribute or method isn't found on SpeciesNetwork, Python automatically redirects the call to
        self.phylox_network.
        For example, we set my_net = SpeciesNetwork(phylox_network).
            If you type my_net.nodes, Python sets name = "nodes".
            If you type my_net.successors(node), Python sets name = "successors".
            If you type my_net.leaves, Python sets name = "leaves".
        """
        return getattr(self.phylox_network, name)

    def copy(self):
        """Returns a deep copy of the SpeciesNetwork instance."""
        return SpeciesNetwork(self.phylox_network.copy(), is_labeled=self.is_labeled)
    def __copy__(self):
        return self.copy()

    ##----- Manipulate labels in nodes on network -----##
    def write_label_in_node(self, label: str, node: int) -> None:
        self.nodes[node][LABEL_ATTR] = label
        self.is_labeled = True

    def erase_label_in_node(self, node: int) -> None:
        del self.nodes[node][LABEL_ATTR]

    def erase_all_node_labels(self):
        """
        Remove labels from all internal nodes of a PhyloX network. A node is considered internal if its out-degree > 0
        (i.e., it is not a leaf_node).
        """
        # Internal nodes (out-degree > 0)
        internal_nodes = self.internal_nodes()

        for node in internal_nodes:
            # Remove label stored in LABEL_ATTR
            if LABEL_ATTR in self.nodes[node]:
                self.erase_label_in_node(node)

        self.is_labeled = False

    def label_speciation_time_idx(self) -> None:
        """
        Reticulation event has two different parental ages.
        Label speciation time indices by preorder traversal of internal nodes.
        Nodes that are predecessors of the same reticulation share the same label.
        """
        # Internal nodes (out-degree > 0)
        internal_nodes = self.internal_nodes()

        label_index = self.num_tau  # total number of tau parameters

        for node in internal_nodes:
            self.write_label_in_node(label_index, node)
            label_index -= 1

        self.is_labeled = True

    def get_label_from_node(self, node: int):
        try:
            return self.nodes[node][LABEL_ATTR]
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Failed to get label from node '{node}'. ") from e

    def get_node_from_label(self, label: str):
        try:
            return self.labels[label][0]
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Failed to get node from label '{label}'. ") from e

    def get_taxa_labels(self):
        """
        Extract taxa labels from a PhyloX network in the order they appear in the network's Newick string.
        Reticulation suffixes (#R0, #H1, etc.) are removed, and duplicates are removed while preserving order.
        """
        newick_str = self.newick().strip()

        # Regex pattern:
        # [A-Za-z0-9_.-]   →   character class. Matches one character from the set inside.
        #       A-Za-z → any ASCII letter (upper or lower).
        # 		0-9 → any digit.
        # 		_.- → underscore _, dot ., or hyphen -.
        # +   →   quantifier. So [A-Za-z0-9_.-]+ matches a run of one or more characters that are in character class.
        # (?=...)   →   positive lookahead. It asserts that what follows the matched characters must match the ...,
        #               but it does not consume those following characters (they remain in the string for subsequent
        #               matching). Using a lookahead ensures we stop where we want (right before # or , or ) ), without
        #               including that separator in the match.
        #       A-Za-z → any ASCII letter (upper or lower).
        #		# → match a literal # after the label (covers reticulation suffixes like 5#R0).
        # 	    [\),] → a small character class matching either ) or , . Inside a [...] the ) and , are literal
        #               characters; \) is fine but not strictly necessary.
        #

        pattern = r"[A-Za-z0-9_.-]+(?=#|[\),])"

        tokens = re.findall(pattern, newick_str)

        # Remove duplicates but preserve order
        seen = set()
        ordered_unique = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                ordered_unique.append(t)

        # Remove elements that are not in self.labels
        taxa_labels = list(self.labels)
        ordered_labels = [l for l in ordered_unique if l in taxa_labels]

        return ordered_labels

    def get_sister_node_label(self, node_label: str):
        """Given a node label, find its sister node label where a sister node shares the same parent with the node."""
        # Find the node associated with the given label
        node = self.get_node_from_label(node_label)

        # Get its parent
        parent = next(self.predecessors(node), None)
        if parent is None:
            raise KeyError(f"Taxon {node_label} has no parent (it may be a root).")

        # Find its sister (any other child of the parent)
        sister = None
        for child in self.successors(parent):
            if child != node:
                sister = child
                break

        if sister is None:
            raise KeyError(
                f"Taxon {node_label} has no sister taxa under the same parent."
            )

        # Return sister label (will raise KeyError automatically if no label)
        return self.get_label_from_node(sister)

    def get_leaf_labels_below_node(self, node: int):
        """Return a set of all leaf_node labels descended from a given node."""
        leaf_labels = set()
        stack = [node]
        while stack:
            node_list = [stack.pop()]
            for curr_node in node_list:
                children = list(self.successors(curr_node))
                if not children:
                    try:
                        leaf_labels.add(self.get_label_from_node(curr_node))
                    except KeyError:
                        pass
                else:
                    stack.extend(children)
        return leaf_labels

    def internal_nodes(self):
        """Get internal nodes (out-degree > 0)"""
        return [node for node in self.nodes if self.out_degree(node) > 0]

    def get_parameter_idx(self):
        """
        Extracts the speciation time indices (param_idx) of the labeled network.
        """
        if not self.is_labeled:
            warnings.warn(
                f"The input network {self.get_taxa_labels()} is not labeled. Use .label_speciation_time_idx() to label the network."
            )
        # If the out-degree of the root node is one, do not include it as speciation_nodes
        root = list(self.roots)[0]
        if self.out_degree(root) == 1:
            # If root is unary, speciation_nodes = [all degree-3 nodes]
            speciation_nodes = [node for node in self.nodes if self.degree(node) == 3]
        else:
            # If root is binary, speciation_nodes = root + [all degree-3 nodes]
            speciation_nodes = [root] + [node for node in self.nodes if self.degree(node) == 3]

        # Get the speciation time index labeled on speciation_nodes by pre-order traversal
        try:
            return np.array([self.get_label_from_node(node) for node in speciation_nodes], dtype=int)
        except (KeyError, AttributeError) as e:
            raise ValueError(
                f"Failed to get speciation time indices (param_idx) from input network. "
            ) from e


    ##----- Manipulate network structures -----##
    def suppress_all_degree2_nodes(self) -> None:
        """Suppress all degree-2 nodes (in-degree 1 and out-degree 1) after edge removal."""
        degree2_nodes = [node for node in self.nodes
                         if self.in_degree(node) == 1 and
                            self.out_degree(node) == 1]
        for node in degree2_nodes:
            suppress_node(self.phylox_network, node)

    def get_need_remove_nodes(self, node: int):
        """Iteratively tracing backward in time to look for nodes that need to be removed"""
        need_remove = [node]
        while True:
            # Get parent_node
            parent = next(self.predecessors(node), None)

            # Case 1: No parent → node is root → nothing should be removed
            if parent is None:
                return []

            # Compute parent's number of children once
            out_degree = self.out_degree(parent)

            # Case 2: Parent is branching (2 or more children) → stop
            if out_degree >= 2:
                return need_remove

            # Case 3: Parent is unary → keep tracing upward
            need_remove.append(parent)
            node = parent

    def prune_leaf_branch(self, leaf_node: int) -> None:
        """Remove the leaf_node node and the selected edges depending on the leaf_node type: (regular, single-child)"""
        need_remove = self.get_need_remove_nodes(leaf_node)
        self.remove_nodes_from(need_remove)

    def get_quartet_with_taxa_labels(self, taxa_labels: list[str]) -> "SpeciesNetwork":
        """
        Using PhyloX features to find a subtree containing only taxa_labels.
        """
        # Work on a copy so we don't modify the original
        pruned_tree = self.copy()

        # First, we find remove_leaves
        all_leaves = list(pruned_tree.leaves)
        retain_leaves = [pruned_tree.get_node_from_label(label) for label in taxa_labels]
        remove_leaves = [leaf for leaf in all_leaves if leaf not in retain_leaves]

        # Then, we prune those remove_leaves.
        for leaf in remove_leaves:
            pruned_tree.prune_leaf_branch(leaf)

        return pruned_tree

    def prune_hybrid_branch(self, in_edge: tuple[int, int]) -> None:
        """Removes the edge but intentionally delays suppressing the node."""
        parent, child = in_edge
        self.remove_edge(parent, child)


    def is_asymmetric_quartet(self):
        """Returns True if quartet_tree(self) is asymmetric quartet and False otherwise."""
        # Create a copy wrapped in a SpeciesNetwork instance so we don't modify the original
        quartet = self.copy()

        # 1. Suppress all degree-2 nodes FIRST
        quartet.suppress_all_degree2_nodes()

        # 2. Find the effective root safely AFTER the graph is simplified
        root = list(quartet.roots)[0]

        # If the root is a unary node (out-degree 1), its child is now guaranteed
        # to be the first branching node because all intermediate degree-2 nodes are gone.
        if quartet.out_degree(root) == 1:
            root = next(quartet.successors(root))

        # Return True if any child of root is a leaf_node
        is_asymm = any(child in quartet.leaves
                      for child in quartet.successors(root))
        return is_asymm

    def ladderize(self, ascending=False) -> None:
        """
        Ladderizes a PhyloX DiNetwork based on the number of descendant leaves.

        Parameters:
        - network (phylox.DiNetwork): The input phylogenetic network.
        - ascending (bool): If False (default), children with more descendant leaves
                            are ordered first. If True, children with fewer come first.
        """

        if not nx.is_directed_acyclic_graph(self.phylox_network):
            raise ValueError("The network must be a directed acyclic graph (DAG).")

        # 1. Identify all leaf_node nodes (nodes with an out-degree of 0)
        leaves = self.leaves  # {n for n, d in self.out_degree() if d == 0}

        # 2. Compute the set of descendant leaves for every node
        # A reverse topological sort ensures we process children before their parents
        descendant_leaves = {n: set() for n in self.nodes()}
        for leaf in leaves:
            descendant_leaves[leaf].add(leaf)

        for node in reversed(list(nx.topological_sort(self.phylox_network))):
            for child in self.successors(node):
                descendant_leaves[node].update(descendant_leaves[child])

        # Calculate the "weight" (number of unique descendant leaves) for each node
        node_weight = {n: len(leaves_set) for n, leaves_set in descendant_leaves.items()}

        # 3. Create a new PhyloX network to hold the sorted structure
        ladderized_net = self.phylox_network.__class__()

        # Copy all nodes of self and their index labels to ladderized_net
        ladderized_net.add_nodes_from(self.nodes(data=True))

        # 4. Add edges back in sorted order
        for node in self.nodes():
            children = list(self.successors(node))

            if ascending:
                # Sort ascending by weight, breaking ties by node ID
                sorted_children = sorted(children, key=lambda c: (node_weight[c], str(c)))
            else:
                # Sort descending by weight, breaking ties by node ID
                sorted_children = sorted(children, key=lambda c: (-node_weight[c], str(c)))

            # Insert the edges into the ladderized_net
            for child in sorted_children:
                ladderized_net.add_edge(node, child)

        self.phylox_network = ladderized_net

    def sorting_asymmetric_quartet(self):
        """
        Get the taxa permutation to ladderize an input quartet_tree into a sorted
        asymmetric quartet: (A,((B,C),D)) -> (A,(D,(B,C))).
        We need this taxa permutation to get the relationship of site patterns for the
        computation of site pattern probabilities.
        """
        unsorted_taxa_order = self.get_taxa_labels()

        # Sort the asymmetric quartet in ascending order using 'ladderize'
        sorted_tree = self.copy()
        sorted_tree.ladderize(ascending=True)  # e.g. Sort (A,((B,C),D)) into (A,(D,(B,C)))

        sorted_taxa_order = sorted_tree.get_taxa_labels()

        # Return the taxa permutation of input asymmetric_quart
        return [unsorted_taxa_order.index(x) for x in sorted_taxa_order]

    def print_tree(self):
        """
        Using DendroPy graphing tool, print a PhyloX self in ASCII format and Newick format.
        """
        clone_tree = self.copy()
        clone_tree.erase_all_node_labels()
        dendropy_tree = dendropy.Tree.get(data=clone_tree.newick(), schema="newick", rooting="default-rooted")
        print(dendropy_tree.as_string(schema="newick"))
        print(dendropy_tree.as_ascii_plot())

    def newick(self):
        """
        Get a simple Newick string from a labeled network, ignoring all internal node labels.
        Usually, we can use command self.newick() to print Newick format. This function exists because
        labeled_network.newick() will report error.
        """

        # --- Step 1: Identify the root (taking care of unary root)
        root = list(self.roots)[0]
        if self.out_degree(root) == 1:
            root = next(self.successors(root))

        # --- Step 2: Recursive function
        def node_to_newick(node):
            # Leaf: return its taxon label
            if node in self.leaves:
                return str(self.nodes[node]["label"])

            # Internal node: process children
            children = list(self.successors(node))
            children_strings = [node_to_newick(c) for c in children]

            # No internal labels included → return "(child1,child2)"
            return "(" + ",".join(children_strings) + ")"

        # --- Step 3: Build final Newick
        return node_to_newick(root) + ";"

    def get_displayed_trees(self):
        """
        Given a phylogenetic network from PhyloX, returns a list of all
        displayed trees with one parental choice at each reticulation.
        """
        # Identify all hybrid nodes (reticulations)
        reticulations = sorted(list(self.reticulations), key=self.get_retic_number)
        h = len(reticulations)

        # Base case: No reticulations → return the network as the only displayed tree
        if h == 0:
            return [self.copy()]

        all_disp_trees = []

        # Enumerate all 2^h parental choices (0 or 1 for which in-edge to remove)
        for bits in itertools.product([0, 1], repeat=h):
            # Start with a fresh copy of the network for this specific displayed tree
            disp_tree = self.copy()

            for i, bit in enumerate(bits):
                retic_node = reticulations[i]

                # Extract the incoming edges to the reticulation node.
                # Converting to a list allows us to easily grab index 0 or 1.
                in_edges = list(disp_tree.in_edges(retic_node))
                edge_to_remove = in_edges[bit]
                parent_node = edge_to_remove[0]

                # 1. Remove the chosen hybrid branch
                disp_tree.remove_edge(*edge_to_remove)

                # 2. Suppress the parent node safely (only if in=1 and out=1)
                if disp_tree.in_degree(parent_node) == 1 and disp_tree.out_degree(parent_node) == 1:
                    suppress_node(disp_tree.phylox_network, parent_node)

                # 3. Suppress the reticulation node safely (only if in=1 and out=1)
                if disp_tree.in_degree(retic_node) == 1 and disp_tree.out_degree(retic_node) == 1:
                    suppress_node(disp_tree.phylox_network, retic_node)

            all_disp_trees.append(disp_tree)

        return all_disp_trees


class DisplayedTreeExtractor:
    """Extracts displayed trees and gamma identifiers from a SpeciesNetwork."""

    def __init__(self, network: SpeciesNetwork, major_tree: "SpeciesNetwork"):
        self.network = network
        self.major_tree = major_tree

    @staticmethod
    def get_retic_number(node: int):
        """Helper function to extract the integer from labels like '#H1'. """
        match = re.search(r'\d+', str(node))
        return int(match.group()) if match else 0

    def get_displayed_tree_gamma_id(self):
        """
        Given a phylogenetic network from PhyloX: self, returns a list of (displayed_tree, gamma_id) pairs.
        The displayed_tree is a DiNetwork with one parental choice at each hybrid, and
        the gamma_id directs computation of weight of a displayed tree from gamma parameters:

        gamma_id=1 -> weight=gamma_1.
        gamma_id=-1 -> weight=1-gamma_1.
        gamma_id=[1,-2] -> weight=gamma_1*(1-gamma_2)
        """
        # label nodes on network with speciation time index
        labeled_network = self.network.copy()
        labeled_network.label_speciation_time_idx()

        # Grab the reticulations and explicitly SORT them by their numerical label
        reticulations = sorted(list(self.network.reticulations), key=self.get_retic_number)
        h = len(reticulations)

        # No reticulations → return network with weight 1
        if h == 0:
            return [(labeled_network, [])]

        all_disp_tree_gamma_id = []
        # Enumerate all 2^h parental trees (or displayed trees)
        for bits in itertools.product([0, 1], repeat=h):
            disp_tree = labeled_network.copy()
            gamma_id = []

            for i, bit in enumerate(bits):
                # Get the bit^th in-edge of the i^th reticulation
                # itertools.islice(iterable, start, stop[, step]) lets us “slice” an iterator to pull out
                # a portion of it efficiently, but without creating an entire list in memory.
                retic_node = reticulations[i]
                in_edge = next(itertools.islice(disp_tree.in_edges(retic_node), bit, None))

                # Remove the in-edge branch
                disp_tree.prune_hybrid_branch(in_edge)

                # Compare leaf_node labels below the parent node of the hybrid taxa
                parent_node_disp = next(disp_tree.predecessors(retic_node))
                leaf_labels_disp = disp_tree.get_leaf_labels_below_node(parent_node_disp)

                retic_label = labeled_network.get_label_from_node(retic_node)
                retic_node_majr = self.major_tree.get_node_from_label(retic_label)
                parent_node_majr = next(self.major_tree.predecessors(retic_node_majr))
                leaf_labels_majr = self.major_tree.get_leaf_labels_below_node(parent_node_majr)

                if leaf_labels_disp == leaf_labels_majr:
                    gamma_id.append(-(i+1)) # adjust for 1-based indexing in gamma_id
                else:
                    gamma_id.append(i+1)

            all_disp_tree_gamma_id.append((disp_tree, gamma_id))

        return all_disp_tree_gamma_id

    def get_newick_displayed_tree_gamma_id(self):
        """
        Given a phylogenetic network from PhyloX, returns a list of (Newick_string, gamma_id) pairs.
        All degree-2 nodes are suppressed and internal labels erased before generating the Newick string.

        gamma_id=1 -> weight=gamma_1.
        gamma_id=-1 -> weight=1-gamma_1.
        gamma_id=[1,-2] -> weight=gamma_1*(1-gamma_2)
        """
        all_weighted_trees = self.get_displayed_tree_gamma_id()
        all_newick_disp_tree_gamma_id = []

        for disp_tree, gamma_id in all_weighted_trees:
            # 1. Suppress the degree-2 nodes
            disp_tree.suppress_all_degree2_nodes()

            # 2. Erase internal labels so they don't pollute the Newick string
            disp_tree.erase_all_node_labels()

            # 3. Generate native Newick string
            newick_str = disp_tree.newick()

            all_newick_disp_tree_gamma_id.append((newick_str, gamma_id))

        return all_newick_disp_tree_gamma_id


class QuartetFeatureExtractor:
    """Extracts quartet features (param_idx, topology, taxa_perm, gamma_id) across all 4-taxa combinations."""

    def __init__(self, network: SpeciesNetwork, major_tree: "SpeciesNetwork"):
        self.network = network
        self.major_tree = major_tree
        # Composition: Instantiate extractor helper during initialization
        self.tree_extractor = DisplayedTreeExtractor(self.network, self.major_tree)

    ##----- Working with quartet features of quartet subnetworks -----##
    def get_all_quartet_features(self):
        """
        For each combination of 4 taxa, get the quartet features (parameter indices: param_idx, quartet types: is_asymm,
        taxa permutations: taxa_perm, and gamma id's: gamma_id) of the sub-quartets pulled from each displayed tree. If
        sub-quartets have the same param_idx for a combination of 4 taxa, we only save one quartet feature. Otherwise,
        we store all quartet features as a list of param_idx, of is_asymm, of taxa_perm, and of gamma_id that correspond
        to the combination of 4 taxa. These quartet features are used to compute the true probabilities of the quartet.

        We use PhyloX to label speciation time indices in preorder traversal.
        """
        # Get all possible combinations of four taxa labels
        taxa_labels = self.network.get_taxa_labels()
        four_taxa_combs = combinations(taxa_labels, 4)

        # Get all displayed trees with gamma_id's used to compute weight
        all_disp_tree_gamma_id = self.tree_extractor.get_displayed_tree_gamma_id()

        # Create empty lists
        quartet_features_list = []

        for taxa_comb in four_taxa_combs:
            # Use param_idx as dict key to deduplicates identical quartets and store other features in dict
            unique_tree_features = {}

            # Pre-convert the tuple to a list once per quartet
            taxa_list = list(taxa_comb)

            for disp_tree, gamma_id in all_disp_tree_gamma_id:
                # Prune the displayed tree to keep only the current set of four taxa
                labeled_quartet = disp_tree.get_quartet_with_taxa_labels(taxa_list)

                # Erase labels on internal nodes for quartet sorting
                quartet_network = labeled_quartet.copy()
                quartet_network.erase_all_node_labels()

                # Use param_idx as key to store other features
                param_idx = labeled_quartet.get_parameter_idx()
                dict_key = tuple(param_idx)
                is_asymm = AsymmQuartet() if labeled_quartet.is_asymmetric_quartet() else SymmQuartet()

                # Only sort the asymmetric quartet
                taxa_perm = quartet_network.sorting_asymmetric_quartet() if is_asymm else [0, 1, 2, 3]

                qt_feature = QuartetTreeFeature(
                    param_idx   =   param_idx,
                    topology    =   is_asymm,
                    taxa_perm   =   taxa_perm,
                    gamma_id    =   gamma_id
                )

                unique_tree_features[dict_key] = qt_feature

            # Create a structured QuartetFeature object
            q_feature = QuartetFeature(
                taxa = taxa_comb,
                tree_features = list(unique_tree_features.values())
            )
            quartet_features_list.append(q_feature)

        return quartet_features_list


class QuartetFeaturePairer:
    """Pairs DNA sequence data rows with corresponding quartet features."""

    def __init__(self,
                 network: SpeciesNetwork,
                 major_tree: "SpeciesNetwork",
                 seq_data: "dendropy.DnaCharacterMatrix",
                 imap_path: str | None = None):
        self.network = network
        self.major_tree = major_tree
        self.seq_data = seq_data
        self.imap_path = imap_path
        # Composition: Instantiate extractor helper during initialization
        self.feature_extractor = QuartetFeatureExtractor(self.network, self.major_tree)

    def get_taxa_partition(self):
        """Produce tax_partition from user input Imap file"""
        # Read the Imap file into a dictionary
        imap_dict = {}
        with open(self.imap_path, "r") as f:
            for line in f:
                old_name, species = line.strip().split()
                imap_dict.setdefault(species,[]).append(old_name)

        # Returns True when seq_taxon_name matches any one of the imap_names
        def matches(seq_taxon_name, imap_names):
            for name in imap_names:
                # Replace space with underscore if there is any
                if ' ' in seq_taxon_name:
                    seq_taxon_name = seq_taxon_name.replace(" ", "_")
                # Remove caret ^ before taxon name
                if name == seq_taxon_name.lstrip("^"):
                    return True
            return False

        taxa_partition = [
            # seq_data row indices of the taxa that contains the species name in the self
            [idx for idx, seq_taxon in enumerate(self.seq_data.taxon_namespace) if matches(seq_taxon.label, imap_dict[label])]
            for label in self.network.get_taxa_labels()
        ]

        return taxa_partition

    def pair_seq_data_rows_quartet_features(self):
        """
        Given seq_data and taxa_partition, we can select one individual per species to form a quartet matrix by
        choosing the rows (individuals) of seq_data. The choice of the four row indices is seq_data_rows.
        For each seq_data_rows, we pair it up with the corresponding quartet features (param_idx, is_asymm,
        taxa_perm, gamma_id).

        This function handles data with multiple individuals per species through "taxa_partition". For example,
        we have species tree ((E,B),((C,D),A)) and data = {A1,A2,B1,B2,C1,C2,D1,D2,E1,E2}, the taxa partition is
        [[8, 9], [2, 3], [4, 5], [6, 7], [0, 1]].
        """
        # Step 1: Taxa partitions.
        if self.imap_path is None:
            taxa_partition = [
                # seq_data row indices of the taxa that contains the species name in the network
                [idx for idx, taxon in enumerate(self.seq_data.taxon_namespace) if label in taxon.label]
                for label in self.network.get_taxa_labels()
            ]
        else:
            taxa_partition = self.get_taxa_partition()

        # Step 2: Get all one-individual-per-species mappings (row indices of seq_data) according to taxa_partition.
        one_indiv_per_species_maps = product(*taxa_partition)

        # Step 3: Get all possible quartet features from self
        all_quartet_features = self.feature_extractor.get_all_quartet_features()

        # Step 4: Pair up seq_data_rows with the corresponding quartet features (param_idx, is_asymm, taxa_perm, gamma_id)
        dict_paired_feature = {}
        for MAP in one_indiv_per_species_maps:
            # For each MAP in one_indiv_per_species_maps, enumerate all possible combinations of 4 rows of seq_data
            # (seq_data_rows).
            for seq_data_rows, q_feature in zip(combinations(MAP, 4), all_quartet_features):
                # Since seq_data_rows will appear twice, we use it as dict key to deduplicates identical quartets
                # and store quartet features.
                dict_key = tuple(seq_data_rows)
                if dict_key not in dict_paired_feature:
                    dict_paired_feature[dict_key] = PairedQuartetFeature(
                        seq_data_rows = np.array(seq_data_rows), quartet_feature = q_feature
                    )

        # Step 5: Convert aggregated dictionary to container object
        return list(dict_paired_feature.values())


###########################################################################
## Class for manipulating parameters (tree_parameters, gamma_parameters) ##
###########################################################################

@dataclass
class TreeParameters:
    """Encapsulates speciation time parameters (tau) and effective population size parameter (theta)."""
    values: np.ndarray  # Format: [tau_J, ..., tau_2, tau_1, theta]

    @property
    def tau(self):
        """Branch lengths / speciation times (all but last element)."""
        return np.asarray(self.values[:-1])

    @property
    def theta(self):
        """effective population size parameter (last element)."""
        return self.values[-1]

    @property
    def num_tau(self):
        """Total number of tau parameters."""
        return len(self.values) - 1

    def mut_to_coal(self):
        """Converts mutation units to coalescent units."""
        result = np.asarray(self.values, dtype=float).copy()
        # Perform operation on all but the last element, then keep the last element (theta)
        result[:-1] /= self.theta
        return result

    def coal_to_mut(self):
        """Converts coalescent units to mutation units."""
        result = np.asarray(self.values, dtype=float).copy()
        # Perform operation on all but the last element, then keep the last element (theta)
        result[:-1] *= self.theta
        return result

    def get_tau_theta(self, tau_idx: list[int]):
        """
        parameters = (tau_J,...,tau_2,tau_1,theta) where tau_J is root age.
        tau_idx are the speciation time indices of tau (1-based).
        Function returns [tau_i1, tau_i2, tau_i3, theta] in reversed order.
        """
        tau_idx_0base = self.num_tau - np.array(tau_idx)  # adjust for 0-based indexing of tau

        # Output in tau1, tau2, tau3 for convenience to input in getTrueProbs functions.
        return [self.tau[i] for i in tau_idx_0base[::-1]] + [self.theta]

@dataclass
class GammaParameters:
    """Encapsulates inheritance probabilities (gamma) and derivative calculations."""
    values: np.ndarray

    @property
    def num_retic(self):
        """Total number of gamma parameters."""
        return len(self.values)

    @staticmethod
    def remove_common_elements(gamma_id):
        """
        Remove gamma indices that appear in ALL rows of the input gamma_id (2D numpy array) to help computing gamma
        weights and taking derivatives with respect to gamma. Output is a cleaned numpy array of original length.
        """
        # Convert rows to sets
        row_sets = [set(row) for row in gamma_id]

        # Find elements present in ALL rows (set intersection)
        common = set.intersection(*row_sets)

        # Remove the common elements from each row
        cleaned = [list(set(row) - common) for row in gamma_id]

        # Convert back to numpy array
        return np.array(cleaned)

    def get_gamma_weight(self, gamma_id):
        """
        Get Gamma_t.
        gamma_id=[] -> 1
        gamma_id=1  -> gamma_1
        gamma_id=-1 -> 1 - gamma_1
        gamma_id=[1,-2] -> gamma_1 * (1 - gamma_2)
        """
        if isinstance(gamma_id, np.ndarray) and gamma_id.size == 0: # returns 1 if gamma_id is ndarray and empty
            return 1.0
        if isinstance(gamma_id, (int, np.integer)):     # Wrap an integer gamma_id to list
            gamma_id = [gamma_id]
        if len(gamma_id) == 0:                          # returns 1 if gamma_id is list and empty
            return 1.0

        Gamma_t = 1.0
        for idx in gamma_id:
            i = abs(idx) - 1  # convert to 0-based index
            g = self.values[i]
            Gamma_t *= g if idx > 0 else (1 - g)

        return Gamma_t

    def get_gamma_1st_deriv(self, gamma_id):
        """
        Get ∂Γ_t / ∂γ_i (dGt_dg)
        gamma_id=1  -> 1
        gamma_id=-1 -> -1
        gamma_id=[1,-2] -> [1 * (1 - gamma_2), gamma_1 * (-1)]
        gamma_id=[-1,2,-3] -> [(-1) * gamma_2 * (1-gamma_3), (1-gamma_1) * 1 * (1-gamma_3), (1-gamma_1) * gamma_2 * (-1)]
        """
        if isinstance(gamma_id, np.ndarray) and gamma_id.size == 0: # returns 0 if gamma_id is ndarray and empty
            return np.array([0.0])
        if isinstance(gamma_id, (int, np.integer)):         # Wrap an integer gamma_id to list
            gamma_id = [gamma_id]
        if len(gamma_id) == 0:                              # returns 0 if gamma_id is list and empty
            return np.array([0.0])

        dGt_dg = np.zeros(len(gamma_id))
        for i,idx in enumerate(gamma_id):
            new_gamma_id = np.delete(gamma_id, i)
            weight = self.get_gamma_weight(new_gamma_id)
            dGt_dg[i] = np.sign(idx) * weight

        return dGt_dg

    def get_gamma_2nd_deriv(self, gamma_id):
        """
        Get ∂²Γ_t / (∂γ_i ∂γ_j) (dGt_dgg)
        gamma_id=1  ->  0
        gamma_id=[1,-2] -> [[0,        1 * (-1)],
                            [1 * (-1), 0       ]]
        gamma_id=[-1,2,-3] -> [[0,                      (-1) * 1 * (1-gamma_3), (-1) * gamma_2 * (-1) ],
                               [(-1) * 1 * (1-gamma_3), 0,                      (1-gamma_1) * 1 * (-1)],
                               [(-1) * gamma_2 * (-1),  (1-gamma_1) * 1 * (-1), 0                     ]]
        """
        if isinstance(gamma_id, np.ndarray) and gamma_id.size == 0: # returns 0 if gamma_id is ndarray and empty
            return np.zeros((1, 1))
        if isinstance(gamma_id, (int, np.integer)):             # Wrap an integer gamma_id to list
            gamma_id = [gamma_id]
        if len(gamma_id) == 0:                                  # returns 0 if gamma_id is list and empty
            return np.zeros((1, 1))

        num_gam = len(gamma_id)
        if num_gam == 1:
            return np.zeros((1, 1))

        dGt_dgg = np.zeros((num_gam, num_gam))
        for i,i_id in enumerate(gamma_id):
            for j, j_id in enumerate(gamma_id):
                if i == j:
                    dGt_dgg[i, j] = 0
                else:
                    new_gamma_id = np.delete(gamma_id, (i,j))
                    weight = self.get_gamma_weight(new_gamma_id)
                    dGt_dgg[i,j] = np.sign(i_id) * np.sign(j_id) * weight

        return dGt_dgg

@dataclass
class NetworkParameters:
    """
    Composite class representing full list of network parameterization.
    HAS-A TreeParameters instance and HAS-A GammaParameters instance.
    """
    def __init__(self, tree_params: TreeParameters, gamma_params: GammaParameters):
        self.values = np.concatenate([self.tree.values, self.gamma.values])
        self.tree = tree_params      # NetworkParameters HAS-A TreeParameters
        self.gamma = gamma_params    # NetworkParameters HAS-A GammaParameters
        self.tree_values = tree_params.values
        self.gamma_values = gamma_params.values

    @classmethod
    def from_vectors(cls, tree_vec, gamma_vec):
        """Convenience constructor to instantiate directly from raw vectors."""
        return cls(
            tree_params=TreeParameters(values=np.asarray(tree_vec)),
            gamma_params=GammaParameters(values=np.asarray(gamma_vec))
        )


#################################################################################################
## Use polymorphism of OOP to bypass "if is_asymm:" for getTrueProbs of symm and asymm quartet ##
#################################################################################################

class QuartetTreeTopology(ABC):
    """Abstract Strategy representing is_asymm of a 4-taxa subtree."""

    @abstractmethod
    def get_true_probs(self, t1: float, t2: float, t3: float, theta: float, alpha: float = 4 / 3):
        """Computes 15-category site pattern probabilities for this quartet subtree."""
        pass

class AsymmQuartet(QuartetTreeTopology):
    """Asymmetric Quartet Topology Strategy: (A,(B,(C,D)))"""

    def get_true_probs(self, myt1: float, myt2: float, myt3: float, theta: float, alpha: float = 4 / 3):
        # For an asymmetric quartet (A,(B,(C,D))), "label_speciation_time_idx" will label [1,2,3] in preorder traversal.
        # /--------------------------------------------------------------------------- A
        # + param_idx = 1
        # |                        /-------------------------------------------------- B
        # \------------------------+ param_idx = 2
        # :                        |                        /------------------------- C
        # :                        \------------------------+ param_idx = 3
        # :                        :                        \------------------------- D
        # tau3                     tau2                     tau1 (as in Chifman & Kubatko 2015)
        #
        # myt1 = Speciation time of taxa C,D
        # myt2 = Speciation time of taxa B,C,D
        # myt3 = Root age
        # These get us the site pattern frequencies that match the data sets generated by the seq-gen
        # pipeline that is commonly used in simulation studies.
        # the order of the site patterns is as in the headings in Table 1 in the Supplement to Chifman and Kubatko (2015):
        # xxxx - 0
        # xxxy = xxyx - 1
        # xyxx - 2
        # yxxx - 3
        # xxyy - 4
        # xyxy = yxxy - 5
        # xxyz - 6
        # yzxx - 7
        # xyxz = xyzx - 8
        # yxxz = yxzx - 9
        # xyzw - 10

        # This is in mutation units, different from the original function of Chifman & Kubatko
        t1 = myt1  # t1
        t2 = myt2  # t2
        t3 = myt3  # t3
        t = 2 * theta  # 2*theta
        m = alpha  # mu=4/3 for JC69



        #compute the C matrix
        cmat = np.matrix(np.zeros((11, 10)))

        # Row 0 (R's row 1)
        cmat[0, 0] = 1 / 256
        cmat[0, 1] = 3 / (256 * (1 + m * t))
        cmat[0, 2] = 6 / (256 * (1 + m * t))
        cmat[0, 3] = 12 / (256 * (1 + m * t) * (2 + m * t))
        cmat[0, 4] = 9 / (256 * (1 + m * t))
        cmat[0, 5] = 12 / (256 * (1 + m * t) * (2 + m * t))
        cmat[0, 6] = 9 / (256 * (1 + m * t) ** 2)
        cmat[0, 7] = 24 / (256 * (1 + m * t) * (2 + m * t))
        cmat[0, 8] = 48 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[0, 9] = (6 * m * t * (4 + m * t) * (4 + 3 * m * t)) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 1 (R's row 2)
        cmat[1, 0] = 1 / 256
        cmat[1, 1] = -1 / (256 * (1 + m * t))
        cmat[1, 2] = 2 / (256 * (1 + m * t))
        cmat[1, 3] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[1, 4] = 5 / (256 * (1 + m * t))
        cmat[1, 5] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[1, 6] = -3 / (256 * (1 + m * t) ** 2)
        cmat[1, 7] = 8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[1, 8] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[1, 9] = -(2 * m * t * (4 + m * t) * (4 + 3 * m * t)) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 2 (R's row 3)
        cmat[2, 0] = 1 / 256
        cmat[2, 1] = 3 / (256 * (1 + m * t))
        cmat[2, 2] = -2 / (256 * (1 + m * t))
        cmat[2, 3] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[2, 4] = 5 / (256 * (1 + m * t))
        cmat[2, 5] = 12 / (256 * (1 + m * t) * (2 + m * t))
        cmat[2, 6] = -3 / (256 * (1 + m * t) ** 2)
        cmat[2, 7] = -8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[2, 8] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[2, 9] = -(2 * m * t * (4 + m * t) * (4 + 3 * m * t)) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 3 (R's row 4)
        cmat[3, 0] = 1 / 256
        cmat[3, 1] = 3 / (256 * (1 + m * t))
        cmat[3, 2] = 6 / (256 * (1 + m * t))
        cmat[3, 3] = 12 / (256 * (1 + m * t) * (2 + m * t))
        cmat[3, 4] = -3 / (256 * (1 + m * t))
        cmat[3, 5] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[3, 6] = -3 / (256 * (1 + m * t) ** 2)
        cmat[3, 7] = -8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[3, 8] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[3, 9] = -(2 * m * t * (4 + m * t) * (4 + 3 * m * t)) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 4 (R's row 5)
        cmat[4, 0] = 1 / 256
        cmat[4, 1] = 3 / (256 * (1 + m * t))
        cmat[4, 2] = -2 / (256 * (1 + m * t))
        cmat[4, 3] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[4, 4] = 1 / (256 * (1 + m * t))
        cmat[4, 5] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[4, 6] = 9 / (256 * (1 + m * t) ** 2)
        cmat[4, 7] = -8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[4, 8] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[4, 9] = (2 * m * t * (4 + m * t) ** 2) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 5 (R's row 6)
        cmat[5, 0] = 1 / 256
        cmat[5, 1] = -1 / (256 * (1 + m * t))
        cmat[5, 2] = 2 / (256 * (1 + m * t))
        cmat[5, 3] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[5, 4] = 1 / (256 * (1 + m * t))
        cmat[5, 5] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[5, 6] = 1 / (256 * (1 + m * t) ** 2)
        cmat[5, 7] = -8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[5, 8] = 16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[5, 9] = m * t * (2 * 16 + 40 * m * t + 10 * (m ** 2) * (t ** 2)) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 6 (R's row 7)
        cmat[6, 0] = 1 / 256
        cmat[6, 1] = -1 / (256 * (1 + m * t))
        cmat[6, 2] = -2 / (256 * (1 + m * t))
        cmat[6, 3] = 4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[6, 4] = 1 / (256 * (1 + m * t))
        cmat[6, 5] = 4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[6, 6] = -3 / (256 * (1 + m * t) ** 2)
        cmat[6, 7] = -8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[6, 8] = 16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[6, 9] = 2 * (m ** 2) * (t ** 2) * (4 + m * t) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 7 (R's row 8)
        cmat[7, 0] = 1 / 256
        cmat[7, 1] = 3 / (256 * (1 + m * t))
        cmat[7, 2] = -2 / (256 * (1 + m * t))
        cmat[7, 3] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[7, 4] = -3 / (256 * (1 + m * t))
        cmat[7, 5] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[7, 6] = -3 / (256 * (1 + m * t) ** 2)
        cmat[7, 7] = 8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[7, 8] = 16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[7, 9] = 2 * (m ** 2) * (t ** 2) * (4 + m * t) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 8 (R's row 9)
        cmat[8, 0] = 1 / 256
        cmat[8, 1] = -1 / (256 * (1 + m * t))
        cmat[8, 2] = -2 / (256 * (1 + m * t))
        cmat[8, 3] = 4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[8, 4] = 1 / (256 * (1 + m * t))
        cmat[8, 5] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[8, 6] = 1 / (256 * (1 + m * t) ** 2)
        cmat[8, 7] = 0
        cmat[8, 8] = 0
        cmat[8, 9] = -(m ** 2) * (t ** 2) * (4 + 2 * m * t) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 9 (R's row 10)
        cmat[9, 0] = 1 / 256
        cmat[9, 1] = -1 / (256 * (1 + m * t))
        cmat[9, 2] = 2 / (256 * (1 + m * t))
        cmat[9, 3] = -4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[9, 4] = -3 / (256 * (1 + m * t))
        cmat[9, 5] = 4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[9, 6] = 1 / (256 * (1 + m * t) ** 2)
        cmat[9, 7] = 0
        cmat[9, 8] = 0
        cmat[9, 9] = -(m ** 2) * (t ** 2) * (4 + 2 * m * t) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Row 10 (R's row 11)
        cmat[10, 0] = 1 / 256
        cmat[10, 1] = -1 / (256 * (1 + m * t))
        cmat[10, 2] = -2 / (256 * (1 + m * t))
        cmat[10, 3] = 4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[10, 4] = -3 / (256 * (1 + m * t))
        cmat[10, 5] = 4 / (256 * (1 + m * t) * (2 + m * t))
        cmat[10, 6] = 1 / (256 * (1 + m * t) ** 2)
        cmat[10, 7] = 8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[10, 8] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[10, 9] = 2 * (m ** 3) * (t ** 3) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # Get the beta vector
        beta = np.matrix(np.zeros((10, 1)))
        beta[0, 0] = 1
        beta[1, 0] = np.exp(-2 * m * t1)
        beta[2, 0] = np.exp(-2 * m * t2)
        beta[3, 0] = np.exp(-m * t1) * np.exp(-2 * m * t2)
        beta[4, 0] = np.exp(-2 * m * t3)
        beta[5, 0] = np.exp(-m * t1) * np.exp(-2 * m * t3)
        beta[6, 0] = np.exp(-2 * m * t1) * np.exp(-2 * m * t3)
        beta[7, 0] = np.exp(-m * t2) * np.exp(-2 * m * t3)
        beta[8, 0] = np.exp(-m * t1) * np.exp(-m * t2) * np.exp(-2 * m * t3)
        # beta[9, 0] = np.exp((2 / t) * (t1 - t2)) * np.exp(-2 * m * (t2 + t3))
        exp_9 = (2 / t) * (t1 - t2) - 2 * m * (t2 + t3)
        beta[9, 0] = np.exp(np.clip(exp_9, a_min=-np.inf, a_max=700))

        # unweighted 11-category site pattern probability
        p = cmat @ beta  # Here, @ means matrix multiplication

        # unweighted 15-category site pattern probability
        p_15 = np.zeros(15)
        p_15[0] = p[0,0]  # xxxx - 0
        p_15[1:3] = p[1,0]  # xxxy - 1 & xxyx - 2
        p_15[3] = p[2,0] # xyxx - 3
        p_15[4] = p[3,0] # yxxx - 4
        p_15[7] = p[4,0]  # xxyy - 7
        p_15[5:7] = p[5,0]  # xyxy - 5 & yxxy - 6
        p_15[12] = p[6,0]  # xxyz - 12
        p_15[13] = p[7,0]  # yzxx - 13
        p_15[8:10] = p[8,0]  # xyxz - 8 & xyzx - 9
        p_15[10:12] = p[9,0]  # yxxz - 10 & yxzx - 11
        p_15[14] = p[10,0]  # xyzw - 14

        weights = np.zeros(15, dtype=int)
        weights[0] = 4  # 4 ways of xxxx-0: AAAA, CCCC, GGGG, TTTT
        weights[1:5] = 12  # 4*3=12 ways of xxxy-1, xxyx-2, xyxx-3 & yxxx-4: AAAC, etc
        weights[5:8] = 12  # 4*3=12 ways of xyxy-5, yxxy-6, xxyy-7: ACAC, etc
        weights[8:14] = 24  # 4*3*2=24 ways of xyxz-8, xyzx-9, yxxz-10, yxzx-11, xxyz-12, yzxx-13
        weights[14] = 24  # 4*3*2*1=24 ways of xyzw-14

        return weights * p_15  # 11-category site pattern probabilities

class SymmQuartet(QuartetTreeTopology):
    """Symmetric Quartet Topology Strategy: ((A,B),(C,D))"""

    def get_true_probs(self, myt1: float, myt2: float, myt3: float, theta: float, alpha: float = 4 / 3):
        # For a symmetric quartet ((A,B),(C,D)), "label_speciation_time_idx" will label [1,2,3] in preorder traversal.
        #                              /---------------------------------------------- A
        # /----------------------------+ param_idx = 2
        # |                            \---------------------------------------------- B
        # + param_idx = 1              :
        # |                            :                    /------------------------- C
        # \-------------------------------------------------+ param_idx = 3
        # :                            :                    \------------------------- D
        # tau3                         tau2                 tau1 (as in Chifman & Kubatko 2015)
        #
        # myt1 = Speciation time of taxa C,D
        # myt2 = Speciation time of taxa A,B
        # myt3 = Root age
        # These get us the site pattern frequencies that match the data sets generated by the seq-gen
        # pipeline that is commonly used in simulation studies.
        # the order of the site patterns is as in the headings in Table 1 in the Supplement to Chifman and Kubatko (2015):
        # xxxx - 0
        # xxxy = xxyx - 1
        # xyxx = yxxx - 2
        # xyxy = yxxy - 3
        # xxyy - 4
        # xxyz - 5
        # yzxx - 6
        # xyzx = yxxz = xyzx = yxzx - 7
        # xyzw - 8

        # This is in mutation units, different from the original function of Chifman & Kubatko
        t1 = myt1  # t1
        t2 = myt2  # t2
        t3 = myt3  # t3
        t = 2 * theta  # 2*theta
        m = alpha  # mu=4/3 for JC69



        # compute the C matrix.
        cmat = np.matrix(np.zeros((9,9)))

        # Row 0 (R's row 1)
        cmat[0, :] = [1 / 256 for i in range(9)]

        # Row 1 (R's row 2)
        cmat[1, [0, 2, 4, 6]] = 3 / (256 * (1 + m * t))
        cmat[1, [1, 3, 5, 7, 8]] = -1 / (256 * (1 + m * t))

        # Row 2 (R's row 3)
        cmat[2, [0, 1, 4, 5]] = 3 / (256 * (1 + m * t))
        cmat[2, [2, 3, 6, 7, 8]] = -1 / (256 * (1 + m * t))

        # Row 3 (R's row 4)
        cmat[3, [0, 4]] = 9 / (256 * (1 + m * t) ** 2)
        cmat[3, [1, 2, 5, 6]] = -3 / (256 * (1 + m * t) ** 2)
        cmat[3, [3, 7, 8]] = 1 / (256 * (1 + m * t) ** 2)

        # Row 4 (R's row 5)
        cmat[4, 0] = 12 / (256 * (1 + m * t))
        cmat[4, [1, 2, 3]] = 4 / (256 * (1 + m * t))
        cmat[4, [4, 5, 6, 8]] = -4 / (256 * (1 + m * t))
        cmat[4, 7] = 0

        # Row 5 (R's row 6)
        cmat[5, 0] = 24 / (256 * (1 + m * t) * (2 + m * t))
        cmat[5, [1, 3, 4, 6]] = -8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[5, 2] = 8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[5, [5, 8]] = 8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[5, 7] = 0

        # Row 6 (R's row 7)
        cmat[6, 0] = 24 / (256 * (1 + m * t) * (2 + m * t))
        cmat[6, 1] = 8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[6, [2, 3, 4, 5]] = -8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[6, [6, 8]] = 8 / (256 * (1 + m * t) * (2 + m * t))
        cmat[6, 7] = 0

        # Row 7 (R's row 8)
        cmat[7, 0] = 48 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[7, [1, 2, 4]] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[7, [3, 5, 6]] = 16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        cmat[7, 7] = 0
        cmat[7, 8] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)

        # Row 8 (R's row 9)
        cmat[8, 0] = 6 * m * t * (4 + m * t) * (4 + 3 * m * t) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        cmat[8, [1, 2]] = -2 * m * t * (4 + m * t) * (4 + 3 * m * t) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        cmat[8, 3] = m * t * (32 + 40 * m * t + 10 * (m ** 2) * (t ** 2)) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        cmat[8, 4] = 2 * m * t * (4 + m * t) ** 2 / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        cmat[8, [5, 6]] = 2 * (m ** 2) * (t ** 2) * (4 + m * t) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        cmat[8, 7] = -(m ** 2) * (t ** 2) * (4 + 2 * m * t) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        cmat[8, 8] = 2 * (m ** 3) * (t ** 3) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # get the beta vector
        beta = np.matrix(np.zeros((9,1)))
        beta[0, 0] = 1
        beta[1, 0] = np.exp(-2 * m * t1)
        beta[2, 0] = np.exp(-2 * m * t2)
        beta[3, 0] = np.exp(-2 * m * t1) * np.exp(-2 * m * t2)
        beta[4, 0] = np.exp(-2 * m * t3)
        beta[5, 0] = np.exp(-m * t1) * np.exp(-2 * m * t3)
        beta[6, 0] = np.exp(-m * t2) * np.exp(-2 * m * t3)
        beta[7, 0] = np.exp(-m * t1) * np.exp(-m * t2) * np.exp(-2 * m * t3)
        # beta[8, 0] = np.exp(2 * t1 / t) * np.exp(2 * t2 / t) * np.exp(-4 * t3 * (m + 1 / t))
        exp_8 = (2 * t1 / t) + (2 * t2 / t) - 4 * t3 * (m + 1 / t)
        beta[8, 0] = np.exp(np.clip(exp_8, a_min=-np.inf, a_max=700))

        # unweighted 9-category site pattern probability
        p = cmat.T @ beta # Here, @ means matrix multiplication

        # unweighted 15-category site pattern probability
        p_15 = np.zeros(15)
        p_15[0] = p[0,0] # xxxx - 0
        p_15[1:3] = p[1,0] # xxxy - 1 & xxyx - 2
        p_15[3:5] = p[2,0] # xyxx - 3 & yxxx - 4
        p_15[5:7] = p[3,0] # xyxy - 5 & yxxy - 6
        p_15[7] = p[4,0] # xxyy - 7
        p_15[12] = p[5,0] # xxyz - 12
        p_15[13] = p[6,0] # yzxx - 13
        p_15[8:12] = p[7,0] # xyxz - 8 & xyzx - 9 & yxxz - 10 & yxzx - 11
        p_15[14] = p[8,0] # xyzw - 14

        weights = np.zeros(15, dtype=int)
        weights[0] = 4  # 4 ways of xxxx-0: AAAA, CCCC, GGGG, TTTT
        weights[1:5] = 12  # 4*3=12 ways of xxxy-1, xxyx-2, xyxx-3 & yxxx-4: AAAC, etc
        weights[5:8] = 12  # 4*3=12 ways of xyxy-5, yxxy-6, xxyy-7: ACAC, etc
        weights[8:14] = 24  # 4*3*2=24 ways of xyxz-8, xyzx-9, yxxz-10, yxzx-11, xxyz-12, yzxx-13
        weights[14] = 24  # 4*3*2*1=24 ways of xyzw-14

        return weights * p_15  # weighted 15-category site pattern probabilities


##############################################################################################################
## A quartet subnetwork may have multiple displayed quartet subtrees, which are saved as QuartetTreeFeature ##
## (param_idx, topology, taxa_perm, gamma_id). This section handles classes regarding quartet features.     ##
##############################################################################################################

def identify_site_pattern_index(site_pattern):
    """Given an input site pattern with length four, identify the corresponding index."""

    from collections import Counter

    # site_pattern = np.array(['A', 'A', 'A', 'C']) # Example site for testing
    repeated_nucleo_count = len(np.unique(site_pattern))
    if repeated_nucleo_count == 1:
        # xxxx - 0
        site_pattern_idx = 0
    elif repeated_nucleo_count == 2:
        # store the unique nucleotides with corresponding counts in a site
        unique_nucleo = Counter(site_pattern)
        if max(unique_nucleo.values()) == 3:
            least_freq_nucleo = min(unique_nucleo, key=unique_nucleo.get)
            locations = np.where(site_pattern == least_freq_nucleo)[0][0]  # location of least frequent nucleotide
            # xxxy - 1. Least frequent nucleotide location is 3
            # xxyx - 2. Least frequent nucleotide location is 2
            # xyxx - 3. Least frequent nucleotide location is 1
            # yxxx - 4. Least frequent nucleotide location is 0
            site_pattern_idx = int(4 - locations)
        else:  # elif max(unique_nucleo.values()) == 2:
            locations = np.where(site_pattern == site_pattern[0])[0][1]
            if locations == 2:
                # xyxy - 5. Second appearance of the first nucleotide (x) is location 2
                site_pattern_idx = 5
            elif locations == 3:
                # yxxy - 6. Second appearance of the first nucleotide (y) is location 3
                site_pattern_idx = 6
            else:  # elif locations == 1:
                # xxyy - 7. Second appearance of the first nucleotide (x) is location 1
                site_pattern_idx = 7
    elif repeated_nucleo_count == 3:
        unique_nucleo = Counter(site_pattern)
        most_freq_nucleo = max(unique_nucleo, key=unique_nucleo.get)
        locations = np.where(site_pattern == most_freq_nucleo)[0]
        if np.all(locations == [0, 2]):
            # xyxz - 8. Most frequent nucleotide location is [0,2]
            site_pattern_idx = 8
        elif np.all(locations == [0, 3]):
            # xyzx - 9. Most frequent nucleotide location is [0,3]
            site_pattern_idx = 9
        elif np.all(locations == [1, 2]):
            # yxxz - 10. Most frequent nucleotide location is [1,2]
            site_pattern_idx = 10
        elif np.all(locations == [1, 3]):
            # yxzx - 11. Most frequent nucleotide location is [1,3]
            site_pattern_idx = 11
        elif np.all(locations == [0, 1]):
            # xxyz - 12. Most frequent nucleotide location is [0,1]
            site_pattern_idx = 12
        else:  # elif np.all(locations == [2,3]):
            # yzxx - 13. Most frequent nucleotide location is [2,3]
            site_pattern_idx = 13
    else:  # elif repeated_nucleo_count == 4:
        # xyzw - 14
        site_pattern_idx = 14

    return site_pattern_idx

# # Below are codes for generating the SITE_PATTERN_RELATIONSHIPS mapping.
# import numpy as np
# import itertools
# index_to_pattern = np.array([
#     ["x", "x", "x", "x"],  # 0
#     ["x", "x", "x", "y"],  # 1
#     ["x", "x", "y", "x"],  # 2
#     ["x", "y", "x", "x"],  # 3
#     ["y", "x", "x", "x"],  # 4
#     ["x", "y", "x", "y"],  # 5
#     ["y", "x", "x", "y"],  # 6
#     ["x", "x", "y", "y"],  # 7
#     ["x", "y", "x", "z"],  # 8
#     ["x", "y", "z", "x"],  # 9
#     ["y", "x", "x", "z"],  # 10
#     ["y", "x", "z", "x"],  # 11
#     ["x", "x", "y", "z"],  # 12
#     ["y", "z", "x", "x"],  # 13
#     ["x", "y", "z", "w"]   # 14
# ])
# SITE_PATTERN_RELATIONSHIPS = {
# perm: [identify_site_pattern_index(sp)
#        for sp in index_to_pattern[:, perm]]
# for perm in itertools.permutations([0, 1, 2, 3])
# }

# For each taxa permutation, get the site pattern relationship from SITE_PATTERN_RELATIONSHIPS
SITE_PATTERN_RELATIONSHIPS = {
    (0, 1, 2, 3): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    (0, 1, 3, 2): [0, 2, 1, 3, 4, 6, 5, 7, 9, 8, 11, 10, 12, 13, 14],
    (0, 2, 1, 3): [0, 1, 3, 2, 4, 7, 6, 5, 12, 9, 10, 13, 8, 11, 14],
    (0, 2, 3, 1): [0, 2, 3, 1, 4, 7, 5, 6, 12, 8, 11, 13, 9, 10, 14],
    (0, 3, 1, 2): [0, 3, 1, 2, 4, 6, 7, 5, 9, 12, 13, 10, 8, 11, 14],
    (0, 3, 2, 1): [0, 3, 2, 1, 4, 5, 7, 6, 8, 12, 13, 11, 9, 10, 14],
    (1, 0, 2, 3): [0, 1, 2, 4, 3, 6, 5, 7, 10, 11, 8, 9, 12, 13, 14],
    (1, 0, 3, 2): [0, 2, 1, 4, 3, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14],
    (1, 2, 0, 3): [0, 1, 3, 4, 2, 6, 7, 5, 10, 13, 12, 9, 8, 11, 14],
    (1, 2, 3, 0): [0, 2, 3, 4, 1, 5, 7, 6, 11, 13, 12, 8, 9, 10, 14],
    (1, 3, 0, 2): [0, 3, 1, 4, 2, 7, 6, 5, 13, 10, 9, 12, 8, 11, 14],
    (1, 3, 2, 0): [0, 3, 2, 4, 1, 7, 5, 6, 13, 11, 8, 12, 9, 10, 14],
    (2, 0, 1, 3): [0, 1, 4, 2, 3, 7, 5, 6, 12, 11, 8, 13, 10, 9, 14],
    (2, 0, 3, 1): [0, 2, 4, 1, 3, 7, 6, 5, 12, 10, 9, 13, 11, 8, 14],
    (2, 1, 0, 3): [0, 1, 4, 3, 2, 5, 7, 6, 8, 13, 12, 11, 10, 9, 14],
    (2, 1, 3, 0): [0, 2, 4, 3, 1, 6, 7, 5, 9, 13, 12, 10, 11, 8, 14],
    (2, 3, 0, 1): [0, 3, 4, 1, 2, 5, 6, 7, 8, 10, 9, 11, 13, 12, 14],
    (2, 3, 1, 0): [0, 3, 4, 2, 1, 6, 5, 7, 9, 11, 8, 10, 13, 12, 14],
    (3, 0, 1, 2): [0, 4, 1, 2, 3, 5, 7, 6, 11, 12, 13, 8, 10, 9, 14],
    (3, 0, 2, 1): [0, 4, 2, 1, 3, 6, 7, 5, 10, 12, 13, 9, 11, 8, 14],
    (3, 1, 0, 2): [0, 4, 1, 3, 2, 7, 5, 6, 13, 8, 11, 12, 10, 9, 14],
    (3, 1, 2, 0): [0, 4, 2, 3, 1, 7, 6, 5, 13, 9, 10, 12, 11, 8, 14],
    (3, 2, 0, 1): [0, 4, 3, 1, 2, 6, 5, 7, 10, 8, 11, 9, 13, 12, 14],
    (3, 2, 1, 0): [0, 4, 3, 2, 1, 5, 6, 7, 11, 9, 10, 8, 13, 12, 14]}

@dataclass
class QuartetTreeFeature:
    """Encapsulates quartet features (param_idx, topology, taxa_perm, gamma_id) for a single displayed quartet subtree."""
    param_idx: list[int]
    topology: QuartetTreeTopology  # Instance of AsymmQuartet or SymmQuartet
    taxa_perm: list[int]
    gamma_id: list[int]

    @property
    def num_retic(self):
        """Total number of gamma parameters in the species network."""
        if not self.gamma_id:
            return 0
        return len(self.gamma_id)

    def get_site_pattern_relationship(self, inverse: bool = False):
        """
        Get the site pattern relationship vector (a list that permutes the indexing of the 15 category site pattern)
        given the taxa permutation from the quartet feature.
        """
        taxa_perm = tuple(np.argsort(self.taxa_perm)) if inverse else tuple(self.taxa_perm)
        return SITE_PATTERN_RELATIONSHIPS[taxa_perm]

    def get_site_pattern_map_matrix(self, inverse: bool = False):
        """
        Get the site pattern mapping matrix (a linear transformation matrix that permutes the 15 categories site pattern
        by matrix multiplication) given the taxa permutation from the quartet feature.
        """

        relationship = self.get_site_pattern_relationship(inverse)
        rows = np.arange(15)
        map_matrix = np.zeros((15, 15), dtype=int)
        map_matrix[rows, relationship] = 1

        return map_matrix


@dataclass
class QuartetFeature:
    """Encapsulates quartet features (param_idx, is_asymm, taxa_perm, gamma_id) for a 4-taxa subnetwork."""
    taxa: tuple
    tree_features: list["QuartetTreeFeature"]

    @property
    def param_idx(self):
        """Delegates param_idx directly to the underlying QuartetTreeFeature."""
        return np.array([tf.param_idx for tf in self.tree_features])

    @property
    def is_asymm(self):
        """Delegates is_asymm directly to the underlying QuartetTreeFeature."""
        return np.array([isinstance(tf.topology, AsymmQuartet) for tf in self.tree_features])

    @property
    def taxa_perm(self):
        """Delegates taxa_perm directly to the underlying QuartetTreeFeature."""
        return np.array([tf.taxa_perm for tf in self.tree_features])

    @property
    def gamma_id(self):
        """Delegates gamma_id directly to the underlying QuartetTreeFeature."""
        return np.array([tf.gamma_id for tf in self.tree_features])

    def to_dict(self):
        """Helper to convert to a dictionary if matrix format is needed for downstream models."""
        return {
            "taxa": self.taxa,
            "param_idx": self.param_idx,
            "is_asymm": self.is_asymm,
            "taxa_perm": self.taxa_perm,
            "gamma_id": self.gamma_id
        }

    @property
    def num_retic(self):
        """Number of reticulations in the quartet subnetwork."""
        if not self.tree_features:
            return 0
        return len(self.tree_features[0].gamma_id)

    def get_true_probs(self, net_params: NetworkParameters, alpha: float = 4 / 3):
        """
        Using the quartet features (param_idx, is_asymm, taxa_perm, gamma_id) of a combination of four taxa, get the true
        site pattern probabilities for the quartet subnetwork pulled from network.
        The site pattern probabilities of a quartet subnetwork is a weighted mixture of site pattern probabilities of
        quartet subtrees pulled from the displayed trees.
        """
        # Remove common gamma indices in gamma_id for the ease of computing gamma weight
        gamma_id_clean = net_params.gamma.remove_common_elements(self.gamma_id)

        all_p_Qt = []
        all_Gamma_t = []

        # For simplicity, we use these notation: qt_feat = quartet tree feature, gi_c = cleaned gamma_id.
        for qt_feat, gi_c in zip(self.tree_features, gamma_id_clean):
            # 1. Get gamma weight
            all_Gamma_t.append(net_params.gamma.get_gamma_weight(gi_c))

            # 2. Get quartet parameters + site pattern mapping
            tau1, tau2, tau3, theta = net_params.tree.get_tau_theta(qt_feat.param_idx)
            sp_relation = qt_feat.get_site_pattern_relationship()

            # 3. Compute site pattern probabilities (symmetric / asymmetric)
            p_Qt = qt_feat.topology.get_true_probs(tau1, tau2, tau3, theta, alpha)
            all_p_Qt.append(p_Qt[sp_relation])  # Append site pattern probabilities with site pattern mapping applied

        # 4. Normalize gamma weights & compute TrueProbs
        all_p_Qt = np.array(all_p_Qt)
        all_Gamma_t = np.array(all_Gamma_t)
        if self.num_retic > 1:  # quartet is a network
            if np.isclose(all_Gamma_t.sum(), 0.0):
                raise ValueError("The denominator all_Gamma_t.sum() is zero when normalizing gamma weights.")
            all_Gamma_t = all_Gamma_t / all_Gamma_t.sum()
            TrueProbs = all_p_Qt.T @ all_Gamma_t
        else:  # quartet is a tree
            TrueProbs = all_p_Qt[0]

        return TrueProbs


@dataclass
class PairedQuartetFeature:
    """Encapsulates paired seq_data_rows with quartet features (param_idx, is_asymm, taxa_perm, gamma_id)"""
    seq_data_rows: np.ndarray
    quartet_feature: QuartetFeature

    def __getattr__(self, name: str):
        """
        If an attribute or method isn't found on PairedQuartetFeature, Python automatically redirects the call to
        self.quartet_feature.
        """
        return getattr(self.quartet_feature, name)



########################################################
## Get site pattern counts of data and of each quartet ##
########################################################

def get_n_D_hasAmbiguityCode(seq_data, ACGT_weight=None, skip_gap=False, skip_missing=False):
    """Get the observed site pattern counts n^D in data and corresponding site pattern ID."""

    from collections import defaultdict

    if ACGT_weight is None:
        ACGT_weight = [1, 1, 1, 1]

    # Extract sequence of each taxa by self tip order into a list, and reformat the elements (nucleotides) into string format
    sequence_matrix = [[str(element) for element in seq_data[taxon.label]] for taxon in seq_data.taxon_namespace]

    # Use numpy to transpose the sequence matrix so that the rows are site patterns
    site_pattern_array = np.array(sequence_matrix).T

    # count unique site patterns of the original sequence matrix
    full_unique_site, full_unique_site_count = np.unique(site_pattern_array, axis=0, return_counts=True)

    # Task 1: Expand each unique site (w/ ambiguity code) into list of possible sites (w/o ambiguity) with weight proportions.
    # Task 2: Convert site patterns into an identification code (by x,y,z,w) and use it as key.
    # Task 3: Assign weighted unique_site_count to the site pattern by key (site pattern identification code) and
    #         collapse site pattern counts with the same key.
    def site_pattern_IDcode(site_pattern):
        """
        Convert a site pattern (in nucleotides) into ID code composed of x,y,z,w.
        When multiple nucleotides have the same frequency in a site pattern, the first nucleotide encountered
        is the first to be converted by letters (x, y, z, w) instead of following the alphabetical order.
        Example:
        ['A', 'A', 'A', 'G', 'G', 'C', 'C', 'T'] => 'xxxyyzzw' by first appearance order. G appears first than C.
        ['A', 'A', 'A', 'G', 'G', 'C', 'C', 'T'] => 'xxxzzyyw' by alphabetical order. C is before G alphabetically.
        """
        from collections import Counter

        # Count nucleotide frequencies. We use 'x' to denote the most frequent nucleotide and so on for 'y','z','w'.
        counts = Counter(site_pattern)

        # When nucleotide frequencies tied in a site pattern, the first appeared nucleotide wins. (not by alphabetical order)
        # Example: ['A', 'A', 'A', 'G', 'G', 'C', 'C', 'T'] => 'xxxyyzzw' by first appearance order
        #          ['A', 'A', 'A', 'G', 'G', 'C', 'C', 'T'] => 'xxxzzyyw' by alphabetical order

        # Store index of first appearance of A,C,G,T
        # first_occurrence = {nuc: i for i, nuc in enumerate(site_pattern)} # index of the last occurrence of A,C,G,T
        first_occurrence = {}
        for i, nuc in enumerate(site_pattern):
            if nuc not in first_occurrence:
                first_occurrence[nuc] = i

        # Sort nucleotides by frequency (descending), breaking ties by index of first appearance
        sorted_nucleotides = sorted(counts, key=lambda x: (-counts[x], first_occurrence[x]))

        # Create a mapping from sorted nucleotides to code {x, y, z, w}
        mapping_letters = ['x', 'y', 'z', 'w']
        mapping = {nuc: mapping_letters[i] for i, nuc in enumerate(sorted_nucleotides)}

        # Convert site pattern to identification code
        code = [mapping[nuc] for nuc in site_pattern]

        return code

    def expand_weighted_site_pattern(site, ACGT_weight):
        """
        Expand a site pattern with ambiguity code into list of possible site patterns with weight proportions.
        This function allows ambiguity code for site pattern counting.
        """
        from itertools import product

        # IUPAC ambiguity code mapping
        ambiguity_code_map = {
            'A': {'A'},
            'C': {'C'},
            'G': {'G'},
            'T': {'T'},
            'R': {'A', 'G'},
            'Y': {'C', 'T'},
            'S': {'G', 'C'},
            'W': {'A', 'T'},
            'K': {'G', 'T'},
            'M': {'A', 'C'},
            'B': {'C', 'G', 'T'},
            'D': {'A', 'G', 'T'},
            'H': {'A', 'C', 'T'},
            'V': {'A', 'C', 'G'},
            'N': {'A', 'C', 'G', 'T'}
        }

        # Fast lookup map for weights to avoid calling .index() repeatedly inside the loop
        weight_map = {'A': ACGT_weight[0], 'C': ACGT_weight[1], 'G': ACGT_weight[2], 'T': ACGT_weight[3]}

        # Map each character in the site pattern to its possible nucleotides
        nucleotide_options = [ambiguity_code_map[char] for char in site]

        # Generate all combinations using product
        expanded_site_patterns = [np.array(site) for site in product(*nucleotide_options)]

        # Calculate the weight of each pattern using the fast lookup dictionary
        site_pattern_weights = np.prod(
            [[weight_map[nuc] for nuc in site_pattern] for site_pattern in expanded_site_patterns],
            axis=1
        )
        site_pattern_weight_proportions = site_pattern_weights / site_pattern_weights.sum()

        return zip(expanded_site_patterns, site_pattern_weight_proportions)

    # --- Vectorized Pre-processing of Gaps and Missing Data ---
    # Create a boolean mask of which sites to keep
    keep_mask = np.ones(len(full_unique_site), dtype=bool)
    if skip_gap:
        keep_mask &= ~(full_unique_site == '-').any(axis=1)  # keep_mask = False when site includes "-"
    if skip_missing:
        keep_mask &= ~(full_unique_site == '?').any(axis=1)  # keep_mask = False when site includes "?"

    # Filter arrays using the mask and copy to prevent mutating the original sequences
    filtered_sites = full_unique_site[keep_mask].copy()
    filtered_counts = full_unique_site_count[keep_mask]

    # Vectorized replacement of remaining gaps/missing data with 'N'
    if not skip_gap:
        filtered_sites[filtered_sites == '-'] = 'N'
    if not skip_missing:
        filtered_sites[filtered_sites == '?'] = 'N'

    # --- Main Processing Loop ---
    dict_n_D = defaultdict(float)

    for unique_site, count in zip(filtered_sites, filtered_counts):
        # Task 1: Expand each unique site (w/ ambiguity code) into list of possible sites (w/o ambiguity) with weight proportions.
        for site, weight in expand_weighted_site_pattern(unique_site, ACGT_weight):
            # Task 2: Convert site patterns into an identification code (by x,y,z,w) and use it as dict_key.
            key_IDcode = tuple(site_pattern_IDcode(site))

            # Task 3: Accumulate weighted counts to the same site pattern ID
            dict_n_D[key_IDcode] += weight * count

    # Extract collapsed result from dictionary
    collapsed_IDcode = np.array(list(dict_n_D.keys()))
    collapsed_n_D    = np.fromiter(dict_n_D.values(), dtype=np.float64)

    return collapsed_IDcode, collapsed_n_D


def get_quartet_site_pattern_count_map_matrix(quartet_sites, collapsed_site_pattern_count):
    """
    Count 15 category site patterns of a quartet from collapsed_site_pattern_count and generate a mapping matrix
    from site pattern counts of data (n^D) to quartet site pattern counts (n^Q).
    xxxx - 0
    xxxy - 1
    xxyx - 2
    xyxx - 3
    yxxx - 4
    xyxy - 5
    yxxy - 6
    xxyy - 7
    xyxz - 8
    xyzx - 9
    yxxz - 10
    yxzx - 11
    xxyz - 12
    yzxx - 13
    xyzw - 14
    """
    from collections import Counter

    def get_quartet_unique_sites(quartet_sites, unique_site_count):
        """Collapse unique_site_count of data matrix for a quartet."""

        # quartet_unique_site_InvIndex are the first occurrence index of quartet_unique_site
        quartet_unique_site, quartet_unique_site_InvIndex = np.unique(quartet_sites, axis=0, return_inverse=True)

        # Use np.bincount to sum counts based on the unique site indices
        quartet_unique_site_count = np.bincount(quartet_unique_site_InvIndex, weights=unique_site_count)

        return quartet_unique_site, quartet_unique_site_count, quartet_unique_site_InvIndex

    site_pattern_counts = np.zeros(15)

    # Maps from the site pattern counts of data to the 15 category site pattern counts
    mapping_matrix = np.matrix(np.zeros((15,len(collapsed_site_pattern_count)), dtype=int))

    # 15 category site pattern counts: unique sites (quartet_unique_site) and collapsed counts (quartet_unique_site_count).
    # Mapping matrix: the inverse indices of the unique sites (quartet_unique_site_InvIndex)
    (quartet_unique_site, quartet_unique_site_count,
     quartet_unique_site_InvIndex) = get_quartet_unique_sites(quartet_sites, collapsed_site_pattern_count)

    for j, unique_site in enumerate(quartet_unique_site):
        # For the j^th unique site, "np.where(quartet_unique_site_InvIndex == j)[0]" gives the vector of indices of
        # where this unique site appears in its original collapsed_site_pattern_count.

        # unique_site = np.array(['A', 'T', 'A', 'C']) # Example site for testing
        repeated_nucleo_count = len(np.unique(unique_site))
        if repeated_nucleo_count == 1:
            # xxxx - 0
            site_pattern_counts[0] += quartet_unique_site_count[j]
            mapping_matrix[0, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
        elif repeated_nucleo_count == 2:
            # store the unique nucleotides with corresponding counts in a site
            unique_nucleo = Counter(unique_site)
            if max(unique_nucleo.values()) == 3:
                least_freq_nucleo = min(unique_nucleo, key=unique_nucleo.get)
                locations = np.where(unique_site == least_freq_nucleo)[0]  # location of least frequent nucleotide
                # xxxy - 1. Least frequent nucleotide location is 3
                # xxyx - 2. Least frequent nucleotide location is 2
                # xyxx - 3. Least frequent nucleotide location is 1
                # yxxx - 4. Least frequent nucleotide location is 0
                site_pattern_counts[4 - locations] += quartet_unique_site_count[j]
                mapping_matrix[4 - locations, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
            else:  # elif max(unique_nucleo.values()) == 2:
                locations = np.where(unique_site == unique_site[0])[0][1]
                if locations == 2:
                    # xyxy - 5. Second appearance of the first nucleotide (x) is location 2
                    site_pattern_counts[5] += quartet_unique_site_count[j]
                    mapping_matrix[5, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
                elif locations == 3:
                    # yxxy - 6. Second appearance of the first nucleotide (y) is location 3
                    site_pattern_counts[6] += quartet_unique_site_count[j]
                    mapping_matrix[6, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
                else:  # elif locations == 1:
                    # xxyy - 7. Second appearance of the first nucleotide (x) is location 1
                    site_pattern_counts[7] += quartet_unique_site_count[j]
                    mapping_matrix[7, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
        elif repeated_nucleo_count == 3:
            unique_nucleo = Counter(unique_site)
            most_freq_nucleo = max(unique_nucleo, key=unique_nucleo.get)
            locations = np.where(unique_site == most_freq_nucleo)[0]
            if np.all(locations == [0, 2]):
                # xyxz - 8. Most frequent nucleotide location is [0,2]
                site_pattern_counts[8] += quartet_unique_site_count[j]
                mapping_matrix[8, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
            elif np.all(locations == [0, 3]):
                # xyzx - 9. Most frequent nucleotide location is [0,3]
                site_pattern_counts[9] += quartet_unique_site_count[j]
                mapping_matrix[9, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
            elif np.all(locations == [1, 2]):
                # yxxz - 10. Most frequent nucleotide location is [1,2]
                site_pattern_counts[10] += quartet_unique_site_count[j]
                mapping_matrix[10, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
            elif np.all(locations == [1, 3]):
                # yxzx - 11. Most frequent nucleotide location is [1,3]
                site_pattern_counts[11] += quartet_unique_site_count[j]
                mapping_matrix[11, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
            elif np.all(locations == [0, 1]):
                # xxyz - 12. Most frequent nucleotide location is [0,1]
                site_pattern_counts[12] += quartet_unique_site_count[j]
                mapping_matrix[12, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
            else:  # elif np.all(locations == [2,3]):
                # yzxx - 13. Most frequent nucleotide location is [2,3]
                site_pattern_counts[13] += quartet_unique_site_count[j]
                mapping_matrix[13, np.where(quartet_unique_site_InvIndex == j)[0]] = 1
        else:  # elif repeated_nucleo_count == 4:
            # xyzw - 14
            site_pattern_counts[14] += quartet_unique_site_count[j]
            mapping_matrix[14, np.where(quartet_unique_site_InvIndex == j)[0]] = 1

    return site_pattern_counts, mapping_matrix


#####################################################
## Compute composite likelihood of species network ##
#####################################################

def get_taxa_partition(seq_data, phylox_network, Imap_file_path):
    """Produce tax_partition from user input Imap file"""
    from collections import defaultdict

    # Read the Imap file into a dictionary
    imap_dict = defaultdict(list)
    with open(Imap_file_path, "r") as f:
        for line in f:
            old_name, species = line.strip().split()
            imap_dict[species].append(old_name)

    # Returns True when seq_taxon_name matches any one of the imap_names
    def matches(seq_taxon_name, imap_names):
        for name in imap_names:
            # Replace space with underscore if there is any
            if ' ' in seq_taxon_name:
                seq_taxon_name = seq_taxon_name.replace(" ", "_")
            # Remove caret ^ before taxon name
            if name == seq_taxon_name.lstrip("^"):
                return True
        return False

    taxa_partition = [
        # seq_data row indices of the taxa that contains the species name in the self
        [idx for idx, seq_taxon in enumerate(seq_data.taxon_namespace) if matches(seq_taxon.label, imap_dict[label])]
        for label in get_taxa_labels(phylox_network)
    ]

    return taxa_partition


def pair_quartetRows_quartetFeatures(seq_data, phylox_network, major_tree, taxa_partition=None):
    """
    Given seq_data and taxa_partition, we can select one individual per species to form a quartet matrix by
    choosing the rows (individuals) of seq_data. The choice of row indices that forms quartet matrix is
    (quartet_rows). For each quartet_rows, we pair it up with the corresponding quartet features (param_idx,
    is_asymm, taxa_perm, gamma_id).

    This function handles data with multiple individuals per species through "taxa_partition". For example,
    we have species tree ((E,B),((C,D),A)) and data = {A1,A2,B1,B2,C1,C2,D1,D2,E1,E2}, the taxa partition is
    [[8, 9], [2, 3], [4, 5], [6, 7], [0, 1]].
    """
    from itertools import combinations, product

    # Step 1: Taxa partitions.
    if taxa_partition is None:
        taxa_partition = [
            # seq_data row indices of the taxa that contains the species name in the network
            [idx for idx, taxon in enumerate(seq_data.taxon_namespace) if label in taxon.label]
            for label in get_taxa_labels(phylox_network)
        ]

    # Step 2: Get all one-individual-per-species mappings (row indices of seq_data) according to taxa_partition.
    all_1indivPerSpecies_maps = product(*taxa_partition)

    # Step 3: Get all possible quartet features from self
    all_param_idx, all_is_asymm, all_taxa_perm, all_gamma_id = get_all_quartet_features(phylox_network, major_tree)

    # Step 4: Pair up quartet features (param_idx, is_asymm, taxa_perm, gamma_id) with the corresponding quartet_rows
    dict_quartet_param_idx = {}
    dict_quartet_is_asymm = {}
    dict_quartet_taxa_perm = {}
    dict_quartet_gamma_id = {}
    for MAP in all_1indivPerSpecies_maps:
        # For each all_1indivPerSpecies_maps, enumerate all possible combinations of 4 row indices of seq_data
        # (quartet_rows).
        for j, quartet_rows in enumerate(combinations(MAP, 4)):
            # Since quartet_rows will appear twice, we use it as dictionary key to collapse repetition.
            dict_key = tuple(quartet_rows)
            dict_quartet_param_idx[dict_key] = all_param_idx[j]
            dict_quartet_is_asymm[dict_key] = all_is_asymm[j]
            dict_quartet_taxa_perm[dict_key] = all_taxa_perm[j]
            dict_quartet_gamma_id[dict_key] = all_gamma_id[j]

    # Extract collapsed result from dictionary
    all_seq_data_rows        = list(dict_quartet_param_idx.keys())
    all_quartet_param_idx      = list(dict_quartet_param_idx.values())
    all_quartet_is_asymm        = list(dict_quartet_is_asymm.values())
    all_quartet_taxa_perm      = list(dict_quartet_taxa_perm.values())
    all_quartet_gamma_id       = list(dict_quartet_gamma_id.values())

    return all_seq_data_rows, all_quartet_param_idx, all_quartet_is_asymm, all_quartet_taxa_perm, all_quartet_gamma_id


def get_zippedData_net_reduced(seq_data, phylox_network, major_tree,
                   taxa_partition=None, ACGT_weight=None, skip_gap=False, skip_missing=False):
    """
    Output reduced quartet-level site pattern counts (n_Q) with corresponding quartet features used to compute
    true site pattern probabilities (p_Q) for each combination of four taxa. Also output E_mat and n_D used to
    compute the variability J and sensitivity H matrices.
    Reduced zippedData_net is used only to improve the computation time of SpeciesNetwork_CompLogLik.
    """

    from collections import defaultdict
    # -------------------------------------------------------------------------
    # 1. Get quartet features for all "one lineage per species" quartet subtree from data.
    # -------------------------------------------------------------------------
    (all_seq_data_rows, all_quartet_param_idx, all_quartet_is_asymm, all_quartet_taxa_perm,
     all_quartet_gamma_id) = pair_quartetRows_quartetFeatures(seq_data, phylox_network, major_tree, taxa_partition)

    # -------------------------------------------------------------------------
    # 2. Compute global site-pattern count with ambiguity handling
    # -------------------------------------------------------------------------
    collapsed_IDcode, n_D = get_n_D_hasAmbiguityCode(seq_data, ACGT_weight, skip_gap, skip_missing)

    # -------------------------------------------------------------------------
    # 3. Dictionary accumulation to collapse quartets with identical quartet features
    # -------------------------------------------------------------------------
    dict_site_pattern_count = defaultdict(int)
    dict_param_idx = {}
    dict_is_asymm = {}
    dict_taxa_perm = {}
    dict_gamma_id = {}
    dict_E_mat = defaultdict(list)

    def make_dict_key(param_idx, is_asymm, taxa_perm, gamma_id):
        """Make these quartet features as one dictionary key."""
        # reshape is_asymm into a column vector
        is_asymm_col = is_asymm[:, None]  # shape (2,1)

        # Stack row-wise: [param_idx | is_asymm | taxa_perm | gamma_id]
        combined = np.concatenate(
            [param_idx, is_asymm_col, taxa_perm, gamma_id],
            axis=1
        )

        return tuple(tuple(row) for row in combined)

    for row_idx, param_idx, is_asymm, taxa_perm, gamma_id in zip(all_seq_data_rows, all_quartet_param_idx,
                                                all_quartet_is_asymm, all_quartet_taxa_perm, all_quartet_gamma_id):
        # Get the quartet site pattern array by the row_idx
        quartet_sites = collapsed_IDcode[:, row_idx]

        # Get n_Q and its mapping matrix E_mat such that n_Q = E_mat @ n_D
        n_Q, E_mat = get_quartet_site_pattern_count_map_matrix(quartet_sites, n_D)

        # Convert everything to immutable tuples
        dict_key = make_dict_key(param_idx, is_asymm, taxa_perm, gamma_id)

        ## Use quartet features as key to get reduced quartet site pattern counts.
        dict_site_pattern_count[dict_key] += n_Q
        dict_param_idx[dict_key]    = param_idx
        dict_is_asymm[dict_key]      = is_asymm
        dict_taxa_perm[dict_key]     = taxa_perm
        dict_gamma_id[dict_key]     = gamma_id
        dict_E_mat[dict_key]        = E_mat

    # -------------------------------------------------------------------------
    # 4. Output exported reduced quartet information
    # -------------------------------------------------------------------------
    reduced_site_pattern_count   = np.array(list(dict_site_pattern_count.values()))
    reduced_param_idx           = list(dict_param_idx.values())
    reduced_is_asymm             = list(dict_is_asymm.values())
    reduced_taxa_perm            = list(dict_taxa_perm.values())
    reduced_gamma_id            = list(dict_gamma_id.values())
    reduced_E_mat               = list(dict_E_mat.values())

    return (list(zip(reduced_site_pattern_count, reduced_param_idx, reduced_is_asymm, reduced_taxa_perm, reduced_gamma_id)),
            reduced_E_mat, n_D)


def compress_zippedData_net(zipped_data_net, all_E_mat, n_D):
    """
    Output reduced quartet-level site pattern counts (n_Q) with corresponding quartet features.
    Takes the pre-computed zipped_data_net, all_E_mat, n_D from get_zippedData_net as input to avoid recalculating.
    """
    import numpy as np
    from collections import defaultdict

    dict_site_pattern_count = defaultdict(int)
    dict_param_idx = {}
    dict_is_asymm = {}
    dict_taxa_perm = {}
    dict_gamma_id = {}
    dict_E_mat = {}

    def make_dict_key(param_idx, is_asymm, taxa_perm, gamma_id):
        """Make these quartet features as one dictionary key."""
        # reshape is_asymm into a column vector
        is_asymm_col = is_asymm[:, None]  # shape (2,1)

        # Stack row-wise: [param_idx | is_asymm | taxa_perm | gamma_id]
        combined = np.concatenate(
            [param_idx, is_asymm_col, taxa_perm, gamma_id],
            axis=1
        )

        return tuple(tuple(row) for row in combined)

    # Loop directly over the outputs from get_zippedData_net
    for (n_Q, param_idx, is_asymm, taxa_perm, gamma_id), E_mat in zip(zipped_data_net, all_E_mat):
        # Convert everything to immutable tuples using your original logic
        dict_key = make_dict_key(param_idx, is_asymm, taxa_perm, gamma_id)

        # Use quartet features as key to get reduced quartet site pattern counts.
        dict_site_pattern_count[dict_key] += n_Q
        dict_param_idx[dict_key] = param_idx
        dict_is_asymm[dict_key] = is_asymm
        dict_taxa_perm[dict_key] = taxa_perm
        dict_gamma_id[dict_key] = gamma_id
        dict_E_mat[dict_key] = E_mat

    # -------------------------------------------------------------------------
    # Output exported reduced quartet information
    # -------------------------------------------------------------------------
    reduced_site_pattern_count = np.array(list(dict_site_pattern_count.values()))
    reduced_param_idx = list(dict_param_idx.values())
    reduced_is_asymm = list(dict_is_asymm.values())
    reduced_taxa_perm = list(dict_taxa_perm.values())
    reduced_gamma_id = list(dict_gamma_id.values())
    reduced_E_mat = list(dict_E_mat.values())

    return (list(zip(reduced_site_pattern_count, reduced_param_idx, reduced_is_asymm, reduced_taxa_perm, reduced_gamma_id)),
            reduced_E_mat, n_D)


def get_zippedData_net(seq_data, phylox_network, major_tree,
                       taxa_partition=None, ACGT_weight=None, skip_gap=False, skip_missing=False):
    """
    Output all quartet-level site pattern counts (n_Q) with corresponding quartet features used to compute true
    site pattern probabilities (p_Q) for each combination of four taxa. Also output E_mat and n_D used to compute
    the variability J and sensitivity H matrices.
    """

    # -------------------------------------------------------------------------
    # 1. Get quartet features for all "one lineage per species" quartet subtree from data.
    # -------------------------------------------------------------------------
    (all_seq_data_rows, all_quartet_param_idx, all_quartet_is_asymm, all_quartet_taxa_perm,
     all_quartet_gamma_id) = pair_quartetRows_quartetFeatures(seq_data, phylox_network, major_tree, taxa_partition)

    # -------------------------------------------------------------------------
    # 2. Compute global site-pattern count with ambiguity handling
    # -------------------------------------------------------------------------
    collapsed_IDcode, n_D = get_n_D_hasAmbiguityCode(seq_data, ACGT_weight, skip_gap, skip_missing)

    # -------------------------------------------------------------------------
    # 3. For all quartets, store its site pattern counts and its quartet features
    # -------------------------------------------------------------------------
    all_n_Q = []
    all_param_idx = []
    all_is_asymm = []
    all_taxa_perm = []
    all_gamma_id = []
    all_E_mat = []

    for row_idx, param_idx, is_asymm, taxa_perm, gamma_id in zip(all_seq_data_rows, all_quartet_param_idx,
                                                                all_quartet_is_asymm, all_quartet_taxa_perm,
                                                                all_quartet_gamma_id):
        # Get the quartet site pattern array by the row_idx
        quartet_sites = collapsed_IDcode[:, row_idx]

        # Get n_Q and its mapping matrix E_mat such that n_Q = E_mat @ n_D
        n_Q, E_mat = get_quartet_site_pattern_count_map_matrix(quartet_sites, n_D)

        all_n_Q.append(n_Q)
        all_param_idx.append(param_idx)
        all_is_asymm.append(is_asymm)
        all_taxa_perm.append(taxa_perm)
        all_gamma_id.append(gamma_id)
        all_E_mat.append(E_mat)

    # -------------------------------------------------------------------------
    # 4. Output full quartet information
    # -------------------------------------------------------------------------
    return list(zip(all_n_Q, all_param_idx, all_is_asymm, all_taxa_perm, all_gamma_id)), all_E_mat, n_D


def SpeciesNetwork_CompLogLik(zipped_data_net, parameters):
    """Computes species network composite log likelihood."""
    CompLogLik = 0
    for n_Q, param_idx, is_asymm, taxa_perm, gamma_id in zipped_data_net:
        TrueProbs = getTrueProbsQuartet(parameters, param_idx, is_asymm, taxa_perm, gamma_id)
        CompLogLik += np.sum(n_Q * np.log(TrueProbs))

    return CompLogLik


########################################################
## Find MCLE for species network composite likelihood ##
########################################################

def parameter_transform(parameters, labeled_network):
    """Transforms parameters into trans_param according to Kong et al. 2025.
    If tau_i/tau_A(i) >= 1, we set tau_i/tau_A(i) = 0.999 to guarantee parameters within constraints."""
    import numpy as np
    h = len(labeled_network.reticulations)  # Number of hybridization
    tree_parameters = parameters[:-h]
    gamma_parameters = parameters[-h:]
    num_t = len(tree_parameters) - 1
    trans_gamma = np.arcsin(np.sqrt(gamma_parameters))
    trans_tree = np.zeros(len(tree_parameters))
    for k, val in enumerate(tree_parameters):
        if k == 0 or k == num_t:
            trans_tree[k] = np.log(val)
        else:
            tau_idx = num_t - k
            node = labeled_network.labels[tau_idx][0]
            parent = next(labeled_network.predecessors(node))
            parent_val = tree_parameters[num_t - get_label_from_node(labeled_network, parent)]
            ratio = min(val / parent_val, 0.999)
            trans_tree[k] = np.arcsin(np.sqrt(ratio))
    return np.concatenate((trans_tree, trans_gamma))

def parameter_backtransform(trans_param, labeled_network):
    """Back-transforms trans_param into parameters according to Kong et al. 2025."""
    import numpy as np
    h = len(labeled_network.reticulations)  # Number of hybridization
    trans_tree = trans_param[:-h]
    trans_gamma = trans_param[-h:]
    num_t = len(trans_tree) - 1
    gamma = np.sin(trans_gamma) ** 2
    tree = np.zeros(len(trans_tree))
    for k, tv in enumerate(trans_tree):
        if k == 0 or k == num_t:
            tree[k] = np.exp(tv)
        else:
            tau_idx = num_t - k
            node = labeled_network.labels[tau_idx][0]
            parent = next(labeled_network.predecessors(node))
            parent_val = tree[num_t - get_label_from_node(labeled_network, parent)]
            tree[k] = parent_val * np.sin(tv) ** 2
    return np.concatenate((tree, gamma))


def get_MCLE_parameters(zipped_data_net, phylox_network, gamma_parameters=None, is_fixed_param=None,
                        MultiStart=None, Warning=False):
    """Get MCLE for self parameters = [tau_1,...,tau_J, theta, gamma_1,...,gamma_h].
    is_fixed_param decides which parameters are fixed for constrained optimization."""
    # ---------------------------------------
    # Imports packages and network informations
    # ---------------------------------------
    import numpy as np
    from scipy.optimize import minimize
    import warnings

    # Get labeled_network
    labeled_network = phylox_network.copy()
    label_speciation_time_idx(labeled_network)

    # Precompute basic network counts once
    N = len(labeled_network.leaves)  # Number of taxa
    h = len(labeled_network.reticulations)  # Number of hybridization
    J = N + h - 1  # Number of speciation times

    # ---------------------------------------
    # Given theta, get MOM estimators of parameters
    # ---------------------------------------
    # 1. Precompute MOM constant matrices ONCE
    # Symmetric case
    W_s = np.matrix([
        [4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 12, 0, 0, 0, 0, 0, 0, 0],
        [0, 12, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 12, 0, 0, 0, 0, 0, 0],
        [0, 0, 12, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 12, 0, 0, 0, 0, 0],
        [0, 0, 0, 12, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 12, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 24, 0],
        [0, 0, 0, 0, 0, 0, 0, 24, 0],
        [0, 0, 0, 0, 0, 0, 0, 24, 0],
        [0, 0, 0, 0, 0, 0, 0, 24, 0],
        [0, 0, 0, 0, 0, 24, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 24, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 24]
    ])
    Coef_s = np.matrix([
        [1, -2, 6, -2, 3, -2, 6, -8, -2],
        [1, 6, -2, -2, 3, 6, -2, -8, -2],
        [1, 2, 2, 2, -1, -2, -2, 0, -2]
    ])
    WS_pinv = np.linalg.inv(W_s.T @ W_s) @ W_s.T  # pseudo-inverse
    MOM_mat_s = Coef_s @ WS_pinv  # Precompute full transformation matrix

    # Asymmetric case
    W_a = np.matrix([
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0],
        [0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24]
    ])
    Coef_a = np.matrix([
        [3, -7, 10, 9, -5, 8, -8, 18, -10, -12, -6],
        [3, 5, -2, 9, 7, -4, -8, -6, -10, 12, -6],
        [3, 10, 5, -3, 2, 1, 2, -6, 4, -12, -6]
    ])
    WA_pinv = np.linalg.inv(W_a.T @ W_a) @ W_a.T
    MOM_mat_a = Coef_a @ WA_pinv

    # Pre-process zipped_data_net so we don't recalculate p_hat_Q during the theta search
    precomputed_data = []
    for n_Q, param_idx, is_asymm, taxa_perm, gamma_id in zipped_data_net:
        precomputed_data.append((n_Q / n_Q.sum(), param_idx, is_asymm, taxa_perm))

    # 2. Functions to get MOM of tau's
    mu = 4 / 3

    def get_MOM_tau_symm(p_hat_Q, theta):
        "Returns MOM estimators of [tau1, tau2, tau3] of symmetric quartet. tau3 is root age."
        y = np.array(4 * (1 + mu * 2 * theta) * (MOM_mat_s @ p_hat_Q)).ravel()  # convert 2D matrix to 1D array
        MOM_tau = -np.log(y ** (1 / (2 * mu)))
        return MOM_tau

    def get_MOM_tau_asymm(p_hat_Q, theta):
        "Returns MOM estimators of [tau1, tau2, tau3] of asymmetric quartet. tau3 is root age."
        y = np.array((4 / 3) * (1 + mu * 2 * theta) * (MOM_mat_a @ p_hat_Q)).ravel()
        MOM_tau = -np.log(y ** (1 / (2 * mu)))
        return MOM_tau

    # 3. Get parameters
    def get_MOM_tree_param(theta):
        tau_sum = np.zeros(J)
        tau_count = np.zeros(J)

        for p_hat_Q, param_idx, is_asymm, taxa_perm in precomputed_data:
            for pi, ia, tp in zip(param_idx, is_asymm, taxa_perm):
                tau_idx = (J - pi)[:3]  # convert to 0-based indexing
                SP_relation = get_site_pattern_relationship(np.argsort(tp))
                tau_vals = (get_MOM_tau_asymm if ia else get_MOM_tau_symm)(p_hat_Q[SP_relation], theta)

                # Accumulate the reversed values
                tau_sum[tau_idx] += tau_vals[::-1]
                tau_count[tau_idx] += 1

        tau_est = tau_sum / tau_count
        return np.append(tau_est, theta)

    # ---------------------------------------
    # Theta bound search for optimization
    # ---------------------------------------
    _, _, tau_in_constraints = get_tau_prior_and_constraint(phylox_network)

    def get_feasible_theta_bound(gamma_parameters):
        def f_star_func(theta, gamma_parameters):
            """Composite likelihood wrapper given only theta as input parameter."""
            raw_tree_param = get_MOM_tree_param(theta)
            if np.any(raw_tree_param < 0) or not tau_in_constraints(raw_tree_param[:-1]):
                return np.inf
            raw_param = np.concatenate((raw_tree_param, gamma_parameters))
            try:
                return -SpeciesNetwork_CompLogLik(zipped_data_net, raw_param)
            except:
                return np.inf

        # Golden section search
        a = 1e-5
        b = 1e-4
        while np.all(get_MOM_tree_param(b) > 0):
            b *= 10

        golden = 2 / (np.sqrt(5) + 1)
        t1 = b - golden * (b - a)
        t2 = a + golden * (b - a)
        f1 = f_star_func(t1, gamma_parameters)
        f2 = f_star_func(t2, gamma_parameters)

        while b - a > .01:
            if f2 > f1 or f2 == np.inf:
                b, t2, f2 = t2, t1, f1
                t1 = b - golden * (b - a)
                f1 = f_star_func(t1, gamma_parameters)
            else:
                a, t1, f1 = t1, t2, f2
                t2 = a + golden * (b - a)
                f2 = f_star_func(t2, gamma_parameters)

        return [a, b]

    # ---------------------------------------
    # Expand compressed vector after constrained optimization
    # ---------------------------------------
    def expand_trans_param(comp, mask):
        """Expand compressed vector after constrained optimization"""
        out = np.zeros(len(mask))
        out[~mask] = comp
        return out

    # ---------------------------------------
    # Negative log composite likelihood for optimizer
    # ---------------------------------------
    def neg_log_lik(x):
        if not np.all(np.isfinite(x)):
            return np.inf

        trans_param = expand_trans_param(x, is_fixed_param)
        params = parameter_backtransform(trans_param, labeled_network)

        if not np.all(np.isfinite(params)) or np.any(params < 0):
            return np.inf

        try:
            return -SpeciesNetwork_CompLogLik(zipped_data_net, params)
        except FloatingPointError:
            print("Floating point error at x =", x)
            raise

    # ---------------------------------------
    # Generate 10 Random Valid Starting Points
    # ---------------------------------------
    def get_random_starts(fixed_gammas, num_starts):
        starts = []
        attempts = 0

        while len(starts) < num_starts and attempts < num_starts * 10:
            attempts += 1
            # 1. Randomize theta (log-uniform to safely cover magnitudes from 1e-5 to 0.1)
            rand_theta = np.exp(np.random.uniform(np.log(1e-5), np.log(0.1)))
            raw_tree_param = get_MOM_tree_param(rand_theta)

            # 2. Randomize Gammas (or use fixed ones if constrained)
            if fixed_gammas is not None:
                current_gamma = fixed_gammas.copy()
            else:
                current_gamma = np.random.uniform(0.1, 0.9, size=h)

            raw_param = np.concatenate((raw_tree_param, current_gamma))

            # 3. Transform to unconstrained space
            try:
                trans_param = parameter_transform(raw_param, labeled_network)
            except Exception:
                continue

            # 4. Add "jitter" (Gaussian noise) to spread starting points across the surface
            jitter = np.random.normal(0, 0.5, size=J)
            trans_param[:J] += jitter
            x0 = trans_param[~is_fixed_param]

            # 5. Verify it evaluates to a finite likelihood
            if np.isfinite(neg_log_lik(x0)):
                starts.append(x0)

        # Fallback: if constraints are incredibly tight, inject completely random vectors
        while len(starts) < num_starts:
            x0 = np.random.normal(0, 1.0, size=np.sum(~is_fixed_param))
            if np.isfinite(neg_log_lik(x0)):
                starts.append(x0)

        return starts

    # ---------------------------------------
    # Multi-start BFGS Optimization Runner
    # ---------------------------------------
    def run_multistart_optimization(gamma_parameters, num_starts):
        if isinstance(num_starts, bool) or not isinstance(num_starts, (int, float, np.number)):
            num_starts = 10

        valid_starts = get_random_starts(gamma_parameters, num_starts)

        best_res = None
        best_lik = np.inf

        for i, x0 in enumerate(valid_starts):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                # Pass 1: Get into the immediate neighborhood
                res1 = minimize(neg_log_lik, x0, method="BFGS", options=dict(gtol=1e-5, maxiter=5000))

                # Pass 2: "Polish" run to cleanly snap to the peak
                res2 = minimize(neg_log_lik, res1.x, method="BFGS", options=dict(gtol=1e-7, maxiter=10000))

                # Keep track of the lowest negative log-likelihood found
                if np.isfinite(res2.fun) and res2.fun < best_lik:
                    best_lik = res2.fun
                    best_res = res2

        if best_res is None:
            raise RuntimeError("All unconstrained multi-start optimization attempts failed.")

        if Warning and not best_res.success:
            print(f"Warning: Best optimizer run did not fully converge. Message: {best_res.message}")

        return best_res

    # ---------------------------------------
    # Fixed-start BFGS Optimization Runner
    # ---------------------------------------
    def run_fixstart_optimization(gamma_parameters, current_is_fixed_param):
        """Run the core optimization steps for a given set of gamma parameters."""
        # Step 1: Get theta bound and initial point of parameters
        theta_bound = get_feasible_theta_bound(gamma_parameters)
        initial_theta = np.mean(theta_bound)
        initial_tree_param = get_MOM_tree_param(initial_theta)
        initial_trans_param = parameter_transform(np.concatenate((initial_tree_param, gamma_parameters)),
                                                  labeled_network)

        # Step 2: Constrained optimization using BFGS method
        x0 = initial_trans_param[~current_is_fixed_param]

        # Pass 1
        res1 = minimize(neg_log_lik, x0, method="BFGS",
                        options=dict(gtol=1e-5, maxiter=5000))

        # Pass 2: "Polish" run.
        res2 = minimize(neg_log_lik, res1.x, method="BFGS",
                        options=dict(gtol=1e-7, maxiter=10000))

        if not res2.success and Warning:
            print(f"Warning: Optimizer did not fully converge! Message: {res2.message}")
            if hasattr(res2, 'jac'):
                print(f"Max gradient remaining: {np.max(np.abs(res2.jac))}")

        return res2

    # ---------------------------------------
    # Setup Optimization Constraints & Execute
    # ---------------------------------------
    if is_fixed_param is not None:
        # --- CONSTRAINED OPTIMIZATION ---
        gamma_parameters = np.where(is_fixed_param[-h:], 0, 0.5)

        if not (MultiStart is None or MultiStart is False):
            result = run_multistart_optimization(gamma_parameters, MultiStart)
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", RuntimeWarning)
                result = run_fixstart_optimization(gamma_parameters, is_fixed_param)

            if any(issubclass(wi.category, RuntimeWarning) for wi in w):
                print("Constrained optimization produced RuntimeWarning(s); results may be unstable.")

    else:
        # --- UNCONSTRAINED OPTIMIZATION ---
        is_fixed_param = np.full(J + 1 + h, False)

        if not (MultiStart is None or MultiStart is False):
            result = run_multistart_optimization(gamma_parameters, MultiStart)
        else:
            if gamma_parameters is not None:
                result = run_fixstart_optimization(gamma_parameters, is_fixed_param)
            else:
                # Allow multiple retry gammas to get minimize()
                gamma_try = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
                for g in gamma_try:
                    current_gamma = np.where(is_fixed_param[-h:], 0, g)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", RuntimeWarning)
                            result = run_fixstart_optimization(current_gamma, is_fixed_param)
                        break  # If we reached here, optimization succeeded
                    except RuntimeWarning:
                        print(f"Unconstrained optimization failed due to numerical instability (gamma = {g})")
                else:
                    raise RuntimeError("All unconstrained optimization attempts failed due to numerical issues.")

    # 4. Export optimized estimators and back-transform them
    trans_estimator = result.x
    comp_loglik = -result.fun
    full_trans = expand_trans_param(trans_estimator, is_fixed_param)
    estimator = parameter_backtransform(full_trans, labeled_network)

    return estimator, comp_loglik

###############################################################################
## Diagnosing quartet information for computing network composite likelihood ##
###############################################################################

def printQuartet(zipped_data_net, phylox_network, parameters):
    """Print all informations about every quartet subnetworks for checking errors."""
    from itertools import combinations

    # Get all taxa_labels
    all_taxa_labels = combinations([str(label) for label in get_taxa_labels(phylox_network)], 4)

    for taxa_labels, (n_Q, param_idx, is_asymm, taxa_perm, gamma_id) in zip(all_taxa_labels, zipped_data_net):
        p_Q = getTrueProbsQuartet(parameters, param_idx, is_asymm, taxa_perm, gamma_id)
        CompLogLik = np.sum(n_Q * np.log(p_Q))
        print(f"""
Quartet taxa names: {taxa_labels}
Site pattern counts: {n_Q}
Site pattern frequencies: {n_Q / sum(n_Q)}
True Probs: {p_Q}
Parameter indices: {list(param_idx)}
is_asymm: {is_asymm}
Sort order: {list(taxa_perm)}
Gamma indices: {list(gamma_id)}
Quartet likelihood: {CompLogLik}""")


###########################################################################################
## Compute variability matrix J, sensitivity matrix H, and curvature adjustment matrix C ##
###########################################################################################

def get_Grad_Hess_Quartet(parameters, n_Q, param_idx, is_asymm, taxa_perm, gamma_id):
    """Get the gradient vector and Hessian matrix of the quartet likelihood with quartet features (param_idx, is_asymm,
    taxa_perm, gamma_id) given parameters."""
    import itertools

    # Our goal: Get (i) gradient vector and (ii) Hessian matrix of the quartet network log likelihood
    # Given p_Q = Σ[Γ_t * Ω_t @ p_{D^Q_t}], we need to 1) get the first and second derivatives of
    # [Γ_t * Ω_t @ p_{D^Q_t}] when D^Q_t is symmetric and asymmetric quartet, 2) accumulate these
    # derivatives to get the first and second derivatives of p_Q, and 3) get the gradient and Hessian
    # according to formulas in my Appendix.

    # ---- Step 1: Get the first and second derivatives of [Γ_t * Ω_t @ p_{D^Q_t}] ----
    def Qt_1st_2nd_deriv_Symm(tree_parameters, gamma_parameters, pi, tp, gi):
        """(pi = param_idx, ia = is_asymm, tp = taxa_perm, gi = gamma_id)
        For Q_t with len(gamma_id) = r, the length of parameters for this Q_t is 3 + 1 + r (tau's + theta + gamma's).
        Get the first order derivatives 15x(4+r) of [Γ_t * Ω_t @ p_{D^Q_t}], and
        get the second order derivatives (4+r)x(4+r)x15 of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is symmetric."""

        # Prepare quartet parameters used to compute site pattern probabilities
        t1, t2, t3, theta_tilde = get_tau_theta(tree_parameters, pi)
        t = 2 * theta_tilde  # 2*theta.tilde
        m = 4 / 3  # mu for JC69

        # 15-categ TrueProbs of quartet p_{D^Q_t} pulled from D_t
        p_Qt = np.matrix(getTrueProbsSymm(t1, t2, t3, theta_tilde, m)).T
        # get gamma weight for Q_t (Γ_t)
        Gamma_t = get_gamma_weight(gamma_parameters, gi)
        # Get 15 category permutation matrix (Ω_t)
        Omega = get_site_pattern_map_matrix(tp)

        # Our goal: (i) first order derivatives 15x(4+r) of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is symmetric and
        # (ii) second order derivatives (4+r)x(4+r)x15 of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is symmetric
        # We need: 1) beta vector, 2) W_s weight matrix, 3) C coefficient matrix, 4) dC/d(theta) 1st order derivative of
        # the C coefficient matrix w.r.t. theta, 5) DC/D(theta) 2nd order derivative the C coefficient matrix w.r.t. theta

        # 1) beta vector
        beta = np.array(
            [1, np.exp(-2 * m * t1), np.exp(-2 * m * t2), np.exp(-2 * m * t1 - 2 * m * t2), np.exp(-2 * m * t3),
             np.exp(-m * t1 - 2 * m * t3),
             np.exp(-m * t2 - 2 * m * t3), np.exp(-m * t1 - m * t2 - 2 * m * t3),
             np.exp(2 * t1 / t + 2 * t2 / t - 4 * t3 * (m + 1 / t))])

        # 2) W_s weight matrix
        W_s = np.matrix([[4, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 12, 0, 0, 0, 0, 0, 0, 0],
                         [0, 12, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 12, 0, 0, 0, 0, 0, 0],
                         [0, 0, 12, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 12, 0, 0, 0, 0, 0],
                         [0, 0, 0, 12, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 12, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 24, 0],
                         [0, 0, 0, 0, 0, 0, 0, 24, 0],
                         [0, 0, 0, 0, 0, 0, 0, 24, 0],
                         [0, 0, 0, 0, 0, 0, 0, 24, 0],
                         [0, 0, 0, 0, 0, 24, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 24, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 24]])

        # 3) C coefficient matrix
        C = np.matrix(np.zeros((9, 9)))
        # Row 0 (R's row 1)
        C[0, :] = 1 / 256
        # Row 1 (R's row 2)
        C[1, [0, 2, 4, 6]] = 3 / (256 * (1 + m * t))
        C[1, [1, 3, 5, 7, 8]] = -1 / (256 * (1 + m * t))
        # Row 2 (R's row 3)
        C[2, [0, 1, 4, 5]] = 3 / (256 * (1 + m * t))
        C[2, [2, 3, 6, 7, 8]] = -1 / (256 * (1 + m * t))
        # Row 3 (R's row 4)
        C[3, [0, 4]] = 9 / (256 * (1 + m * t) ** 2)
        C[3, [1, 2, 5, 6]] = -3 / (256 * (1 + m * t) ** 2)
        C[3, [3, 7, 8]] = 1 / (256 * (1 + m * t) ** 2)
        # Row 4 (R's row 5)
        C[4, 0] = 12 / (256 * (1 + m * t))
        C[4, [1, 2, 3]] = 4 / (256 * (1 + m * t))
        C[4, [4, 5, 6, 8]] = -4 / (256 * (1 + m * t))
        C[4, 7] = 0
        # Row 5 (R's row 6)
        C[5, 0] = 24 / (256 * (1 + m * t) * (2 + m * t))
        C[5, [1, 3, 4, 6]] = -8 / (256 * (1 + m * t) * (2 + m * t))
        C[5, 2] = 8 / (256 * (1 + m * t) * (2 + m * t))
        C[5, [5, 8]] = 8 / (256 * (1 + m * t) * (2 + m * t))
        C[5, 7] = 0
        # Row 6 (R's row 7)
        C[6, 0] = 24 / (256 * (1 + m * t) * (2 + m * t))
        C[6, 1] = 8 / (256 * (1 + m * t) * (2 + m * t))
        C[6, [2, 3, 4, 5]] = -8 / (256 * (1 + m * t) * (2 + m * t))
        C[6, [6, 8]] = 8 / (256 * (1 + m * t) * (2 + m * t))
        C[6, 7] = 0
        # Row 7 (R's row 8)
        C[7, 0] = 48 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        C[7, [1, 2, 4]] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        C[7, [3, 5, 6]] = 16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        C[7, 7] = 0
        C[7, 8] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        # Row 8 (R's row 9)
        C[8, 0] = 6 * m * t * (4 + m * t) * (4 + 3 * m * t) / (
                256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        C[8, [1, 2]] = -2 * m * t * (4 + m * t) * (4 + 3 * m * t) / (
                256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        C[8, 3] = m * t * (32 + 40 * m * t + 10 * (m ** 2) * (t ** 2)) / (
                256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        C[8, 4] = 2 * m * t * (4 + m * t) ** 2 / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        C[8, [5, 6]] = 2 * (m ** 2) * (t ** 2) * (4 + m * t) / (
                256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        C[8, 7] = -(m ** 2) * (t ** 2) * (4 + 2 * m * t) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))
        C[8, 8] = 2 * (m ** 3) * (t ** 3) / (256 * ((1 + m * t) ** 2) * ((2 + m * t) ** 2) * (3 + m * t))

        # 4) dC/d(theta) 1st order derivative
        dC_dt = np.matrix(np.zeros((9, 9)))
        # Row 0 (R's row 1)
        dC_dt[0, :] = 0
        # Row 1 (R's row 2)
        dC_dt[1, [0, 2, 4, 6]] = -3 * m / (256 * (1 + m * t) ** 2)
        dC_dt[1, [1, 3, 5, 7, 8]] = m / (256 * (1 + m * t) ** 2)
        # Row 2 (R's row 3)
        dC_dt[2, [0, 1, 4, 5]] = -3 * m / (256 * (1 + m * t) ** 2)
        dC_dt[2, [2, 3, 6, 7, 8]] = m / (256 * (1 + m * t) ** 2)
        # Row 3 (R's row 4)
        dC_dt[3, [0, 4]] = -18 * m / (256 * (1 + m * t) ** 3)
        dC_dt[3, [1, 2, 5, 6]] = 6 * m / (256 * (1 + m * t) ** 3)
        dC_dt[3, [3, 7, 8]] = -2 * m / (256 * (1 + m * t) ** 3)
        # Row 4 (R's row 5)
        dC_dt[4, 0] = -12 * m / (256 * (1 + m * t) ** 2)
        dC_dt[4, [1, 2, 3]] = -4 * m / (256 * (1 + m * t) ** 2)
        dC_dt[4, [4, 5, 6, 8]] = 4 * m / (256 * (1 + m * t) ** 2)
        dC_dt[4, 7] = 0
        # Row 5 (R's row 6)
        dC_dt[5, 0] = -24 * m * (2 * m * t + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[5, [1, 3, 4, 6]] = 8 * m * (2 * m * t + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[5, 2] = -8 * m * (2 * m * t + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[5, [5, 8]] = -8 * m * (2 * m * t + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[5, 7] = 0
        # Row 6 (R's row 7)
        dC_dt[6, 0] = -24 * m * (2 * m * t + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[6, 1] = -8 * m * (2 * m * t + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[6, [2, 3, 4, 5]] = 8 * m * (2 * m * t + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[6, [6, 8]] = -8 * m * (2 * m * t + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[6, 7] = 0
        # Row 7 (R's row 8)
        dC_dt[7, 0] = -48 * m * (3 * m * t + 4) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 3)
        dC_dt[7, [1, 2, 4]] = 16 * m * (3 * m * t + 4) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 3)
        dC_dt[7, [3, 5, 6]] = -16 * m * (3 * m * t + 4) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 3)
        dC_dt[7, 7] = 0
        dC_dt[7, 8] = 16 * m * (3 * m * t + 4) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 3)
        # Row 8 (R's row 9)
        mt = m * t  # for ease of notation
        dC_dt[8, 0] = -6 * m * (mt ** 2 * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) + 4 * mt ** 2 * (
                    2 * mt ** 3 + 3 * mt ** 2 - 13 * mt - 18)
                                + 16 * (3 * mt ** 4 + 13 * mt ** 3 + 13 * mt ** 2 - 3 * mt - 6)) / (
                              256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[8, [1, 2]] = 2 * m * (mt ** 2 * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) + 4 * mt ** 2 * (
                    2 * mt ** 3 + 3 * mt ** 2 - 13 * mt - 18)
                                    + 16 * (3 * mt ** 4 + 13 * mt ** 3 + 13 * mt ** 2 - 3 * mt - 6)) / (
                                   256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[8, 3] = m * (2 * mt ** 2 * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) + 8 * mt * (
                    2 * mt ** 4 + 6 * mt ** 3 - 4 * mt ** 2 - 20 * mt - 12)
                           + 16 * (-2 * mt ** 5 - 12 * mt ** 4 - 22 * mt ** 3 - 6 * mt ** 2 + 18 * mt + 12)) / (
                              256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[8, 4] = -2 * m * (mt ** 2 * (2 * mt ** 3 + 3 * mt ** 2 - 13 * mt - 18) + 8 * mt * (
                    3 * mt ** 3 + 9 * mt ** 2 - 2 * mt - 12)
                                + 16 * (4 * mt ** 3 + 15 * mt ** 2 + 9 * mt - 6)) / (
                              256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[8, [5, 6]] = 2 * (m ** 2) * t * (4 * (-3 * mt ** 3 - 9 * mt ** 2 + 2 * mt + 12) + mt * (
                    -2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18)) / (
                                   256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[8, 7] = 2 * (m ** 2) * t * (mt * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) + 4 * (
                    mt ** 4 + 3 * mt ** 3 - 2 * mt ** 2 - 10 * mt - 6)) / (
                              256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[8, 8] = 2 * (m ** 3) * (t ** 2) * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) / (
                256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)

        # 5) DC/D(theta) 2nd order derivative
        mt = m * t  # for ease of notation
        DC_Dt = np.matrix(np.zeros((9, 9)))
        # Row 0 (R's row 1)
        DC_Dt[0, :] = 0
        # Row 1 (R's row 2)
        DC_Dt[1, [0, 2, 4, 6]] = 6 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[1, [1, 3, 5, 7, 8]] = -2 * m ** 2 / (256 * (1 + m * t) ** 3)
        # Row 2 (R's row 3)
        DC_Dt[2, [0, 1, 4, 5]] = 6 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[2, [2, 3, 6, 7, 8]] = -2 * m ** 2 / (256 * (1 + m * t) ** 3)
        # Row 3 (R's row 4)
        DC_Dt[3, [0, 4]] = 54 * m ** 2 / (256 * (1 + m * t) ** 4)
        DC_Dt[3, [1, 2, 5, 6]] = -18 * m ** 2 / (256 * (1 + m * t) ** 4)
        DC_Dt[3, [3, 7, 8]] = 6 * m ** 2 / (256 * (1 + m * t) ** 4)
        # Row 4 (R's row 5)
        DC_Dt[4, 0] = 24 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[4, [1, 2, 3]] = 8 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[4, [4, 5, 6, 8]] = -8 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[4, 7] = 0
        # Row 5 (R's row 6)
        DC_Dt[5, 0] = 48 * m ** 2 * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[5, [1, 3, 4, 6]] = -16 * m ** 2 * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[5, 2] = 16 * m ** 2 * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[5, [5, 8]] = 16 * m ** 2 * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[5, 7] = 0
        # Row 6 (R's row 7)
        DC_Dt[6, 0] = 48 * m ** 2 * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[6, 1] = 16 * m ** 2 * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[6, [2, 3, 4, 5]] = -16 * m ** 2 * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[6, [6, 8]] = 16 * m ** 2 * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[6, 7] = 0
        # Row 7 (R's row 8)
        DC_Dt[7, 0] = 96 * m ** 2 * (6 * mt ** 2 + 16 * mt + 11) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 4)
        DC_Dt[7, [1, 2, 4]] = -32 * m ** 2 * (6 * mt ** 2 + 16 * mt + 11) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 4)
        DC_Dt[7, [3, 5, 6]] = 32 * m ** 2 * (6 * mt ** 2 + 16 * mt + 11) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 4)
        DC_Dt[7, 7] = 0
        DC_Dt[7, 8] = 32 * m ** 2 * (6 * mt ** 2 + 16 * mt + 11) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 4)
        # Row 8 (R's row 9)
        DC_Dt[8, 0] = 12 * m ** 2 * (
                    mt * (-3 * mt ** 6 - 9 * mt ** 5 + 43 * mt ** 4 + 201 * mt ** 3 + 212 * mt ** 2 - 36 * mt - 108)
                    + 4 * mt * (
                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)
                    + 16 * (
                                6 * mt ** 6 + 46 * mt ** 5 + 117 * mt ** 4 + 77 * mt ** 3 - 123 * mt ** 2 - 207 * mt - 84)) / (
                              256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[8, [1, 2]] = -4 * m ** 2 * (
                    mt * (-3 * mt ** 6 - 9 * mt ** 5 + 43 * mt ** 4 + 201 * mt ** 3 + 212 * mt ** 2 - 36 * mt - 108)
                    + 4 * mt * (
                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)
                    + 16 * (
                                6 * mt ** 6 + 46 * mt ** 5 + 117 * mt ** 4 + 77 * mt ** 3 - 123 * mt ** 2 - 207 * mt - 84)) / (
                                   256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[8, 3] = 2 * m ** 2 * (
                    2 * mt * (3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)
                    + 16 * (
                                3 * mt ** 7 + 27 * mt ** 6 + 85 * mt ** 5 + 75 * mt ** 4 - 166 * mt ** 3 - 462 * mt ** 2 - 414 * mt - 132)
                    - 8 * (
                                3 * mt ** 7 + 15 * mt ** 6 - 7 * mt ** 5 - 159 * mt ** 4 - 320 * mt ** 3 - 216 * mt ** 2 + 36)) / (
                              256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[8, 4] = 4 * m ** 2 * (16 * (10 * mt ** 5 + 75 * mt ** 4 + 185 * mt ** 3 + 129 * mt ** 2 - 99 * mt - 120)
                                    + 48 * (
                                                mt ** 6 + 6 * mt ** 5 + 7 * mt ** 4 - 18 * mt ** 3 - 42 * mt ** 2 - 18 * mt + 6)
                                    + mt * (
                                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)) / (
                              256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[8, [5, 6]] = 4 * m ** 2 * (
                    24 * (mt ** 6 + 6 * mt ** 5 + 7 * mt ** 4 - 18 * mt ** 3 - 42 * mt ** 2 - 18 * mt + 6)
                    + mt * (
                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)) / (
                                   256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[8, 7] = -2 * m ** 2 * (
                    4 * (3 * mt ** 7 + 15 * mt ** 6 - 7 * mt ** 5 - 159 * mt ** 4 - 320 * mt ** 3 - 216 * mt ** 2 + 36)
                    - 2 * mt * (
                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)) / (
                              256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[8, 8] = 4 * (m ** 3) * t * (
                    3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108) / (
                              256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)

        # Compute output first and second derivatives
        # First derivatives of tau's
        delta_t1 = np.array([0, -2 * m, 0, -2 * m, 0, -m, 0, -m, 2 / t])
        dp_dt1 = (W_s @ C.T @ (delta_t1 * beta)).T
        delta_t2 = np.array([0, 0, -2 * m, -2 * m, 0, 0, -m, -m, 2 / t])
        dp_dt2 = (W_s @ C.T @ (delta_t2 * beta)).T
        delta_t3 = np.array([0, 0, 0, 0, -2 * m, -2 * m, -2 * m, -2 * m, -4 * (m + 1 / t)])
        dp_dt3 = (W_s @ C.T @ (delta_t3 * beta)).T

        # First derivatives of theta
        delta_t = np.array([0, 0, 0, 0, 0, 0, 0, 0, (4 * t3 - 2 * (t1 + t2)) / (t ** 2)])
        dp_dt = W_s @ ((C.T @ (2 * delta_t * beta)).T + 2 * dC_dt.T @ np.matrix(beta).T)

        # First derivatives of gamma's
        dGt_dg = np.matrix(get_gamma_1st_deriv(gamma_parameters, gi))  # ∂Γ_t / ∂γ_i for i in gamma_id
        dGt_dg_p = p_Qt @ dGt_dg

        # ------ First derivatives 15x(4+r) of [Γ_t * Ω_t @ p_{D^Q_t}] ------
        r = gi.size
        columns = [Gamma_t * dp_dt1, Gamma_t * dp_dt2, Gamma_t * dp_dt3, Gamma_t * dp_dt]
        if r > 0: # Add the derivatives w.r.t. gamma's if r>0
            columns.append(dGt_dg_p)
        first_deriv = Omega @ np.hstack(columns)
        # -------------------------------------------------------------------

        # Second derivatives of tau's
        dp_dt1t1 = (W_s @ C.T @ (delta_t1 * delta_t1 * beta)).T
        dp_dt1t2 = (W_s @ C.T @ (delta_t1 * delta_t2 * beta)).T
        dp_dt1t3 = (W_s @ C.T @ (delta_t1 * delta_t3 * beta)).T
        dp_dt2t2 = (W_s @ C.T @ (delta_t2 * delta_t2 * beta)).T
        dp_dt2t3 = (W_s @ C.T @ (delta_t2 * delta_t3 * beta)).T
        dp_dt3t3 = (W_s @ C.T @ (delta_t3 * delta_t3 * beta)).T

        # Second derivatives of tau & theta
        delta_t1t = delta_t2t = np.array([0, 0, 0, 0, 0, 0, 0, 0, -2 * (t + 2 * t1 + 2 * t2 - 4 * t3) / (t ** 3)])
        delta_t3t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 4 * (t + (2 * t1 + 2 * t2 - 4 * t3) * (1 + m * t)) / (t ** 3)])
        dp_dt1t = W_s @ ((C.T @ (2 * delta_t1t * beta)).T + 2 * dC_dt.T @ np.matrix(delta_t1 * beta).T)
        dp_dt2t = W_s @ ((C.T @ (2 * delta_t2t * beta)).T + 2 * dC_dt.T @ np.matrix(delta_t2 * beta).T)
        dp_dt3t = W_s @ ((C.T @ (2 * delta_t3t * beta)).T + 2 * dC_dt.T @ np.matrix(delta_t3 * beta).T)

        # Second derivatives of tau & gamma
        dGt_dg_dp_dt1 = dp_dt1 @ dGt_dg
        dGt_dg_dp_dt2 = dp_dt2 @ dGt_dg
        dGt_dg_dp_dt3 = dp_dt3 @ dGt_dg

        # Second derivatives of theta
        delta_tt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 4 * (t1 + t2 - 2 * t3) * (t + t1 + t2 - 2 * t3) / (t ** 4)])
        dp_dtt = W_s @ ((4 * C.T @ (delta_tt * beta)).T + 8 * dC_dt.T @ np.matrix(
            delta_t * beta).T + 4 * DC_Dt.T @ np.matrix(beta).T)

        # Second derivatives of theta & gamma
        dGt_dg_dp_dt = dp_dt @ dGt_dg

        # Second derivatives of gamma's
        dGt_dgg = get_gamma_2nd_deriv(gamma_parameters, gi)  # ∂²Γ_t / (∂γ_i ∂γ_j) for i,j in gamma_id
        # dGt_dgg_p[i,j] = p_Qt * dGt_dgg[i, j] would be a 3D matrix with shape (r, r, 15)

        # ------ Second derivatives (4+r)x(4+r)x15 of [Γ_t * Ω_t @ p_{D^Q_t}] ------
        second_deriv = np.zeros((4+r, 4+r, 15))
        second_deriv[0, 0, :] = (Gamma_t * Omega @ dp_dt1t1).flatten()
        second_deriv[0, 1, :] = (Gamma_t * Omega @ dp_dt1t2).flatten()
        second_deriv[0, 2, :] = (Gamma_t * Omega @ dp_dt1t3).flatten()
        second_deriv[0, 3, :] = (Gamma_t * Omega @ dp_dt1t).flatten()
        second_deriv[1, 1, :] = (Gamma_t * Omega @ dp_dt2t2).flatten()
        second_deriv[1, 2, :] = (Gamma_t * Omega @ dp_dt2t3).flatten()
        second_deriv[1, 3, :] = (Gamma_t * Omega @ dp_dt2t).flatten()
        second_deriv[2, 2, :] = (Gamma_t * Omega @ dp_dt3t3).flatten()
        second_deriv[2, 3, :] = (Gamma_t * Omega @ dp_dt3t).flatten()
        second_deriv[3, 3, :] = (Gamma_t * Omega @ dp_dtt).flatten()
        # Add the derivatives w.r.t. gamma's if r>0
        if r > 0:
            for i in range(r):
                gamma_idx = 4 + i  # This shifts the index to start at 4

                second_deriv[0, gamma_idx, :] = (Omega @ dGt_dg_dp_dt1[:, i]).flatten()
                second_deriv[1, gamma_idx, :] = (Omega @ dGt_dg_dp_dt2[:, i]).flatten()
                second_deriv[2, gamma_idx, :] = (Omega @ dGt_dg_dp_dt3[:, i]).flatten()
                second_deriv[3, gamma_idx, :] = (Omega @ dGt_dg_dp_dt[:, i]).flatten()
                second_deriv[gamma_idx, gamma_idx, :] = (Omega @ p_Qt * dGt_dgg[i, i]).flatten()
                # Cross-derivatives between different gammas
                for j in range(i + 1, r):
                    gamma_jdx = 4 + j
                    second_deriv[gamma_idx, gamma_jdx, :] = (Omega @ p_Qt * dGt_dgg[i, j]).flatten()
        # Mirror the upper triangle to the lower triangle across all 15 slices
        for i in range(4 + r):
            for j in range(i + 1, 4 + r):
                second_deriv[j, i, :] = second_deriv[i, j, :]
        # ---------------------------------------------------------------------------

        # ---- Output ----
        # first_deriv: first order derivatives 15x(4+r) of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is symmetric
        # second_deriv: second order derivatives (4+r)x(4+r)x15 of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is symmetric
        # ----------------
        return first_deriv, second_deriv

    def Qt_1st_2nd_deriv_Asymm(tree_parameters, gamma_parameters, pi, tp, gi):
        """(pi = param_idx, ia = is_asymm, tp = taxa_perm, gi = gamma_id)
        For Q_t with len(gamma_id) = r, the length of parameters for this Q_t is 3 + 1 + r (tau's + theta + gamma's).
        Get the first order derivatives 15x(4+r) of [Γ_t * Ω_t @ p_{D^Q_t}], and
        get the second order derivatives (4+r)x(4+r)x15 of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is asymmetric."""

        # Prepare quartet parameters used to compute site pattern probabilities
        t1, t2, t3, theta_tilde = get_tau_theta(tree_parameters, pi)
        t = 2 * theta_tilde  # 2*theta.tilde
        m = 4 / 3  # mu for JC69

        # 15-categ TrueProbs of quartet Q_t pulled from D_t
        p_Qt = np.matrix(getTrueProbsAsymm(t1, t2, t3, theta_tilde, m)).T
        # get gamma weight for Q_t (Gamma_t)
        Gamma_t = get_gamma_weight(gamma_parameters, gi)
        # Get 15 category permutation matrix (Ω_t)
        Omega = get_site_pattern_map_matrix(tp)

        # Our goal: (i) first order derivatives 15x(4+r) of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is asymmetric and
        # (ii) second order derivatives (4+r)x(4+r)x15 of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is asymmetric
        # We need: 1) beta vector, 2) W_s weight matrix, 3) C coefficient matrix, 4) dC/d(theta) 1st order derivative of
        # the C coefficient matrix w.r.t. theta, 5) DC/D(theta) 2nd order derivative the C coefficient matrix w.r.t. theta

        # 1) beta vector
        beta = np.array([1, np.exp(-2 * m * t1), np.exp(-2 * m * t2), np.exp(-m * t1 - 2 * m * t2), np.exp(-2 * m * t3),
                         np.exp(-m * t1 - 2 * m * t3),
                         np.exp(-2 * m * t1 - 2 * m * t3), np.exp(-m * t2 - 2 * m * t3),
                         np.exp(-m * t1 - m * t2 - 2 * m * t3), np.exp((t1 - t2) * 2 / t - 2 * m * (t2 + t3))])

        # 2) W_a weight matrix
        W_a = np.matrix([[4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0],
                         [0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24]])

        # 3) C coefficient matrix
        C = np.matrix(np.zeros((10, 11)))
        # Row 0 (R's row 1)
        C[0, :] = 1 / 256
        # Row 1 (R's row 2)
        C[1, [0, 2, 3, 4, 7]] = 3 / (256 * (1 + m * t))
        C[1, [1, 5, 6, 8, 9, 10]] = -1 / (256 * (1 + m * t))
        # Row 2 (R's row 3)
        C[2, [0, 3]] = 6 / (256 * (1 + m * t))
        C[2, [1, 5, 9]] = 2 / (256 * (1 + m * t))
        C[2, [2, 4, 6, 7, 8, 10]] = -2 / (256 * (1 + m * t))
        # Row 3 (R's row 4)
        C[3, [0, 3]] = 12 / (256 * (1 + m * t) * (2 + m * t))
        C[3, [1, 2, 4, 5, 7, 9]] = -4 / (256 * (1 + m * t) * (2 + m * t))
        C[3, [6, 8, 10]] = 4 / (256 * (1 + m * t) * (2 + m * t))
        # Row 4 (R's row 5)
        C[4, 0] = 9 / (256 * (1 + m * t))
        C[4, [1, 2]] = 5 / (256 * (1 + m * t))
        C[4, [3, 7, 9, 10]] = -3 / (256 * (1 + m * t))
        C[4, [4, 5, 6, 8]] = 1 / (256 * (1 + m * t))
        # Row 5 (R's row 6)
        C[5, [0, 2]] = 12 / (256 * (1 + m * t) * (2 + m * t))
        C[5, [1, 3, 4, 5, 7, 8]] = -4 / (256 * (1 + m * t) * (2 + m * t))
        C[5, [6, 9, 10]] = 4 / (256 * (1 + m * t) * (2 + m * t))
        # Row 6 (R's row 7)
        C[6, [0, 4]] = 9 / (256 * (1 + m * t) ** 2)
        C[6, [1, 2, 3, 6, 7]] = -3 / (256 * (1 + m * t) ** 2)
        C[6, [5, 8, 9, 10]] = 1 / (256 * (1 + m * t) ** 2)
        # Row 7 (R's row 8)
        C[7, 0] = 24 / (256 * (1 + m * t) * (2 + m * t))
        C[7, 1] = 8 / (256 * (1 + m * t) * (2 + m * t))
        C[7, [2, 3, 4, 5, 6]] = -8 / (256 * (1 + m * t) * (2 + m * t))
        C[7, [7, 10]] = 8 / (256 * (1 + m * t) * (2 + m * t))
        C[7, [8, 9]] = 0
        # Row 8 (R's row 9)
        C[8, 0] = 48 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        C[8, [1, 2, 3, 4]] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        C[8, [5, 6, 7]] = 16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        C[8, [8, 9]] = 0
        C[8, 10] = -16 / (256 * (1 + m * t) * (2 + m * t) ** 2)
        # Row 9 (R's row 10)
        C[9, 0] = (6 * m * t * (4 + m * t) * (4 + 3 * m * t)) / (
                    256 * (1 + m * t) ** 2 * (2 + m * t) ** 2 * (3 + m * t))
        C[9, [1, 2, 3]] = -(2 * m * t * (4 + m * t) * (4 + 3 * m * t)) / (
                    256 * (1 + m * t) ** 2 * (2 + m * t) ** 2 * (3 + m * t))
        C[9, 4] = (2 * m * t * (4 + m * t) ** 2) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2 * (3 + m * t))
        C[9, 5] = m * t * (2 * 16 + 40 * m * t + 10 * (m ** 2) * (t ** 2)) / (
                    256 * (1 + m * t) ** 2 * (2 + m * t) ** 2 * (3 + m * t))
        C[9, [6, 7]] = 2 * (m ** 2) * (t ** 2) * (4 + m * t) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2 * (3 + m * t))
        C[9, [8, 9]] = -(m ** 2) * (t ** 2) * (4 + 2 * m * t) / (
                    256 * (1 + m * t) ** 2 * (2 + m * t) ** 2 * (3 + m * t))
        C[9, 10] = 2 * (m ** 3) * (t ** 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2 * (3 + m * t))

        # 4) dC/d(theta)
        mt = m * t  # for ease of notation
        dC_dt = np.matrix(np.zeros((10, 11)))
        # Row 0 (R's row 1)
        dC_dt[0, :] = 0
        # Row 1 (R's row 2)
        dC_dt[1, [0, 2, 3, 4, 7]] = -3 * m / (256 * (1 + m * t) ** 2)
        dC_dt[1, [1, 5, 6, 8, 9, 10]] = m / (256 * (1 + m * t) ** 2)
        # Row 2 (R's row 3)
        dC_dt[2, [0, 3]] = -6 * m / (256 * (1 + m * t) ** 2)
        dC_dt[2, [1, 5, 9]] = -2 * m / (256 * (1 + m * t) ** 2)
        dC_dt[2, [2, 4, 6, 7, 8, 10]] = 2 * m / (256 * (1 + m * t) ** 2)
        # Row 3 (R's row 4)
        dC_dt[3, [0, 3]] = -12 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[3, [1, 2, 4, 5, 7, 9]] = 4 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[3, [6, 8, 10]] = -4 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        # Row 4 (R's row 5)
        dC_dt[4, 0] = -9 * m / (256 * (1 + m * t) ** 2)
        dC_dt[4, [1, 2]] = -5 * m / (256 * (1 + m * t) ** 2)
        dC_dt[4, [3, 7, 9, 10]] = 3 * m / (256 * (1 + m * t) ** 2)
        dC_dt[4, [4, 5, 6, 8]] = -1 * m / (256 * (1 + m * t) ** 2)
        # Row 5 (R's row 6)
        dC_dt[5, [0, 2]] = -12 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[5, [1, 3, 4, 5, 7, 8]] = 4 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[5, [6, 9, 10]] = -4 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        # Row 6 (R's row 7)
        dC_dt[6, [0, 4]] = -18 * m / (256 * (1 + m * t) ** 3)
        dC_dt[6, [1, 2, 3, 6, 7]] = 6 * m / (256 * (1 + m * t) ** 3)
        dC_dt[6, [5, 8, 9, 10]] = -2 * m / (256 * (1 + m * t) ** 3)
        # Row 7 (R's row 8)
        dC_dt[7, 0] = -24 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[7, 1] = -8 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[7, [2, 3, 4, 5, 6]] = 8 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[7, [7, 10]] = -8 * m * (2 * mt + 3) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 2)
        dC_dt[7, [8, 9]] = 0
        # Row 8 (R's row 9)
        dC_dt[8, 0] = -48 * m * (3 * mt + 4) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 3)
        dC_dt[8, [1, 2, 3, 4]] = 16 * m * (3 * mt + 4) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 3)
        dC_dt[8, [5, 6, 7]] = -16 * m * (3 * mt + 4) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 3)
        dC_dt[8, [8, 9]] = 0
        dC_dt[8, 10] = 16 * m * (3 * mt + 4) / (256 * (1 + m * t) ** 2 * (2 + m * t) ** 3)
        # Row 9 (R's row 10)
        dC_dt[9, 0] = -6 * m * (mt ** 2 * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) + 4 * mt ** 2 * (
                    2 * mt ** 3 + 3 * mt ** 2 - 13 * mt - 18)
                                + 16 * (3 * mt ** 4 + 13 * mt ** 3 + 13 * mt ** 2 - 3 * mt - 6)) / (
                              256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[9, [1, 2, 3]] = 2 * m * (mt ** 2 * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) + 4 * mt ** 2 * (
                    2 * mt ** 3 + 3 * mt ** 2 - 13 * mt - 18)
                                       + 16 * (3 * mt ** 4 + 13 * mt ** 3 + 13 * mt ** 2 - 3 * mt - 6)) / (
                                      256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[9, 4] = -2 * m * (mt ** 2 * (2 * mt ** 3 + 3 * mt ** 2 - 13 * mt - 18) + 8 * mt * (
                    3 * mt ** 3 + 9 * mt ** 2 - 2 * mt - 12)
                                + 16 * (4 * mt ** 3 + 15 * mt ** 2 + 9 * mt - 6)) / (
                              256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[9, 5] = m * (2 * mt ** 2 * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) + 8 * mt * (
                    2 * mt ** 4 + 6 * mt ** 3 - 4 * mt ** 2 - 20 * mt - 12)
                           + 16 * (-2 * mt ** 5 - 12 * mt ** 4 - 22 * mt ** 3 - 6 * mt ** 2 + 18 * mt + 12)) / (
                              256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[9, [6, 7]] = 2 * (m ** 2) * t * (4 * (-3 * mt ** 3 - 9 * mt ** 2 + 2 * mt + 12) + mt * (
                    -2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18)) / (
                                   256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[9, [8, 9]] = 2 * (m ** 2) * t * (mt * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) + 4 * (
                    mt ** 4 + 3 * mt ** 3 - 2 * mt ** 2 - 10 * mt - 6)) / (
                                   256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)
        dC_dt[9, 10] = 2 * (m ** 3) * (t ** 2) * (-2 * mt ** 3 - 3 * mt ** 2 + 13 * mt + 18) / (
                    256 * (1 + m * t) ** 3 * (2 + m * t) ** 3 * (3 + m * t) ** 2)

        # 5) DC/D(theta) 2nd order derivative
        mt = m * t  # for ease of notation
        DC_Dt = np.matrix(np.zeros((10, 11)))
        # Row 0 (R's row 1)
        DC_Dt[0, :] = 0
        # Row 1 (R's row 2)
        DC_Dt[1, [0, 2, 3, 4, 7]] = 3 * 2 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[1, [1, 5, 6, 8, 9, 10]] = -1 * 2 * m ** 2 / (256 * (1 + m * t) ** 3)
        # Row 2 (R's row 3)
        DC_Dt[2, [0, 3]] = 6 * 2 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[2, [1, 5, 9]] = 2 * 2 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[2, [2, 4, 6, 7, 8, 10]] = -2 * 2 * m ** 2 / (256 * (1 + m * t) ** 3)
        # Row 3 (R's row 4)
        DC_Dt[3, [0, 3]] = 12 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[3, [1, 2, 4, 5, 7, 9]] = -4 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (
                    256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[3, [6, 8, 10]] = 4 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (
                    256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        # Row 4 (R's row 5)
        DC_Dt[4, 0] = 9 * 2 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[4, [1, 2]] = 5 * 2 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[4, [3, 7, 9, 10]] = -3 * 2 * m ** 2 / (256 * (1 + m * t) ** 3)
        DC_Dt[4, [4, 5, 6, 8]] = 1 * 2 * m ** 2 / (256 * (1 + m * t) ** 3)
        # Row 5 (R's row 6)
        DC_Dt[5, [0, 2]] = 12 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[5, [1, 3, 4, 5, 7, 8]] = -4 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (
                    256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[5, [6, 9, 10]] = 4 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (
                    256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        # Row 6 (R's row 7)
        DC_Dt[6, [0, 4]] = 9 * 6 * m ** 2 / (256 * (1 + m * t) ** 4)
        DC_Dt[6, [1, 2, 3, 6, 7]] = -3 * 6 * m ** 2 / (256 * (1 + m * t) ** 4)
        DC_Dt[6, [5, 8, 9, 10]] = 1 * 6 * m ** 2 / (256 * (1 + m * t) ** 4)
        # Row 7 (R's row 8)
        DC_Dt[7, 0] = 24 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[7, 1] = 8 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[7, [2, 3, 4, 5, 6]] = -8 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (
                    256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[7, [7, 10]] = 8 * 2 * (m ** 2) * (3 * mt ** 2 + 9 * mt + 7) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 3)
        DC_Dt[7, [8, 9]] = 0
        # Row 8 (R's row 9)
        DC_Dt[8, 0] = 48 * 2 * (m ** 2) * (6 * mt ** 2 + 16 * mt + 11) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 4)
        DC_Dt[8, [1, 2, 3, 4]] = -16 * 2 * (m ** 2) * (6 * mt ** 2 + 16 * mt + 11) / (
                    256 * (1 + m * t) ** 3 * (2 + m * t) ** 4)
        DC_Dt[8, [5, 6, 7]] = 16 * 2 * (m ** 2) * (6 * mt ** 2 + 16 * mt + 11) / (
                    256 * (1 + m * t) ** 3 * (2 + m * t) ** 4)
        DC_Dt[8, [8, 9]] = 0
        DC_Dt[8, 10] = -16 * 2 * (m ** 2) * (6 * mt ** 2 + 16 * mt + 11) / (256 * (1 + m * t) ** 3 * (2 + m * t) ** 4)
        # Row 9 (R's row 10)
        DC_Dt[9, 0] = 12 * m ** 2 * (
                    mt * (-3 * mt ** 6 - 9 * mt ** 5 + 43 * mt ** 4 + 201 * mt ** 3 + 212 * mt ** 2 - 36 * mt - 108)
                    + 4 * mt * (
                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)
                    + 16 * (
                                6 * mt ** 6 + 46 * mt ** 5 + 117 * mt ** 4 + 77 * mt ** 3 - 123 * mt ** 2 - 207 * mt - 84)) / (
                              256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[9, [1, 2, 3]] = -4 * m ** 2 * (
                    mt * (-3 * mt ** 6 - 9 * mt ** 5 + 43 * mt ** 4 + 201 * mt ** 3 + 212 * mt ** 2 - 36 * mt - 108)
                    + 4 * mt * (
                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)
                    + 16 * (
                                6 * mt ** 6 + 46 * mt ** 5 + 117 * mt ** 4 + 77 * mt ** 3 - 123 * mt ** 2 - 207 * mt - 84)) / (
                                      256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[9, 4] = 4 * m ** 2 * (16 * (10 * mt ** 5 + 75 * mt ** 4 + 185 * mt ** 3 + 129 * mt ** 2 - 99 * mt - 120)
                                    + 48 * (
                                                mt ** 6 + 6 * mt ** 5 + 7 * mt ** 4 - 18 * mt ** 3 - 42 * mt ** 2 - 18 * mt + 6)
                                    + mt * (
                                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)) / (
                              256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[9, 5] = 2 * m ** 2 * (
                    2 * mt * (3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)
                    + 16 * (
                                3 * mt ** 7 + 27 * mt ** 6 + 85 * mt ** 5 + 75 * mt ** 4 - 166 * mt ** 3 - 462 * mt ** 2 - 414 * mt - 132)
                    - 8 * (
                                3 * mt ** 7 + 15 * mt ** 6 - 7 * mt ** 5 - 159 * mt ** 4 - 320 * mt ** 3 - 216 * mt ** 2 + 36)) / (
                              256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[9, [6, 7]] = 4 * m ** 2 * (
                    24 * (mt ** 6 + 6 * mt ** 5 + 7 * mt ** 4 - 18 * mt ** 3 - 42 * mt ** 2 - 18 * mt + 6)
                    + mt * (
                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)) / (
                                   256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[9, [8, 9]] = -2 * m ** 2 * (
                    4 * (3 * mt ** 7 + 15 * mt ** 6 - 7 * mt ** 5 - 159 * mt ** 4 - 320 * mt ** 3 - 216 * mt ** 2 + 36)
                    - 2 * mt * (
                                3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108)) / (
                                   256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)
        DC_Dt[9, 10] = 4 * (m ** 3) * t * (
                    3 * mt ** 6 + 9 * mt ** 5 - 43 * mt ** 4 - 201 * mt ** 3 - 212 * mt ** 2 + 36 * mt + 108) / (
                               256 * (1 + m * t) ** 4 * (2 + m * t) ** 4 * (3 + m * t) ** 3)

        # Compute output first and second derivatives
        # First derivatives of tau's
        delta_t1 = np.array([0, -2 * m, 0, -m, 0, -m, -2 * m, 0, -m, 2 / t])
        dp_dt1 = (W_a @ C.T @ (delta_t1 * beta)).T
        delta_t2 = np.array([0, 0, -2 * m, -2 * m, 0, 0, 0, -m, -m, -2 * (m + 1 / t)])
        dp_dt2 = (W_a @ C.T @ (delta_t2 * beta)).T
        delta_t3 = np.array([0, 0, 0, 0, -2 * m, -2 * m, -2 * m, -2 * m, -2 * m, -2 * m])
        dp_dt3 = (W_a @ C.T @ (delta_t3 * beta)).T

        # First derivatives of theta
        delta_t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * (t2 - t1) / (t ** 2)])
        dp_dt = W_a @ ((C.T @ (2 * delta_t * beta)).T + 2 * dC_dt.T @ np.matrix(beta).T)

        # First derivatives of gamma's
        dGt_dg = np.matrix(get_gamma_1st_deriv(gamma_parameters, gi))  # ∂Γ_t / ∂γ_i for i in gamma_id
        dGt_dg_p = p_Qt @ dGt_dg

        # ------ First derivatives 15x(4+r) of [Γ_t * Ω_t @ p_{D^Q_t}] ------
        r = gi.size
        columns = [Gamma_t * dp_dt1, Gamma_t * dp_dt2, Gamma_t * dp_dt3, Gamma_t * dp_dt]
        if r > 0:  # Add the derivatives w.r.t. gamma's if r>0
            columns.append(dGt_dg_p)
        first_deriv = Omega @ np.hstack(columns)
        # -------------------------------------------------------------------

        # Second derivatives of tau's
        dp_dt1t1 = (W_a @ C.T @ (delta_t1 * delta_t1 * beta)).T
        dp_dt1t2 = (W_a @ C.T @ (delta_t1 * delta_t2 * beta)).T
        dp_dt1t3 = (W_a @ C.T @ (delta_t1 * delta_t3 * beta)).T
        dp_dt2t2 = (W_a @ C.T @ (delta_t2 * delta_t2 * beta)).T
        dp_dt2t3 = (W_a @ C.T @ (delta_t2 * delta_t3 * beta)).T
        dp_dt3t3 = (W_a @ C.T @ (delta_t3 * delta_t3 * beta)).T

        # Second derivatives of tau & theta
        delta_t1t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * (t + 2 * t1 - 2 * t2) / (t ** 3)])
        delta_t2t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * (t + 2 * (1 + mt) * (t1 - t2)) / (t ** 3)])
        delta_t3t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 4 * m * (t1 - t2) / (t ** 2)])
        dp_dt1t = W_a @ ((C.T @ (2 * delta_t1t * beta)).T + 2 * dC_dt.T @ np.matrix(delta_t1 * beta).T)
        dp_dt2t = W_a @ ((C.T @ (2 * delta_t2t * beta)).T + 2 * dC_dt.T @ np.matrix(delta_t2 * beta).T)
        dp_dt3t = W_a @ ((C.T @ (2 * delta_t3t * beta)).T + 2 * dC_dt.T @ np.matrix(delta_t3 * beta).T)

        # Second derivatives of tau & gamma
        dGt_dg_dp_dt1 = dp_dt1 @ dGt_dg
        dGt_dg_dp_dt2 = dp_dt2 @ dGt_dg
        dGt_dg_dp_dt3 = dp_dt3 @ dGt_dg

        # Second derivatives of theta
        delta_tt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 4 * (t1 - t2) * (t + t1 - t2) / (t ** 4)])
        dp_dtt = W_a @ ((4 * C.T @ (delta_tt * beta)).T + 8 * dC_dt.T @ np.matrix(
            delta_t * beta).T + 4 * DC_Dt.T @ np.matrix(beta).T)

        # Second derivatives of theta & gamma
        dGt_dg_dp_dt = dp_dt @ dGt_dg

        # Second derivatives of gamma's
        dGt_dgg = get_gamma_2nd_deriv(gamma_parameters, gi)  # ∂²Γ_t / (∂γ_i ∂γ_j) for i,j in gamma_id
        # dGt_dgg_p[i,j] = p_Qt * dGt_dgg[i, j] would be a 3D matrix with shape (r, r, 15)

        # ------ Second derivatives (4+r)x(4+r)x15 of [Γ_t * Ω_t @ p_{D^Q_t}] ------
        second_deriv = np.zeros((4 + r, 4 + r, 15))
        second_deriv[0, 0, :] = (Gamma_t * Omega @ dp_dt1t1).flatten()
        second_deriv[0, 1, :] = (Gamma_t * Omega @ dp_dt1t2).flatten()
        second_deriv[0, 2, :] = (Gamma_t * Omega @ dp_dt1t3).flatten()
        second_deriv[0, 3, :] = (Gamma_t * Omega @ dp_dt1t).flatten()
        second_deriv[1, 1, :] = (Gamma_t * Omega @ dp_dt2t2).flatten()
        second_deriv[1, 2, :] = (Gamma_t * Omega @ dp_dt2t3).flatten()
        second_deriv[1, 3, :] = (Gamma_t * Omega @ dp_dt2t).flatten()
        second_deriv[2, 2, :] = (Gamma_t * Omega @ dp_dt3t3).flatten()
        second_deriv[2, 3, :] = (Gamma_t * Omega @ dp_dt3t).flatten()
        second_deriv[3, 3, :] = (Gamma_t * Omega @ dp_dtt).flatten()
        # Add the derivatives w.r.t. gamma's if r>0
        if r > 0:
            for i in range(r):
                gamma_idx = 4 + i  # This shifts the index to start at 4

                second_deriv[0, gamma_idx, :] = (Omega @ dGt_dg_dp_dt1[:, i]).flatten()
                second_deriv[1, gamma_idx, :] = (Omega @ dGt_dg_dp_dt2[:, i]).flatten()
                second_deriv[2, gamma_idx, :] = (Omega @ dGt_dg_dp_dt3[:, i]).flatten()
                second_deriv[3, gamma_idx, :] = (Omega @ dGt_dg_dp_dt[:, i]).flatten()
                second_deriv[gamma_idx, gamma_idx, :] = (Omega @ p_Qt * dGt_dgg[i, i]).flatten()
                # Cross-derivatives between different gammas
                for j in range(i + 1, r):
                    gamma_jdx = 4 + j
                    second_deriv[gamma_idx, gamma_jdx, :] = (Omega @ p_Qt * dGt_dgg[i, j]).flatten()
        # Mirror the upper triangle to the lower triangle across all 15 slices
        for i in range(4 + r):
            for j in range(i + 1, 4 + r):
                second_deriv[j, i, :] = second_deriv[i, j, :]
        # ---------------------------------------------------------------------------

        # ---- Output ----
        # first_deriv: first order derivatives 15x(4+r) of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is asymmetric
        # second_deriv: second order derivatives (4+r)x(4+r)x15 of [Γ_t * Ω_t @ p_{D^Q_t}] when Qt is asymmetric
        # ----------------
        return first_deriv, second_deriv

    # ---- Step 2: Get the first and second derivatives of p_Q ----
    # Get parameters, gamma_weights and true site pattern probs
    num_param = len(parameters)
    h = gamma_id[0].size            # get the number of hybridizations
    tree_parameters = parameters[:-h]
    J = len(tree_parameters) - 1    # number of tau's
    gamma_parameters = parameters[-h:]
    Q_param_idx = set()

    # Create zero matrices for summation.
    firstDer_p_Q = np.zeros((15, num_param))
    secondDer_p_Q = np.zeros((num_param, num_param, 15))

    # Remove common gamma indices in gamma_id for the ease of taking derivative of Gamma_t
    gamma_id_clean = remove_common_elements(gamma_id)

    # Take derivatives of quartet (pi = param_idx, ia = is_asymm, tp = taxa_perm, gi = gamma_id)
    for pi, ia, tp, gi in zip(param_idx, is_asymm, taxa_perm, gamma_id_clean):
        # Get the first and second derivatives of [Γ_t * Ω_t @ p_{D^Q_t}] by Symm or Asymm
        firstDer_Qt, secondDer_Qt = (Qt_1st_2nd_deriv_Asymm if ia
                                     else Qt_1st_2nd_deriv_Symm)(tree_parameters, gamma_parameters, pi, tp, gi)

        # Get all indices of the tau's, theta, gamma's of this D^Q_t quartet
        # pi reversed because tau_id in Qt_1st_2nd_deriv*() is [t1,t2,t3] but pi is [t3,t2,t1]
        tau_idx = J - pi[::-1]      # Convert to 0-based indices for tau.
        theta_idx = [J]             # Convert to 0-based indices for theta
        if gi.size > 0:
            gamma_idx = J + np.abs(gi)     # Convert to 0-based indices for gamma
            Qt_param_idx = np.concatenate((tau_idx, theta_idx, gamma_idx))
        else:
            Qt_param_idx = np.concatenate((tau_idx, theta_idx))
        Q_param_idx.update(Qt_param_idx)

        # Accumulate to firstDer_p_Q by Qt_param_idx
        firstDer_p_Q[:, Qt_param_idx] += firstDer_Qt

        # Accumulate to secondDer_p_Q by Qt_param_idx
        row_idx, col_idx = np.ix_(Qt_param_idx, Qt_param_idx)
        secondDer_p_Q[row_idx, col_idx, :] += secondDer_Qt

    # ---- Step 3: get the gradient and Hessian of this quartet subnetwork ----
    p_Q = getTrueProbsQuartet(parameters, param_idx, is_asymm, taxa_perm, gamma_id)
    # R_Q matrix = gradient vector of log(p_Q)
    R_Q_mat = firstDer_p_Q.T / p_Q

    n_Q_div_p2_Q = n_Q / np.square(p_Q)     # shape (15,)
    n_Q_div_p_Q = n_Q / p_Q                 # shape (15,)
    # H_Q matrix = Hessian matrix of l(Q)
    H_Q_mat = (n_Q_div_p2_Q * firstDer_p_Q.T) @ firstDer_p_Q - secondDer_p_Q @ n_Q_div_p_Q
    return R_Q_mat, H_Q_mat


def get_Vari_Sens_Mat(parameters, zipped_data_net, all_E_mat, n_D):
    """Get the composite score function (U_c), the variability matrix (J_mat) and
    the sensitivity matrix (H_mat) of the composite likelihood of network."""

    # -------------------------------------------------------------------------
    # 1. Precompute parameter size and create zero matrices for summation
    # -------------------------------------------------------------------------
    num_param = len(parameters)        # Number of parameters

    # sum_RE is sum of R_Q_mat @ E_Q_mat over all quartet Q where R_Q_mat is deriv of log(p) for quartet Q,
    # E_Q_mat is a map matrix that n_Q = E_Q_mat @ n_D. H_mat is the sensitivity matrix
    sum_RE = np.matrix(np.zeros((num_param, len(n_D))))
    H_mat = np.matrix(np.zeros((num_param, num_param)))

    # -------------------------------------------------------------------------
    # 2. Compute variability matrix J_mat and sensitivity matrix H_mat
    # -------------------------------------------------------------------------
    for (n_Q, param_idx, is_asymm, taxa_perm, gamma_id), E_Q_mat in zip(zipped_data_net, all_E_mat):
        # Get R_Q_mat, H_Q_mat from quartet Q
        R_Q_mat, H_Q_mat = get_Grad_Hess_Quartet(parameters, n_Q, param_idx, is_asymm, taxa_perm, gamma_id)

        # Accumulate R_Q_mat @ E_Q_mat of each quartet Q to sum_RE, and accumulate H_Q_mat to H_mat
        sum_RE += R_Q_mat @ E_Q_mat
        H_mat += H_Q_mat

    # Get the empirical variance to calculate the variability matrix
    empirical_var = np.diag(n_D) - np.outer(n_D,n_D) / n_D.sum()
    J_mat = sum_RE @ empirical_var @ sum_RE.T
    # U_c = sum_RE @ n_D

    return J_mat, H_mat


def get_Score_Vari_Sens_Mat(parameters, zipped_data_net, all_E_mat, n_D):
    """Get the composite score function (U_c), the variability matrix (J_mat) and
    the sensitivity matrix (H_mat) of the composite likelihood of network."""

    # -------------------------------------------------------------------------
    # 1. Precompute parameter size and create zero matrices for summation
    # -------------------------------------------------------------------------
    num_param = len(parameters)        # Number of parameters

    # sum_RE is sum of R_Q_mat @ E_Q_mat over all quartet Q where R_Q_mat is deriv of log(p) for quartet Q,
    # E_Q_mat is a map matrix that n_Q = E_Q_mat @ n_D. H_mat is the sensitivity matrix
    sum_RE = np.matrix(np.zeros((num_param, len(n_D))))
    H_mat = np.matrix(np.zeros((num_param, num_param)))

    # -------------------------------------------------------------------------
    # 2. Compute variability matrix J_mat and sensitivity matrix H_mat
    # -------------------------------------------------------------------------
    for (n_Q, param_idx, is_asymm, taxa_perm, gamma_id), E_Q_mat in zip(zipped_data_net, all_E_mat):
        # Get R_Q_mat, H_Q_mat from quartet Q
        R_Q_mat, H_Q_mat = get_Grad_Hess_Quartet(parameters, n_Q, param_idx, is_asymm, taxa_perm, gamma_id)

        # Accumulate R_Q_mat @ E_Q_mat of each quartet Q to sum_RE, and accumulate H_Q_mat to H_mat
        sum_RE += R_Q_mat @ E_Q_mat
        H_mat += H_Q_mat

    # Get the empirical variance to calculate the variability matrix
    empirical_var = np.diag(n_D) - np.outer(n_D,n_D) / n_D.sum()
    J_mat = sum_RE @ empirical_var @ sum_RE.T
    U_c = (sum_RE @ n_D).T

    return U_c, J_mat, H_mat


def get_curvAdjust_matrix(MCLE, zipped_data_net, all_E_mat, n_D):
    """Use J_mat and H_mat derived from get_Vari_Sens_Mat() to get curvature adjustment matrix.
    Use zipped_data_net from get_zippedData_net()."""
    import numpy as np

    J_mat, H_mat = get_Vari_Sens_Mat(MCLE, zipped_data_net, all_E_mat, n_D)

    def compute_M_inv(H_mat):
        U, D, _ = np.linalg.svd(H_mat)
        return U @ np.diag(D ** (-1/2)) @ U.T

    def compute_M_A(H_mat, J_mat):
        Godambe_mat = H_mat @ np.linalg.inv(J_mat) @ H_mat
        U, D, _ = np.linalg.svd(Godambe_mat)
        return U @ np.diag(np.sqrt(D)) @ U.T

    M_inv = compute_M_inv(H_mat)
    M_A = compute_M_A(H_mat, J_mat)
    curv_adj_mat = M_inv @ M_A

    return curv_adj_mat


#####################################################
## Code Metropolis-within-Gibbs sampling algorithm ##
#####################################################

def log_invgamma(x, alpha, beta):
    return -(alpha + 1) * np.log(x) - beta / x

def log_beta(x, alpha, beta):
    return np.sum((alpha - 1) * np.log(x) * (beta - 1) * np.log(1 - x))

def get_tau_prior_and_constraint(phylox_network):
    """Given a self topology, automatically generate a joint prior function for tau: "log_prior_tau",
    boundaries of tau for proposal kernal: "get_tau_boundaries", and a boolean function of whether tau is in
    its constraint so that we reject bad tau's after curvature adjustment: "tau_in_constraints". """

    # Label the speciation times in pre-order traversal
    clone_network = phylox_network.copy() # Copy the self to avoid modifying the original
    label_speciation_time_idx(clone_network)

    # Sorts child nodes in descending order in terms of the number of children each child node has.
    # This is for the convenience of counting the number of continuous nodes in an asymmetric pattern
    clone_network = ladderize(clone_network, ascending=False)

    # Preparation to set up the tau prior
    h = len(phylox_network.reticulations)   # number of hybrids
    J = len(phylox_network.leaves) + h - 1  # total number of tau parameters
    tauPrior_idx = []
    tauPrior_pwr = []
    asymm_pattern_count = 0
    leaves = clone_network.leaves   # Cache the leaves
    preorder_internal_node = [n for n in clone_network.nodes() if n not in clone_network.leaves] # Cache the internal nodes

    # Preorder traversing the internal nodes of a sorted clone_network
    for node in preorder_internal_node:
        # If an internal node has both child nodes being leaves, then we skip it.
        if sum(child in leaves for child in clone_network.successors(node)) == 2:
            if asymm_pattern_count > 0:  # store the asymm_pattern_count in tauPrior_pwr and reset asymm_pattern_count
                tauPrior_pwr.append(-asymm_pattern_count)
                asymm_pattern_count = 0
            continue

        # True/False: one of the two child nodes is leaf_node.
        any_leaf_node = any(child in leaves for child in clone_network.successors(node))

        # If this node has at least one child as leaf_node, then it is a start of asymmetric pattern.
        if any_leaf_node:
            if asymm_pattern_count == 0:  # Only save the most ancestral node_idx of an asymmetric pattern
                tauPrior_idx.append(get_label_from_node(clone_network, node))
            asymm_pattern_count += 1

        # If this node has no child as leaf_node, then this node together with its two child nodes are symmetric pattern.
        else:
            if asymm_pattern_count > 0:  # store the asymm_pattern_count in tauPrior_pwr and reset asymm_pattern_count
                tauPrior_pwr.append(-asymm_pattern_count)
                asymm_pattern_count = 0
            # Save the most ancestral node_idx of a symmetric pattern and its power is -2
            tauPrior_idx.append(get_label_from_node(clone_network, node))
            tauPrior_pwr.append(-2)

    # Store the asymm_pattern_count in tauPrior_pwr if it ends the internal nodes iterator
    if asymm_pattern_count > 0:
        tauPrior_pwr.append(-asymm_pattern_count)

    # Get the constraints of tau as a list of pairs
    constraint = [[get_label_from_node(clone_network, list(clone_network.predecessors(node))[0]), # parent node
                   get_label_from_node(clone_network, node)]                                      # child node
                  for node in [n for n in clone_network.nodes() if n not in clone_network.leaves]
                  if next(clone_network.predecessors(node), None) is not None]

    # Adjust for the indexing of tau = [tau_J,...,tau_1]
    tauPrior_idx_adj = J - np.array(tauPrior_idx)
    constraint_adj = J - np.array(constraint)
    # Extract the 'i' and 'j' columns as separate 1D arrays
    i_idx = constraint_adj[:, 0]
    j_idx = constraint_adj[:, 1]

    # Get boolean function of whether tau is in constraint
    def tau_in_constraints(tau):
        # return all([tau[i] > tau[j] for i, j in constraint_adj])
        return np.all(tau[i_idx] > tau[j_idx])

    # Get joint prior for tau
    def log_prior_tau(tau, invgamma_alpha=None, invgamma_beta=None):
        # If user do not specify alpha, the prior is flat by default.
        if invgamma_alpha == None:
            invgamma_alpha = 3

        # If user do not specify beta, the prior for root age is conservative by default.
        if invgamma_beta == None:
            invgamma_beta = 0.002

        # Check if all tau's satisfy the constraint
        if not tau_in_constraints(tau):
            return -np.inf

        # The prior of root age follows inverse gamma. The prior of other speciation times follow kernel of uniform
        log_prior = log_invgamma(tau[0], invgamma_alpha, invgamma_beta) + \
                    np.sum(tauPrior_pwr * np.log(tau[tauPrior_idx_adj]))

        return log_prior

    # Get boundaries of each tau
    def get_tau_boundaries(current_tau):
        lwr_b = np.zeros(J)          # Default lower bound is 0
        upr_b = np.full(J, np.inf)   # Default upper bound is inf

        # Vectorized in-place updates.
        np.maximum.at(lwr_b, i_idx, current_tau[j_idx])
        np.minimum.at(upr_b, j_idx, current_tau[i_idx])

        return np.column_stack((lwr_b, upr_b))

    return log_prior_tau, get_tau_boundaries, tau_in_constraints

def proposal_kernel(current_values, step_width, boundaries, type=None):
    """Reference (Yang 2014, page 222~225)"""
    is_scalar = np.isscalar(current_values) or np.ndim(current_values) == 0
    # Ensure inputs are standard numpy arrays for the math
    current_values = np.atleast_1d(current_values)
    boundaries = np.atleast_2d(boundaries)
    sample_size = current_values.size
    w = step_width

    # Check sizes of user input
    if sample_size != boundaries.shape[0]:
        raise ValueError("Size of current_value does not match rows of boundaries")

    # Propose values
    if type == "normal" or type is None:  # Default proposal kernal is normal
        proposal_val = current_values + np.random.normal(0, w, size=sample_size)
    elif type == "uniform":
        proposal_val = current_values+ np.random.uniform(-w / 2, w / 2, size=sample_size)
    elif type == "Bactrian":
        m = 0.95
        bactrian_noise = (2 * np.random.binomial(1, 0.5, size=sample_size) - 1) * m + \
                          np.random.normal(size=sample_size) * np.sqrt(1 - m ** 2)
        proposal_val = current_values + bactrian_noise * w
    else:
        raise ValueError("Invalid distribution type. Choose 'uniform', 'normal', or 'Bactrian'.")

    # ------ Vectorized Boundary Reflection ------
    # Extract lower and upper bounds as 1D arrays
    lwr_b = boundaries[:, 0]
    upr_b = boundaries[:, 1]

    # As long as ANY value is outside its bounds, apply reflections
    while np.any((proposal_val < lwr_b) | (proposal_val > upr_b)):
        # "under" finds proposal_val that are below the lower bounds "a" and bounce it up
        under = proposal_val < lwr_b
        proposal_val[under] = 2 * lwr_b[under] - proposal_val[under]

        # Find everything above the upper bounds and bounce it down
        over = proposal_val > upr_b
        proposal_val[over] = 2 * upr_b[over] - proposal_val[over]

    if is_scalar:
        return float(proposal_val[0])

    return proposal_val


def MCMC_rawCompLik(zipped_data_net, zipped_data_net_reduce, phylox_network,     # species network info
                        nsample, thin, step_width, prop_kern=None,               # MCMC settings
                        thetaPr=None, tauPr=None, gammaPr=None, MCLE=None,       # User costomized prior and MCLE
                        prog_bar = None):                                       # show progress bar: yes/no
    """We use zipped_data_net from get_zippedData_net() to get MCLE and use zipped_data_net_reduced from
    get_zippedData_net_reduced() to compute likelihood for a faster computation during MCMC runs."""
    from tqdm import trange     # Included to show progress bar
    import numpy as np

    h = len(phylox_network.reticulations)   # number of hybrids
    J = len(phylox_network.leaves) + h - 1  # total number of tau parameters

    if MCLE is None:
        # Get MCLE of network parameters if user does not provide one
        MCLE, _ = get_MCLE_parameters(zipped_data_net, phylox_network)

    # initialize prior (inverse gamma distr) for tau
    log_prior_tau, get_tau_boundaries, tau_in_constraints = get_tau_prior_and_constraint(phylox_network)
    if thetaPr is None:
        a_theta = 3                         # default alpha parameter for theta
        b_theta = MCLE[J] * (a_theta - 1)   # default beta parameter for theta
    else:
        a_theta = thetaPr[0]  # user input alpha parameter for theta
        b_theta = thetaPr[1]  # user input beta parameter for theta

    # initialize prior (inverse gamma distr) for theta
    if tauPr is None:
        a_tau = 3                       # default alpha parameter for tau_root
        b_tau = MCLE[0] * (a_tau - 1)   # default beta parameter for tau_root
    else:
        a_tau = tauPr[0]  # user input alpha parameter for tau_root
        b_tau = tauPr[1]  # user input beta parameter for tau_root

    # initialize prior (beta distr) for gamma
    gamma_bound = [[0,1]] * h
    if gammaPr is None:
        a_gamma = 1  # default alpha parameter for gamma
        b_gamma = 1  # default beta parameter for gamma
    else:
        a_gamma = gammaPr[0]  # user input alpha parameter for gamma
        b_gamma = gammaPr[1]  # user input beta parameter for gamma

    # initialize acceptance counter
    accept_count_tau = 0
    accept_count_theta = 0
    accept_count_gamma = 0

    # initialize parameter chain
    MCMC_samples = []
    curr_param = MCLE.copy()
    # Calculate the starting states ONCE before the loop begins. Update them only when proposal accepted.
    curr_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, curr_param)

    # Progress bar setup
    total_iter = int(nsample * thin)
    if prog_bar is None or prog_bar:
        iter_range = trange(total_iter, desc="MCMC raw CL")
    else:
        iter_range = range(total_iter)

    for iteration in iter_range:
        # Step 1: Sampling theta
        new_param = curr_param.copy() # RESET new_param to current accepted state
        # curr_loglik have been calculated before Step 1
        curr_logprior = log_invgamma(curr_param[J], a_theta,b_theta)
        new_param[J] = proposal_kernel(curr_param[J], step_width[1], (5e-5, 0.2), prop_kern)
        new_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, new_param)
        new_logprior = log_invgamma(new_param[J], a_theta,b_theta)
        # accept or reject new proposal
        if np.log(np.random.rand()) < new_logprior+new_loglik - curr_logprior-curr_loglik:
            curr_param[J] = new_param[J] # accept new proposal and update it
            curr_loglik = new_loglik     # Carry accepted loglik forward
            accept_count_theta += 1
        # else reject the new_theta proposal

        # Step 2: Sampling tau
        new_param = curr_param.copy() # RESET new_param to current accepted state
        # curr_loglik have been updated before Step 2
        curr_logprior = log_prior_tau(curr_param[:J], a_tau,b_tau)
        new_param[:J] = proposal_kernel(curr_param[:J], step_width[0], get_tau_boundaries(curr_param[:J]), prop_kern)
        new_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, new_param)
        new_logprior = log_prior_tau(new_param[:J], a_tau,b_tau)
        # accept or reject new proposal
        if np.log(np.random.rand()) < new_logprior + new_loglik - curr_logprior - curr_loglik:
            curr_param[:J] = new_param[:J]  # accept new proposal and update it
            curr_loglik = new_loglik        # Carry accepted loglik forward
            accept_count_tau += 1
        # else reject the new_tau proposal

        # Step 3: Sampling gamma
        new_param = curr_param.copy() # RESET new_param to current accepted state
        # curr_loglik have been updated before Step 3
        curr_logprior = log_beta(curr_param[-h:], a_gamma, b_gamma)
        new_param[-h:] = proposal_kernel(curr_param[-h:], step_width[2], gamma_bound, prop_kern)
        new_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, new_param)
        new_logprior = log_beta(new_param[-h:], a_gamma, b_gamma)
        # accept or reject new proposal
        if np.log(np.random.rand()) < new_logprior + new_loglik - curr_logprior - curr_loglik:
            curr_param[-h:] = new_param[-h:]  # accept new proposal and update it
            curr_loglik = new_loglik        # Carry accepted loglik forward
            accept_count_gamma += 1
        # else reject the new_gamma proposal

        # Step 4: store samples every 'thin'-th iteration
        if iteration % thin == 0:
            MCMC_samples.append(curr_param.copy())
            # Step 5: Update progress bar postfix every 'thin'-th iteration
            if prog_bar is None or prog_bar:
                accept_ratio_tau = accept_count_tau / (iteration + 1)
                accept_ratio_theta = accept_count_theta / (iteration + 1)
                accept_ratio_gamma = accept_count_gamma / (iteration + 1)
                iter_range.set_postfix(tau_Pjump=f"{accept_ratio_tau:.2f}", theta_Pjump=f"{accept_ratio_theta:.2f}",
                                     gamma_Pjump=f"{accept_ratio_gamma:.2f}")

    # Calculate the acceptance ratio
    accept_ratio_tau = accept_count_tau / total_iter
    accept_ratio_theta = accept_count_theta / total_iter
    accept_ratio_gamma = accept_count_gamma / total_iter
    print(f"rawCL, P_jump of tau: {accept_ratio_tau:.3f}, P_jump of theta: {accept_ratio_theta:.3f}, P_jump of gamma: {accept_ratio_gamma:.3f}")

    return np.array(MCMC_samples)


def MCMC_curvAdjCompLik(zipped_data_net, zipped_data_net_reduce, phylox_network,    # species network info
                        nsample, thin, step_width, curvAdj, prop_kern=None,         # MCMC settings
                        thetaPr=None, tauPr=None, gammaPr=None, MCLE=None,          # User costomized prior and MCLE
                        prog_bar = None):                                       # show progress bar: yes/no
    """We use zipped_data_net from get_zippedData_net() to get MCLE and use zipped_data_net_reduced from
    get_zippedData_net_reduced() to compute likelihood for a faster computation during MCMC runs."""
    from tqdm import trange     # Included to show progress bar
    import numpy as np

    h = len(phylox_network.reticulations)   # number of hybrids
    J = len(phylox_network.leaves) + h - 1  # total number of tau parameters

    if MCLE is None:
        # Get MCLE of network parameters if user does not provide one
        MCLE, _ = get_MCLE_parameters(zipped_data_net, phylox_network)

    # initialize prior (inverse gamma distr) for tau
    log_prior_tau, get_tau_boundaries, tau_in_constraints = get_tau_prior_and_constraint(phylox_network)
    if thetaPr is None:
        a_theta = 3                         # default alpha parameter for theta
        b_theta = MCLE[J] * (a_theta - 1)   # default beta parameter for theta
    else:
        a_theta = thetaPr[0]  # user input alpha parameter for theta
        b_theta = thetaPr[1]  # user input beta parameter for theta

    # initialize prior (inverse gamma distr) for theta
    if tauPr is None:
        a_tau = 3                       # default alpha parameter for tau_root
        b_tau = MCLE[0] * (a_tau - 1)   # default beta parameter for tau_root
    else:
        a_tau = tauPr[0]  # user input alpha parameter for tau_root
        b_tau = tauPr[1]  # user input beta parameter for tau_root

    # initialize prior (beta distr) for gamma
    gamma_bound = [[0,1]] * h
    if gammaPr is None:
        a_gamma = 1  # default alpha parameter for gamma
        b_gamma = 1  # default beta parameter for gamma
    else:
        a_gamma = gammaPr[0]  # user input alpha parameter for gamma
        b_gamma = gammaPr[1]  # user input beta parameter for gamma

    # initialize acceptance counter
    accept_count_tau = 0
    accept_count_theta = 0
    accept_count_gamma = 0

    # initialize parameter chain
    MCMC_samples = []
    curr_param = MCLE.copy()
    C = np.asarray(curvAdj)
    # Calculate the starting states ONCE before the loop begins. Update them only when proposal accepted.
    curr_star = MCLE + C @ (curr_param - MCLE)
    curr_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, curr_star)

    # Progress bar setup
    total_iter = int(nsample * thin)
    if prog_bar is None or prog_bar:
        iter_range = trange(total_iter, desc="MCMC curvAdjust_matrix CL")
    else:
        iter_range = range(total_iter)

    for iteration in iter_range:
        # Step 1: Sampling theta
        new_param = curr_param.copy() # RESET new_param to current accepted state
        # curr_star & curr_loglik have been calculated before Step 1
        curr_logprior = log_invgamma(curr_param[J], a_theta,b_theta)
        while True:
            new_param[J] = proposal_kernel(curr_param[J], step_width[1], (5e-5,0.2), prop_kern)
            new_star = MCLE + C @ (new_param - MCLE)
            if tau_in_constraints(new_star[:J]) and new_star[J]>0:
                break
        new_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, new_star)
        new_logprior = log_invgamma(new_param[J], a_theta,b_theta)
        # accept or reject new proposal
        if np.log(np.random.rand()) < new_logprior+new_loglik - curr_logprior-curr_loglik:
            curr_param[J] = new_param[J] # accept new proposal and update it
            curr_loglik = new_loglik     # Carry accepted loglik forward
            accept_count_theta += 1
        # else reject the new_theta proposal

        # Step 2: Sampling tau
        new_param = curr_param.copy() # RESET new_param to current accepted state
        # curr_loglik have been updated before Step 2
        curr_logprior = log_prior_tau(curr_param[:J], a_tau,b_tau)
        while True:
            new_param[:J] = proposal_kernel(curr_param[:J], step_width[0], get_tau_boundaries(curr_param[:J]), prop_kern)
            new_star = MCLE + C @ (new_param - MCLE)
            if tau_in_constraints(new_star[:J]) and new_star[J]>0:
                break
        new_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, new_star)
        new_logprior = log_prior_tau(new_param[:J], a_tau,b_tau)
        # accept or reject new proposal
        if np.log(np.random.rand()) < new_logprior + new_loglik - curr_logprior - curr_loglik:
            curr_param[:J] = new_param[:J]  # accept new proposal and update it
            curr_loglik = new_loglik        # Carry accepted loglik forward
            accept_count_tau += 1
        # else reject the new_tau proposal

        # Step 3: Sampling gamma
        new_param = curr_param.copy() # RESET new_param to current accepted state
        # curr_loglik have been updated before Step 3
        curr_logprior = log_beta(curr_param[-h:], a_gamma, b_gamma)
        while True:
            new_param[-h:] = proposal_kernel(curr_param[-h:], step_width[2], gamma_bound, prop_kern)
            new_star = MCLE + C @ (new_param - MCLE)
            if tau_in_constraints(new_star[:J]) and new_star[J] > 0:
                break
        new_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, new_star)
        new_logprior = log_beta(new_param[-h:], a_gamma, b_gamma)
        # accept or reject new proposal
        if np.log(np.random.rand()) < new_logprior + new_loglik - curr_logprior - curr_loglik:
            curr_param[-h:] = new_param[-h:]  # accept new proposal and update it
            curr_loglik = new_loglik        # Carry accepted loglik forward
            accept_count_gamma += 1
        # else reject the new_gamma proposal

        # Step 4: store samples every 'thin'-th iteration
        if iteration % thin == 0:
            MCMC_samples.append(curr_param.copy())
            # Step 5: Update progress bar postfix every 'thin'-th iteration
            if prog_bar is None or prog_bar:
                accept_ratio_tau = accept_count_tau / (iteration + 1)
                accept_ratio_theta = accept_count_theta / (iteration + 1)
                accept_ratio_gamma = accept_count_gamma / (iteration + 1)
                iter_range.set_postfix(tau_Pjump=f"{accept_ratio_tau:.2f}", theta_Pjump=f"{accept_ratio_theta:.2f}",
                                     gamma_Pjump=f"{accept_ratio_gamma:.2f}")

    # Calculate the acceptance ratio
    accept_ratio_tau = accept_count_tau / total_iter
    accept_ratio_theta = accept_count_theta / total_iter
    accept_ratio_gamma = accept_count_gamma / total_iter
    print(f"adjCL, P_jump of tau: {accept_ratio_tau:.3f}, P_jump of theta: {accept_ratio_theta:.3f}, P_jump of gamma: {accept_ratio_gamma:.3f}")

    return np.array(MCMC_samples)


# def MCMC_curvAdjCompLik(zipped_data_net, zipped_data_net_reduce, self,  # species network info
#                         nsample, thin, step_width, curvAdjust_matrix, prop_kern=None,  # MCMC settings
#                         thetaPr=None, tauPr=None, gammaPr=None, MCLE=None,  # User costomized prior and MCLE
#                         prog_bar=None):  # show progress bar: yes/no
#     """We use zipped_data_net from get_zippedData_net() to get MCLE and use zipped_data_net_reduced from
#     get_zippedData_net_reduced() to compute likelihood for a faster computation during MCMC runs."""
#     from tqdm import trange  # Included to show progress bar
#     import numpy as np
#
#     h = len(self.reticulations)  # number of hybrids
#     J = len(self.leaves) + h - 1  # total number of tau parameters
#
#     if MCLE is None:
#         MCLE, _ = get_MCLE_parameters(zipped_data_net, self)
#
#     log_prior_tau, get_tau_boundaries, tau_in_constraints = get_tau_prior_and_constraint(self)
#
#     if thetaPr is None:
#         a_theta = 3
#         b_theta = MCLE[J] * (a_theta - 1)
#     else:
#         a_theta = thetaPr[0]
#         b_theta = thetaPr[1]
#
#     if tauPr is None:
#         a_tau = 3
#         b_tau = MCLE[0] * (a_tau - 1)
#     else:
#         a_tau = tauPr[0]
#         b_tau = tauPr[1]
#
#     gamma_bound = [[0, 1]] * h
#     if gammaPr is None:
#         a_gamma = 1
#         b_gamma = 1
#     else:
#         a_gamma = gammaPr[0]
#         b_gamma = gammaPr[1]
#
#     # 1. CHANGE: Initialize tau acceptance count as an array of size J
#     accept_count_tau = np.zeros(J)
#     accept_count_theta = 0
#     accept_count_gamma = 0
#
#     MCMC_samples = []
#     curr_param = MCLE.copy()
#     C = np.asarray(curvAdjust_matrix)
#
#     curr_star = MCLE + C @ (curr_param - MCLE)
#     curr_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, curr_star)
#
#     total_iter = int(nsample * thin)
#     if prog_bar is None or prog_bar:
#         iter_range = trange(total_iter, desc="MCMC curvAdjust_matrix CL")
#     else:
#         iter_range = range(total_iter)
#
#     for iteration in iter_range:
#         # ---------------------------------------------------------
#         # Step 1: Sampling theta
#         # ---------------------------------------------------------
#         new_param = curr_param.copy()
#         curr_logprior = log_invgamma(curr_param[J], a_theta, b_theta)
#         while True:
#             new_param[J] = proposal_kernel(curr_param[J], step_width[1], (5e-5, 0.2), prop_kern)
#             new_star = MCLE + C @ (new_param - MCLE)
#             if tau_in_constraints(new_star[:J]) and new_star[J] > 0:
#                 break
#         new_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, new_star)
#         new_logprior = log_invgamma(new_param[J], a_theta, b_theta)
#
#         if np.log(np.random.rand()) < new_logprior + new_loglik - curr_logprior - curr_loglik:
#             curr_param[J] = new_param[J]
#             curr_loglik = new_loglik
#             accept_count_theta += 1
#
#         # ---------------------------------------------------------
#         # Step 2: Sampling tau (ELEMENT-WISE)
#         # ---------------------------------------------------------
#         for i in range(J):
#             new_param = curr_param.copy()
#             curr_logprior = log_prior_tau(curr_param[:J], a_tau, b_tau)
#
#             while True:
#                 boundaries = get_tau_boundaries(curr_param[:J])
#                 step_i = step_width[0][i] if isinstance(step_width[0], (list, np.ndarray)) else step_width[0]
#
#                 new_param[i] = proposal_kernel(curr_param[i], step_i, boundaries[i], prop_kern)
#
#                 new_star = MCLE + C @ (new_param - MCLE)
#                 if tau_in_constraints(new_star[:J]) and new_star[J] > 0:
#                     break
#
#             new_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, new_star)
#             new_logprior = log_prior_tau(new_param[:J], a_tau, b_tau)
#
#             if np.log(np.random.rand()) < new_logprior + new_loglik - curr_logprior - curr_loglik:
#                 curr_param[i] = new_param[i]
#                 curr_loglik = new_loglik
#                 # 2. CHANGE: Increment only the specific i-th counter
#                 accept_count_tau[i] += 1
#
#         # ---------------------------------------------------------
#         # Step 3: Sampling gamma
#         # ---------------------------------------------------------
#         new_param = curr_param.copy()
#         curr_logprior = log_beta(curr_param[-h:], a_gamma, b_gamma)
#         while True:
#             new_param[-h:] = proposal_kernel(curr_param[-h:], step_width[2], gamma_bound, prop_kern)
#             new_star = MCLE + C @ (new_param - MCLE)
#             if tau_in_constraints(new_star[:J]) and new_star[J] > 0:
#                 break
#         new_loglik = SpeciesNetwork_CompLogLik(zipped_data_net_reduce, new_star)
#         new_logprior = log_beta(new_param[-h:], a_gamma, b_gamma)
#
#         if np.log(np.random.rand()) < new_logprior + new_loglik - curr_logprior - curr_loglik:
#             curr_param[-h:] = new_param[-h:]
#             curr_loglik = new_loglik
#             accept_count_gamma += 1
#
#         # ---------------------------------------------------------
#         # Step 4: store samples and update progress
#         # ---------------------------------------------------------
#         if iteration % thin == 0:
#             MCMC_samples.append(curr_param.copy())
#             if prog_bar is None or prog_bar:
#                 # 3. CHANGE: Denominator is back to (iteration + 1). Format array as string.
#                 accept_ratio_tau = accept_count_tau / (iteration + 1)
#                 tau_str = np.array2string(accept_ratio_tau, precision=2, separator=',', floatmode='fixed')
#
#                 accept_ratio_theta = accept_count_theta / (iteration + 1)
#                 accept_ratio_gamma = accept_count_gamma / (iteration + 1)
#
#                 # tau_str will output an array like: [0.34,0.45,0.20,0.50]
#                 iter_range.set_postfix(tau_Pjump=tau_str,
#                                        theta_Pjump=f"{accept_ratio_theta:.2f}",
#                                        gamma_Pjump=f"{accept_ratio_gamma:.2f}")
#
#     # 4. CHANGE: Final ratio calculations
#     accept_ratio_tau = accept_count_tau / total_iter
#     accept_ratio_theta = accept_count_theta / total_iter
#     accept_ratio_gamma = accept_count_gamma / total_iter
#
#     tau_str_final = np.array2string(accept_ratio_tau, precision=3, separator=',', floatmode='fixed')
#     print(
#         f"adjCL, P_jump of taus: {tau_str_final}, P_jump of theta: {accept_ratio_theta:.3f}, P_jump of gamma: {accept_ratio_gamma:.3f}")
#
#     return np.array(MCMC_samples)


####################################################################
## Compute modified composite likelihood ratio statistics (mCLRT) ##
####################################################################

def modified_CompLik_ratio_stat(zipped_data_net, all_E_mat, n_D, phylox_network, gamma_test=None):
    """Compute the modified composite likelihood ratio statistics (Chen et al. 2018) given user input a boolean
    vector gamma_test:
    gamma_test = [True]         -> H0: γ_1=0 vs Ha: γ_1∈(0,0.5]
    gamma_test = [True, True]   -> H0: γ_1=γ_2=0 vs Ha: at least one γ_i∈(0,0.5] for i=1,2.
    gamma_test = [True, False]  -> H0: γ_1=0 vs Ha: γ_1∈(0,0.5]"""

    from numpy.linalg import inv
    from scipy.optimize import minimize
    # -------------------------------------------------------------------------
    # 1. Precompute matrices needed for later computation
    # -------------------------------------------------------------------------
    # L = sum(n_D)                            # Sample size
    N = len(phylox_network.leaves)         # Number of taxa
    h = len(phylox_network.reticulations)  # Number of hybridizations

    # Get unconstrained MCLE
    MCLE, CL_opt = get_MCLE_parameters(zipped_data_net, phylox_network)
    gamma_hat_c = MCLE[-h:]

    # Get constrained MCLE where some gamma_j are fixed at zero
    if gamma_test is None:
        gamma_test = np.full(h, True)
    is_fixed_param = np.append(np.full(N + h, False), gamma_test)
    MCLE_cons, CL_cons = get_MCLE_parameters(zipped_data_net, phylox_network, is_fixed_param)

    # Get estimates of U_pc, J_p and H_p according to p.8 of Supplementary Material of Chen et al. (2018)
    U_c, J_hat, H_hat = get_Score_Vari_Sens_Mat(MCLE, zipped_data_net, all_E_mat, n_D)
    U_e = U_c[:-h]                  # U_{\eta}
    U_g = U_c[-h:]                  # U_{\gamma}
    H_ee = H_hat[:-h, :-h] #/ L      # H_{\eta\eta}
    H_ge = H_hat[-h:, :-h] #/ L      # H_{\gamma\eta}
    H_gg = H_hat[-h:, -h:] #/ L      # H_{\gamma\gamma}
    K_mat = H_ge @ inv(H_ee)        # H_{\gamma\eta} \times H_{\eta\eta}^{-1}
    J_ee = J_hat[:-h, :-h] #/ L      # J_{\eta\eta}
    J_ge = J_hat[-h:, :-h] #/ L      # J_{\gamma\eta}
    J_gg = J_hat[-h:, -h:] #/ L      # J_{\gamma\gamma}
    U_pc = U_g - K_mat @ U_e
    H_p = H_gg - K_mat @ H_ge.T
    J_p = J_gg + K_mat @ J_ee @ K_mat.T - K_mat @ J_ge.T - (K_mat @ J_ge.T).T
    # # Naive estimators
    # U_pc = U_g
    # H_p = H_gg
    # J_p = J_gg

    # Precompute H_p^{-1} and H_pA
    H_p_inv = inv(H_p)
    H_pA = H_p @ inv(J_p) @ H_p

    # -------------------------------------------------------------------------
    # 2. Functions used to compute modified composite likelihood ratio statistics: T_p(γ), ϕ_p(γ), l_MP(γ)
    # -------------------------------------------------------------------------
    def T_p(gamma):
        return H_p_inv @ U_pc - np.matrix(gamma - gamma_hat_c).T
        # return 1/np.sqrt(L) * H_p_inv @ U_pc - np.sqrt(L) * np.matrix(gamma - gamma_hat_c).T

    def phi_p(params):
        """𝜼 = (τ_1,...,τ_{N+h-1},θ) and γ = (γ_1,...,γ_h) are evaluated jointly in this function so that
        we can jointly optimize 𝜼 and γ in l_MP(𝜼,γ)."""
        gamma = params[-h:]
        numerator = SpeciesNetwork_CompLogLik(zipped_data_net, params) - CL_opt
        denominator = (- T_p(gamma).T @ H_p @ T_p(gamma) + U_pc.T @ H_p_inv @ U_pc)[0,0]
        phi_p = numerator / denominator if numerator != 0 else 1/2
        return phi_p
        # return numerator / (- T_p(gamma).T @ H_p @ T_p(gamma) + 1/L * U_pc.T @ H_p_inv @ U_pc)[0,0]

    def l_MP(params):
        """𝜼 = (τ_1,...,τ_{N+h-1},θ) and γ = (γ_1,...,γ_h) are evaluated jointly in this function so that
        we can jointly optimize l_MP(𝜼,γ) to find hat{γ}_M."""
        gamma = params[-h:]
        return - (T_p(gamma).T @ H_pA @ T_p(gamma))[0,0] * phi_p(params)

    # -------------------------------------------------------------------------
    # 3. Find optimized l_MP(hat{γ}_M)
    # -------------------------------------------------------------------------
    labeled_network = phylox_network.copy()
    label_speciation_time_idx(labeled_network)

    def neg_l_MP(trans_param):
        if not np.all(np.isfinite(trans_param)):
            return np.inf

        params = parameter_backtransform(trans_param, labeled_network)

        if not np.all(np.isfinite(params)):
            return np.inf
        if np.any(params < 0):
            return np.inf

        try:
            return -l_MP(params)
        except FloatingPointError:
            print("Floating point error at x =", trans_param)
            raise

    x0 = parameter_transform(np.mean([MCLE,MCLE_cons], axis=0), labeled_network)
    result = minimize(neg_l_MP, x0, method="BFGS",
                      options=dict(gtol=1e-12, maxiter=10000))
    l_MP_opt = -result.fun
    MCLE_M = parameter_backtransform(result.x, labeled_network)

    # -------------------------------------------------------------------------
    # 4. Calculate modified profile composite likelihood ratio statistics and the degree of freedom for 𝜒^2
    # -------------------------------------------------------------------------
    mpCLRT = -2 * (l_MP(MCLE_cons) - l_MP_opt)

    # mpCLRT follows chi-square with degrees of freedom = V
    threshold = 1e-6
    V = np.sum((threshold < MCLE_M[-h:]) & (MCLE_M[-h:] < (0.5 - threshold)))

    # return the modified profile composite likelihood ratio statistics and the degree of freedom
    return mpCLRT, V


#######################################################
## Code automated network data simulation using PAUP ##
#######################################################

def read_chopped_data(list_file_path, schema):
    import dendropy

    # Shared taxon namespace ensures taxa match across loci
    taxa = dendropy.TaxonNamespace()

    # Use the generic CharacterMatrix so DendroPy respects the file's internal datatype
    all_data_matrix = [
        dendropy.NucleotideCharacterMatrix.get(
            path=fp,
            schema=schema,
            taxon_namespace=taxa
        )
        for fp in list_file_path
    ]

    # Efficient concatenation using that specific class's method
    merged_data = dendropy.NucleotideCharacterMatrix.concatenate(all_data_matrix)

    return merged_data


def merge_data(list_file_path, outfile_path, schema):

    merged_data = read_chopped_data(list_file_path, schema)

    # Write output
    merged_data.write(
        path=outfile_path,
        schema=schema
    )

    print(f"Concatenated DNA alignment matrix are written to: {outfile_path}")


def simdata_nex_file(taxa_labels, trees, theta, total_loci, gamma_weights, indPerSpecies, outfile):
    """
    Generate a PAUP* NEXUS script to simulate multilocus DNA data under a network model using two trees.
    """
    # Fixed constants
    Ne = 100000
    mu = theta / (2 * Ne)

    ntax = len(taxa_labels)
    taxa_str = " ".join(taxa_labels)

    with open(outfile, "w") as f:
        f.write("#NEXUS\n\n")

        # Taxa block
        f.write("begin taxa;\n"
                f"\tdimensions ntax={ntax};\n"
                f"\ttaxlabels {taxa_str};\n"
                "end;\n\n")

        # Evaluate this once outside the loop for speed
        is_single_tree = len(gamma_weights) == 1
        allocated_loci = 0

        for i, gamma in enumerate(gamma_weights):
            tree_idx = i + 1

            # Safely calculate loci to prevent losing loci to float truncation
            if tree_idx == len(gamma_weights):
                loci = total_loci - allocated_loci  # Last tree gets whatever is left
            else:
                loci = int(total_loci * gamma)
                allocated_loci += loci

            # Set outfile name
            outfile = "network_data.nex" if is_single_tree else f"tree_data{tree_idx}.nex"

            # Write the block
            f.write(
                f"[ Simulate {gamma:.0%} of data for species tree {tree_idx} ]\n"
                "begin trees;\n"
                f"\ttree 1 = [&R] {trees[i]};\n"
                "end;\n\n"
                "begin dnasim;\n"
                f"\tsimdata multilocus=y nloci={loci} nsitesperlocus=1 indPerSpecies={indPerSpecies};\n"
                "\ttruetree source=memory treenum=1 units=2Ngen\n"
                "\t         scalebrlen=1\n"
                f"\t         mscoal=y Ne={Ne} mu={mu:.4e} seed=0\n"
                "\t         showtruetree=brlens showgenetrees=n storetruetrees=n;\n"
                "\tlset nst=1 basefreq=(0.25 0.25 0.25 0.25);\n"
                "\tbeginsim nreps=1 seed=0;\n"
                f"\t\texport file={outfile} format=nexus charsperline=all;\n"
                "\tendsim;\n"
                "end;\n\n"
            )


def parse_config_file(config_path):
    import re
    config = {}
    with open(config_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip blank lines and lines that START with a comment
            if line and not line.startswith("#"):
                # Only split if '#' is preceded by whitespace (\s+). This protects hybrid tags like '#H1' from being deleted
                line = re.split(r'\s+#', line, maxsplit=1)[0].strip()
                if "=" in line:
                    key, value = [part.strip() for part in line.split("=", 1)]
                    config[key] = value

    # Convert values to appropriate types
    config['seed'] = int(config['seed'])
    config['nsample'] = int(config['nsample'])
    config['thin'] = int(config['thin'])
    config['step_width'] = [float(config['step_width_tau']), float(config['step_width_theta']), float(config['step_width_gamma'])]
    config['tauprior'] = [float(config['tau_prior_alpha']), float(config['tau_prior_beta'])]
    config['thetaprior'] = [float(config['theta_prior_alpha']), float(config['theta_prior_beta'])]

    return config
