"""
Disclaimer: Code refactored with AI assistance.
"""

import math
import random as rd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce
import copy
from tqdm import tqdm
from .search_on_graphs import find_k_length_path

class TreeSamplerMH:
    """
    MCMC Sampler for inferring infected nodes on a network with partial 
    information available.
    """
    
    def __init__(self, G, T_initial, infected_nodes, flag=0):
        self.G = G
        self.T_current = copy.deepcopy(T_initial)
        self.infected_nodes = infected_nodes 
        self.flag = flag

        self.nodes_to_sample = infected_nodes
        self.unobserved_leaves = []
        self.samplings_trees = [T_initial]
        self.log_likelihood_history = []

        # Initialization logic
        # Adding intermediate nodes to our list of possible nodes to sample
        for path in T_initial:
            if not path: continue
            for node in path:
                if node not in self.infected_nodes and node not in self.nodes_to_sample:
                    self.nodes_to_sample.append(node)
        
    def run(self, n_iterations):
        """
        This function executes the Metropolis-Hastings sampling loop.

        Args:
            n_iterations (int): The number of iterations of the loop.

        Returns:
            self.samplings_trees (list): List of states sampled.
        """

        if self.T_current == [[0]]:
            return None
        
        accepted_count = 0
        
        for _ in tqdm(range(n_iterations), desc="Sampling trees"):
            if not self.nodes_to_sample:
                break 
            
            # Beta clamping to prevent math domain error
            shape, scale = 2., 2.  
            beta = np.random.beta(shape, scale)
            beta = max(1e-9, min(beta, 1 - 1e-9))

            # Capture full state for reversion
            previous_T = copy.deepcopy(self.T_current)
            previous_G = copy.deepcopy(self.G)
            previous_nodes_list = list(self.nodes_to_sample)
            previous_leaves_list = list(self.unobserved_leaves)

            valid_proposal, q_ratio = self._propose_next_state()

            current_ll = self._prob_tree_log(previous_G, previous_T, beta)

            if valid_proposal:
                alpha = self._compute_acceptance_prob(q_ratio, beta, previous_G, previous_T)
                p_uniform = math.log(np.random.uniform())

                if p_uniform < alpha:
                    #ACCEPT
                    accepted_count += 1
                    self.samplings_trees.append(copy.deepcopy(self.T_current))
                    self.log_likelihood_history.append(self._prob_tree_log(self.G, self.T_current, beta))

                else:
                    #REJECT
                    self._revert_state(previous_G, previous_T, previous_nodes_list, previous_leaves_list)
                    self.samplings_trees.append(copy.deepcopy(self.T_current))
                    self.log_likelihood_history.append(current_ll)

            else:
                self._revert_state(previous_G, previous_T, previous_nodes_list, previous_leaves_list)
                self.samplings_trees.append(copy.deepcopy(self.T_current))
                self.log_likelihood_history.append(current_ll)
            
            """
            print()
            print(f"Current Tree: {self.T_current}")
            print()
            print(f"Nodes to sample: {self.nodes_to_sample}")
            print()
            print(f"Unobserved leaves: {self.unobserved_leaves}")
            print()
            print(f"Current infection times: {nx.get_node_attributes(self.G, "inf_time")}")
            """

        print(f"Final Acceptance Rate: {accepted_count / n_iterations:.2%}")
        return self.samplings_trees

    # --------- Helper methods ------------ #

    def _propose_next_state(self):
        """
        Handles the logic for proposing a move.

        Returns:
            (bool, float) (tuple) -> (valid_proposal, q_ratio).
        """
        
        p = np.random.uniform()
        q_ratio = 0 # Default (log(1) = 0)
        valid_proposal = True

        if p < 1/3:
            #Addition
            node_addition = self._choose_random_node(self.nodes_to_sample)
            len_neigh = self._add_neighbor(node_addition)

            if len_neigh is not None and len_neigh > 0:
                # Foward: 1/N * 1/len_neigh | Reverse: 1/Leaves_new
                n_sample = len(self.nodes_to_sample)
                n_leaves_new = len(self.unobserved_leaves)

                if n_leaves_new > 0:
                    q_ratio = math.log((n_sample *len_neigh)) - math.log(n_leaves_new)

                else:
                    valid_proposal = False
            
            else:
                valid_proposal = False

        elif p < 2/3:
            #Changing path
            node_change_path = self._choose_random_node(self.infected_nodes)
            self._change_path(node_change_path)

            #We assume the number of available paths doesn't change much between states
            q_ratio = 0 #Assumed symmetric

        else:
            #Deletion 
            if len(self.unobserved_leaves) > 0:
                node_delete = self._choose_random_node(self.unobserved_leaves)
                parent_node = self._delete_node(node_delete)
                valid_proposal, q_ratio = self._compute_q_ratio_deletion(parent_node)
            
            else:
                valid_proposal = False

        return valid_proposal, q_ratio

    def _revert_state(self, prev_G, prev_T, prev_nodes, prev_leaves):
        """
        Clean helper to revert all state components.
        """
        self.G = copy.deepcopy(prev_G)
        self.T_current = copy.deepcopy(prev_T)
        self.nodes_to_sample = list(prev_nodes)
        self.unobserved_leaves = list(prev_leaves)


    def _choose_random_node(self, list_of_nodes):
        """
        Chooses a random element uniformly from a list.

        Args:
            list_of_nodes (list): List with nodes (integers).

        Returns:
            random_node (int): Random element from list_of_nodes.
        """

        rand_idx = rd.randrange(0, len(list_of_nodes))
        random_node = list_of_nodes[rand_idx]

        return random_node

    def _get_path_index_for_node(self, node):
        """
        Gets the index of a specific element in a list.

        Args:
            node (int): The specific element to be found.

        Returns:
            i (int): -1 if element is not found, otherwise the index of node in the list.
        """

        for i, path in enumerate(self.T_current):
            if path and path[0] == node:
                return i
        return -1

    def _is_node_used_in_tree(self, node):
        """
        Checks if an intermediate node is part of any other path in T_current.
        
        node (int): The node which we want to be checked.
        """
        counter = 0
        for path in self.T_current:
            if node in path:
                counter +=1
        
        if counter > 1: return True
        else: return False


    def _calculate_new_path(self,source_node):
        """
        Finds a path of fixed length from a source node to a target node.

        Args:
            source_node (int): Source node for the s-t k-path search.

        Returns:
            new_path (list): s-t k-path found.
        """

        new_path = find_k_length_path(
            self.G, source_node, 0, self.G.nodes[source_node]['inf_time'], self.flag
        )

        return new_path

    def _assign_and_track_new_path(self, index, new_path):
        """
        Assign a new_path for node in a given index.

        Args:
            index (int): Index of the node that is the source of the new path.
            new_path (list): Path of fixed length from a source node to a target node.
        """

        self.T_current[index] = new_path

        #Clean up tracking lists 
        for n in new_path[1:-1]:
            #If a leaf becomes part of a path for the new_node, it is no longer a leaf
            if n in self.unobserved_leaves:
                self.unobserved_leaves.remove(n)
            
            if n not in self.nodes_to_sample:
                self.nodes_to_sample.append(n)
    
    # ---------- Operations function --------------- #
    def _change_path(self, target_node):
        """
        This function performs a change of path operation.

        Args:
            target_node (int): The node for which the path in the current tree will be changed.
        """
        t_index = self._get_path_index_for_node(target_node)
        
        if t_index == -1: return
        
        old_path = self.T_current[t_index]

        if len(old_path) > 1:
            old_parent = old_path[1]
        else:
            old_parent = None

        new_path = self._calculate_new_path(target_node)

        if new_path is not None:
            self._assign_and_track_new_path(t_index, new_path)

            if old_parent is not None:
                if not self._is_node_used_in_tree(old_parent):

                    if old_parent not in self.unobserved_leaves:
                        self.unobserved_leaves.append(old_parent)
                
                    self.T_current.append(old_path[1:])
                    

    def _add_neighbor(self, node):
        """
        Performs an addition of a node operation.

        Args:
            node (int): The parent of the new node that will be added.

        Returns:
            len_neighb_available (int): The size of the list of available neighbors of the 
            parent node that can be added in the tree.
        """

        neighbors = list(self.G.neighbors(node))
        neighb_available = [node for node in neighbors if self.G.nodes[node]['inf_time'] == math.inf]        
        
        if not neighb_available: return None

        new_node =  neighb_available[rd.randrange(0, len( neighb_available))]

        #print(f"New node added: {new_node}")
        self.G.nodes[new_node]['inf_time'] = self.G.nodes[node]['inf_time'] + 1

        #Update state
        self.T_current.append([new_node, node])
        self.nodes_to_sample.append(new_node)
        self.unobserved_leaves.append(new_node)

        #If the source was a leaf, it is not anymore
        if node in self.unobserved_leaves:
            self.unobserved_leaves.remove(node)

        len_neighb_available = len(neighb_available)

        return len_neighb_available

    def _delete_node(self, node):
        """
        Performs a deletion of a node operation.

        Args:
            node (int): The node which will be deleted from the current feasible tree.

        Returns:
            parent_node (int or None): The parent of the deleted node in the current feasible tree.
        """
        t_index = self._get_path_index_for_node(node)
        if t_index == -1: return None

        current_path = list(self.T_current[t_index])

        #The path structure is [Leaf, Parent] based on _add_neighbor logic.
        # Hence the parent is at index 1.
        if len(current_path) > 1: parent_node = current_path[1]
        else: parent_node = None

        #Reset infection time for deleted node
        self.G.nodes[node]['inf_time'] = math.inf

        if node in self.nodes_to_sample:
            idx = self.nodes_to_sample.index(node)

            #Swap and pop
            self.nodes_to_sample[idx] = self.nodes_to_sample[-1]
            self.nodes_to_sample.pop()
        
        if node in self.unobserved_leaves:
            self.unobserved_leaves.remove(node)

        del self.T_current[t_index]

        #Handle the parent node
        if parent_node is not None:
            if not self._is_node_used_in_tree(parent_node):
                retained_path = current_path[1:]

                if len(retained_path) > 0:
                    self.T_current.append(retained_path)

                    if parent_node not in self.infected_nodes:
                        self.unobserved_leaves.append(parent_node)
                    
                    if parent_node not in self.nodes_to_sample:
                        self.nodes_to_sample.append(parent_node)

        
        return parent_node

    # ---------- Compute probabilities -------------------#
    def _prob_tree_log(self, G, T, beta):
        """
        This function calculates the log-likelihood of a tree for a given infection rate.

        Args:
            G (networkx graph): Graph for which we will calculate the failed infection events.
            T (list): The transmission tree for which we will compute the log-likelihood.
            beta (float): Infection rate.

        Returns:
            prob_log (float): The log-likelihood for T for a fixed beta.
        """

        #Identify all nodes in the tree
        nodes_in_tree = set()
        for path in T:
            for node in path:
                nodes_in_tree.add(node)

        if len(nodes_in_tree) <= 1:
            return 0
        
        # Succes Events (V_T - 1)
        succes_events = len(nodes_in_tree) - 1
        failed_events = 0

        for u in nodes_in_tree:
            for v in G.neighbors(u):
                if v not in nodes_in_tree:
                    failed_events += 1
        
        log_beta = math.log(max(1e-9, beta))
        log_beta_aux = math.log(max(1e-9, 1 - beta)) # Use 1-beta here
        
        prob_log = succes_events * log_beta + failed_events * log_beta_aux
        print(beta)
        print(f"Succes events: {succes_events}")
        print(f"Failed events: {failed_events}")
        print(prob_log)

        return prob_log
    
    def _compute_acceptance_prob(self, q_ratio, beta, previous_G, previous_T):
        """
        Calculates alpha using the log-likelihoods

        Args:
            q_ratio (float): Proposal distribution ratio.
            beta (float): Infection rate.
            previous_G (networkx graph): Graph G before proposal.
            previos_T (list): Feasible tree before proposal.

        Returns:
            alpha (float): Acceptance probability for proposal.
        """

        prob_tree_prop = self._prob_tree_log(self.G, self.T_current, beta)
        prob_tree_curr = self._prob_tree_log(previous_G, previous_T, beta)

        print(f"log-likelihood: {prob_tree_curr} ")

        # Alpha = (Log P_new - Log P_old) + Log Q_ratio
        alpha_aux = prob_tree_prop - prob_tree_curr + q_ratio

        alpha = min(0, alpha_aux) 
        return alpha
    
    def _compute_q_ratio_deletion(self, parent_node):
        """
        Calculates the Hastings ratio for deletion

        Args:
            parent_node (int): The parent node of the deleted node.

        Returns:
            (bool, float) -> (valid_proposal, q_ratio)
        """

        valid_proposal = False
        q_ratio = 0

        if parent_node is not None:

            neigh_available = 0

            # Recalculate available neighbors for the reverse move (Addition)
            for neighbor in self.G.neighbors(parent_node):

                if self.G.nodes[neighbor]['inf_time'] == math.inf:
                    neigh_available += 1

            n_sample_new = len(self.nodes_to_sample)
            
            # Check for the Math Domain Error condition (Denominator must be non-zero)
            if n_sample_new == 0 or neigh_available == 0:
                valid_proposal = False

            else:
                # q(T|T') / q(T'|T) = N_leaves_old / (N_sample_new * neigh_available)
                n_leaves_old = len(self.unobserved_leaves) + 1
                q_ratio = math.log(n_leaves_old) - math.log(n_sample_new) - math.log(neigh_available)
                valid_proposal = True

        else:
            # Deletion failed
            valid_proposal = False     

        return valid_proposal, q_ratio

    # ---------------- Visualise results ------------------------- #
    def _trace_plot_log_likelihood(self):
        """
        Plots the trace plot for the log likelihood of the samplings.
        """
        if len(self.log_likelihood_history) <= 1:
            print("No sampling has been performed yet!")
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(self.log_likelihood_history)
            #plt.title("Log-Likelihood Trace")
            plt.xlabel("Iteration")
            plt.ylabel("Log-Likelihood")
            plt.show()

    def _size_of_the_tree(self, T):
        """
        Calculates the size of a tree.

        Args:
            T (list): A list of lists with the paths of the tree.

        Returns:
           tree_size (int): Returns the size of the tree |V_T|
        """
        #Identify all nodes in the tree
        nodes_in_tree = set()
        for path in T:
            nodes_in_tree.update(path)
        
        tree_size = len(nodes_in_tree)

        return tree_size
    
    def _trace_plot_tree_size(self):
        return None
