"""
Disclaimer: Code refactored with AI assistance.
"""

import math
import random as rd
import numpy as np
import networkx as nx
from functools import reduce
import copy
from tqdm import tqdm
from time import sleep
from .search_on_graphs import find_k_length_path

class TreeSampler:
    """
    This function performs an example operation.

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    
    def __init__(self, G, T_initial, infected_nodes, flag=0):
        self.G = G
        self.T_current = copy.deepcopy(T_initial)
        self.infected_nodes = infected_nodes 
        self.flag = flag

        self.nodes_to_sample = infected_nodes
        self.unobserved_leaves = []
        self.samplings = [T_initial]

        #Adding intermediate nodes to our list of possible nodes to sample
        #Initially we don't have unobserved nodes as leaves, hence the respective list remains empty.
        for path in T_initial:
            if not path: continue
            for node in path:
                if node not in self.infected_nodes and node not in self.nodes_to_sample:
                    self.nodes_to_sample.append(node)
        
    def run(self, n_iterations):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        if self.T_current == [[0]]:
            return None
        
        accepted_count = 0
        
        for _ in tqdm(range(n_iterations), desc="Sampling trees"):
            if not self.nodes_to_sample:
                break #Safety check
            
            valid_proposal = True

            #Drawing beta from a gamma distribution
            shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
            beta = np.random.beta(shape, scale)

            previous_T = copy.deepcopy(self.T_current)
            previous_G = copy.deepcopy(self.G)

            q_ratio = 1
            p = np.random.uniform()

            if p < 1/3:
                #Addition operation
                node_addition = self._choose_random_node(self.nodes_to_sample)
                len_neigh = self._add_neighbor(node_addition)

                if len_neigh is not None:
                    q_ratio = math.log((len(self.nodes_to_sample) *len_neigh)/ len(self.unobserved_leaves))

            elif p < 2/3:
                #Changing path operation
                node_change_path = self._choose_random_node(self.infected_nodes)
                self._change_path(node_change_path)

                #We assume the number of available paths doesn't change much between states
                #so q_ratio = 1

            else:
                #Delete operation 
                if len(self.unobserved_leaves) > 0:
                    node_delete = self._choose_random_node(self.unobserved_leaves)
                    parent_node = self._delete_node(node_delete)

                    valid_proposal, q_ratio = self._compute_q_ratio_deletion(parent_node)

            #Compute the acceptance probability 
            alpha = self._compute_acceptance_prob(q_ratio, beta, previous_G, previous_T)
            
            p_uniform = math.log(np.random.uniform())

            # We accept the proposed state
            if p_uniform < alpha:
                #Counter to track acceptance
                accepted_count += 1
        
            else:
                self.G = copy.deepcopy(previous_G)
                self.T_current = copy.deepcopy(previous_T)

            #Record state
            self.samplings.append(copy.deepcopy(self.T_current))
                        
            sleep(0.001)

        print(f"Final Acceptance Rate: {accepted_count / n_iterations:.2%}")

        return self.samplings

    # --------- Helper methods ------------ #

    def _choose_random_node(self, list_of_nodes):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        rand_idx = rd.randrange(0, len(list_of_nodes))
        return list_of_nodes[rand_idx]

    def _get_path_index_for_node(self, node):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        for i, path in enumerate(self.T_current):
            if path and path[0] == node:
                return i
        return -1
    
    def _clean_intermediate_nodes(self, index):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        current_path = self.T_current[index]

        for intermediate_node in current_path[1:-1]:
            
            self.G.nodes[intermediate_node]['inf_time'] = math.inf       

            if intermediate_node in self.nodes_to_sample:
                self.nodes_to_sample.remove(intermediate_node)

            #Remove their descendants as well
            descendants = nx.descendants(self.G, intermediate_node)

            for descendant in descendants:

                if descendant in self.infected_nodes:
                    continue

                if descendant in self.nodes_to_sample:
                    self.nodes_to_sample.remove(descendant)
                    if descendant in self.unobserved_leaves:
                        self.unobserved_leaves.remove(descendant)

                    #Remove path that contains the descendant
                    node_index = self._get_path_index_for_node(descendant)
                    if node_index != -1:
                        del self.T_current[node_index]

    def _calculate_new_path(self,target_node):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        return find_k_length_path(
            self.G, target_node, 0, self.G.nodes[target_node]['inf_time'], self.flag
        )

    def _assign_and_track_new_path(self, index, new_path):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
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
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        #print("Changing path ...")

        t_index = self._get_path_index_for_node(target_node)

        if t_index == -1:
            # If the node is intermediate, it might not have its own path in T_current
            return
        
        self._clean_intermediate_nodes(t_index)

        #Calculates the new path
        new_path = self._calculate_new_path(target_node)

        #Since we deleted the descendants of intermediate nodes
        #it might have changed the index for the path
        t_index = self._get_path_index_for_node(target_node)

        if new_path is not None:
            self._assign_and_track_new_path(t_index, new_path)


    def _add_neighbor(self, node):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        neighbors = list(self.G.neighbors(node))
        neighb_available = [node for node in neighbors if self.G.nodes[node]['inf_time'] == math.inf]        
        
        if not neighb_available:
            return

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

        return len(neighb_available)

    def _delete_node(self, node):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        t_index = self._get_path_index_for_node(node)
        parent_node = self.T_current[t_index][0]

        if t_index != -1:
            #print(f"Node deleted: {node}")

            # Delete path from T_current 
            del self.T_current[t_index]
            self.G.nodes[node]['inf_time'] = math.inf

            list_index = self.nodes_to_sample.index(node)

            #Swap the element
            last_element = self.nodes_to_sample[-1]
            self.nodes_to_sample[list_index] = last_element

            self.nodes_to_sample.pop()

            self.unobserved_leaves.remove(node)
        
            return parent_node

    # ---------- Compute probabilities -------------------#
    def _prob_tree_log(self, G, T, beta):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        succes_events = reduce(lambda count, l: count + len(l) - 1, T, 0)
        total_events = G.degree[0]

        for lis in T[1:]:
            for node in lis[0:-1]:
                total_events += G.degree[node] - 1
        
        failed_events = total_events - succes_events

        beta_aux = 1 - beta
        prob_log = succes_events*math.log(beta)+ failed_events*math.log(beta_aux)

        return prob_log
    
    def _compute_acceptance_prob(self, q_ratio, beta, previous_G, previous_T):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        prob_tree_prop = self._prob_tree_log(self.G, self.T_current, beta)
        prob_tree_curr = self._prob_tree_log(previous_G, previous_T, beta)

        # Returns acceptance threshold
        alpha_aux = prob_tree_prop - prob_tree_curr + q_ratio
        alpha = min(0, alpha_aux) 
        return alpha
    
    def _compute_q_ratio_deletion(self, parent_node):
        """
        This function performs an example operation.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

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
