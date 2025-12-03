"""
Disclaimer: Code refactored with AI assistance.
"""

import math
import random as rd
import numpy as np
import copy
from sampling_trees import find_k_length_path

class TreeSampler:
    def __init__(self, G, T_initial, infected_nodes, flag=0):
        self.G = G
        self.T_current = copy.deepcopy(T_initial)
        self.infected_nodes = infected_nodes  #The fixed observed nodes
        self.flag = flag

        self.nodes_to_sample = infected_nodes
        self.unobserved_leaves = []
        self.samplings = [T_initial]

    # ---- Helper methods ----
    def _handle_observed_nodes(self, node):
        # Logic for observed nodes: 
        if np.random.uniform() < 0.5:
            self._change_path(node)
        else:
            self._add_neighbor(node)

    def _handle_unobserved_leaf(self, node):
        # Logic for unobserved leaves: 1/3 add, 1/3 change path, 1/3 delete path.
        p = np.random.uniform()
        if p < 1/3:
            self._add_neighbor(node)
        elif p < 2/3:
            self._change_path(node)
        else:
            self._delete_node(node)
    
    # ---- Operations function ----
    def _change_path(self, node, index):
        print("Attempting to change path ...")

        current_path = self.T_current[index]

        #Reset inf_time for intermediate nodes
        if current_path is not None:
            for node in current_path[1:-1]:
                self.G.nodes[node]['inf_time'] = math.inf

        #Calculates the new path
        new_path = find_k_length_path(
            self.G, node, 0, self.G.nodes[node]['inf_time'], self.flag
        )

        if new_path is not None:
            self.T_current[index] = new_path

# ---------------------------- REVIEW THIS LOGIC ---------------------------
            #Clean up tracking lists 
            for n in new_path[1:]:
                #Here we have that an observed node now has an observed node as a child
                if n in self.nodes_to_sample and n not in self.infected_nodes:
                    self.nodes_to_sample.remove(n)
                    if n in self.unobserved_leaves:
                        self.unobserved_leaves.remove(n)


    def _add_neighbor(self):
        