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

        #Adding intermediate nodes to our list of possible nodes to sample
        #Initially we don't have unobserved nodes as leaves, hence the respective list remains empty.
        for path in T_initial:
            for node in path:
                if node not in self.infected_nodes and node not in self.nodes_to_sample:
                    self.nodes_to_sample.append(node)

    # ---- Helper methods ----
    def _handle_observed_nodes(self, node, index):
        # Logic for observed nodes: 
        if np.random.uniform() < 0.5:
            self._change_path(node, index)
        else:
            self._add_neighbor(node)

    def _handle_unobserved_leaf(self, node, index):
        # Logic for unobserved leaves: 1/3 add, 1/3 change path, 1/3 delete path.
        p = np.random.uniform()
        if p < 1/3:
            self._add_neighbor(node)
        elif p < 2/3:
            self._change_path(node, index)
        else:
            self._delete_node(node, index)
    
    def _handle_intermediate_node(self, node):
        #The logic for intermediate unobserved nodes is addition of a neighbor node
        self._add_neighbor(node)
    
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

            #Clean up tracking lists 
            for n in new_path[1:]:
                #If a leaf becomes part of a path for the new_node, it is no longer a leaf
                if n in self.unobserved_leaves:
                    self.unobserved_leaves.remove(n)


    def _add_neighbor(self, node):
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return
    
        new_node = neighbors[rd.randrange(0, len(neighbors))]


        if self.G.nodes[new_node]['inf_time'] == math.inf:
            print(f"New node added: {new_node}")
            self.G.nodes[new_node]['inf_time'] = self.G.nodes[node]['inf_time'] + 1

        #Update state
        self.T_current.append([new_node, node])
        self.nodes_to_sample.append(new_node)
        self.unobserved_leaves.append(new_node)

        #If the source was a leaf, it is not anymore
        if node in self.unobserved_leaves:
            self.unobserved_leaves.remove(node)


    def _delete_node(self, node, index):
        print(f"Node deleted: {node}")
        del self.T_current[index]
        self.G.nodes[node]['inf_time'] = math.inf

        if node in self.nodes_to_sample:
            self.nodes_to_sample.remove(node)
            if node in self.unobserved_leaves:
                self.unobserved_leaves.remove(node)


