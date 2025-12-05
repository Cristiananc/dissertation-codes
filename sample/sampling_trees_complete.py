"""
Disclaimer: Code refactored with AI assistance.
"""

import math
import random as rd
import numpy as np
import networkx as nx
import copy
from tqdm import tqdm
from time import sleep
from .sampling_trees import find_k_length_path

class TreeSampler:
    def __init__(self, G, T_initial, infected_nodes, flag=0):
        self.G = G
        self.T_current = copy.deepcopy(T_initial)
        self.infected_nodes = infected_nodes  #The fixed observed nodes
        self.flag = flag

        self.nodes_to_sample = infected_nodes
        self.unobserved_leaves = []
        self.samplings = [T_initial]
        self.intermediate_nodes = []

        #Adding intermediate nodes to our list of possible nodes to sample
        #Initially we don't have unobserved nodes as leaves, hence the respective list remains empty.
        for path in T_initial:
            if not path: continue
            for node in path:
                if node not in self.infected_nodes and node not in self.nodes_to_sample:
                    self.intermediate_nodes.append(node)
        
        self.nodes_to_sample += self.intermediate_nodes

    def run(self, n_iterations):
        if self.T_current == [[0]]:
            return None

        for _ in tqdm(range(n_iterations), desc="Sampling trees"):
            if not self.nodes_to_sample:
                break #Safety check

            p = np.random.uniform()

            if p < 1/3:
                #Addition operation
                node_addition = self._choose_random_node(self.nodes_to_sample)
                self._add_neighbor(node_addition)
        
            elif p < 2/3:
                #Changing path operation
                node_change_path = self._choose_random_node(self.infected_nodes)
                self._change_path(node_change_path)

            #Delete operation 
            else:
                if len(self.unobserved_leaves) > 0:
                    node_delete = self._choose_random_node(self.unobserved_leaves)
                    self._delete_node(node_delete)
                else:
                    q = np.random.uniform()
                    if q < 0.5:
                        node_addition = self._choose_random_node(self.nodes_to_sample)
                        self._add_neighbor(node_addition)

                    else:
                        node_change_path = self._choose_random_node(self.infected_nodes)
                        self._change_path(node_change_path)

            #Record state
            self.samplings.append(copy.deepcopy(self.T_current))
            sleep(0.01)

        return self.samplings

    # --------- Helper methods ------------ #

    def _choose_random_node(self, list_of_nodes):
        rand_idx = rd.randrange(0, len(list_of_nodes))
        return list_of_nodes[rand_idx]

    def _get_path_index_for_node(self, node):
        #Finds the index in T_current where the path starts with node
        for i, path in enumerate(self.T_current):
            if path and path[0] == node:
                return i
        return -1
    
    def _clean_intermediate_nodes(self, index):
        current_path = self.T_current[index]

        for intermediate_node in current_path[1:-1]:
            
            self.G.nodes[intermediate_node]['inf_time'] = math.inf       

            if intermediate_node in self.nodes_to_sample:
                self.nodes_to_sample.remove(intermediate_node)

            if intermediate_node in self.intermediate_nodes:
                self.intermediate_nodes.remove(intermediate_node)

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
        return find_k_length_path(
            self.G, target_node, 0, self.G.nodes[target_node]['inf_time'], self.flag
        )

    def _assign_and_track_new_path(self, index, new_path):
        self.T_current[index] = new_path

        #Clean up tracking lists 
        for n in new_path[1:-1]:
            #If a leaf becomes part of a path for the new_node, it is no longer a leaf
            if n in self.unobserved_leaves:
                self.unobserved_leaves.remove(n)

            if n not in self.intermediate_nodes:
                self.intermediate_nodes.append(n)
            
            if n not in self.nodes_to_sample:
                self.nodes_to_sample.append(n)
    
    # ---------- Operations function --------------- #
    def _change_path(self, target_node):
        #print("Attempting to change path ...")

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
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return
    
        new_node = neighbors[rd.randrange(0, len(neighbors))]

        if self.G.nodes[new_node]['inf_time'] == math.inf:
            #print(f"New node added: {new_node}")
            self.G.nodes[new_node]['inf_time'] = self.G.nodes[node]['inf_time'] + 1

            #Update state
            self.T_current.append([new_node, node])
            self.nodes_to_sample.append(new_node)
            self.unobserved_leaves.append(new_node)

            #If the source was a leaf, it is not anymore
            if node in self.unobserved_leaves:
                self.unobserved_leaves.remove(node)


    def _delete_node(self, node):
        t_index = self._get_path_index_for_node(node)

        if t_index != -1:
            #print(f"Node deleted: {node}")

            del self.T_current[t_index]
            self.G.nodes[node]['inf_time'] = math.inf

            self.nodes_to_sample.remove(node)
            self.unobserved_leaves.remove(node)
