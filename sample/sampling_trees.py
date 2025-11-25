
"""
 This file contains the modified DFS algorithm for sampling the feasible trees.  
"""

import math
import random as rd
import numpy as np
import copy
from helpers import *

# INPUT:
# A flag that states which version of the DFS modified we would like to use. 
# OUTPUT:
def DFS(G, v, t,k, path, visited, flag=0): 
    #Disclaimer: Code refactored with AI assistance.

    # Base Case: Sucess
    #If the current node is the target node and the length of the path - 1 is k, we have found
    #a desired path 
    if v == t and len(path) == k:
        return path
    
    # Base Case: Path too long
    if len(path) > k:
        return None
    
    # Base Case: Target reached too early 
    #If the target node is reached before the length of the desired path, then no path is returned
    if len(path) < k and v == t:
        return None
    
    # Observed Node Check (Existing "Infection Time" Logic)
    # If we hit a node that was already visited in a previous valid context
    # and the timing matches, we consider this path valid
    if len(path) > 1 and (k -  G.nodes[v]['inf_time'] == len(path) - 1):
        return path

    #If the time doesn't match we return None
    if G.nodes[v]['inf_time'] != math.inf and (k - G.nodes[v]['inf_time'] != len(path) - 1):
        return None

    # Set Infection Time
    # Only set this if it hasn't been set (implied by passing step 4)
    # Each unobserved node we pass in the path receives a infection time.
    if G.nodes[v]['inf_time'] == math.inf:
        G.nodes[v]['inf_time'] = k - len(path) + 1
        reset_inf_time = True #A flag to remember we need to reset it later
    
    else:
        reset_inf_time = False # We didn't change it
    
    visited[v] = True

    # Neighbor Handling
    neighbors  = list(G.neighbors(v))

    # Check in case the node doesn't have neighbors:
    if not neighbors:
        visited[v] = False

        #In case we changed the inf time, reset it back to infinite
        if reset_inf_time:
            G.nodes[v]['inf_time'] = math.inf
        
        return None

    # The flag determines in which ordering the neighbors will be consider
    # The default ordering is 0.
    if flag == 1:
        rd.shuffle(neighbors) #rd.shuffle is an in-place function

    # We shuffle the list, considering the nodes with higher degree
    # may appear first in the list. 
    elif flag == 2:
        neighbors_degree = []
        
        for node in neighbors:
            neighbors_degree.append(G.degree(node))

        norm = sum(neighbors_degree)
        probs = [d/norm for d in neighbors_degree]
        shuffled_list = np.random.choice(neighbors, len(neighbors), replace = False, p=probs)
        neighbors = shuffled_list.tolist()

    # Recursive Step
    for neighbor in neighbors:
        if not visited[neighbor]:
            result = DFS(G, neighbor, t, k, path + [neighbor], visited, flag)
            #If a desired path is returned we return it
            if result:
                return result
    
    # Backtracking
    #The lines below deals with the process of backtracking in the search
    visited[v] = False
    # Only reset inf_time if We changed it in this specific call
    if reset_inf_time:
        G.nodes[v]['inf_time'] = math.inf


def find_k_length_path(G, s, t, k, flag=0):
    visited = {}
    for node in G.nodes:
        visited[node] = False
    return DFS(G, s, t, k, [s], visited, flag)

def feasible_tree(G, infected_nodes, flag=0):
    tree = []
    for node in infected_nodes:
        tree_aux = find_k_length_path(G, node, 0, G.nodes[node]['inf_time'], flag)
        tree.append(tree_aux)
    return tree


def sampling_trees(G,T_initial,n, infected_nodes, flag=0):
  sampling = [T_initial]

  for i in range(n):
    T_current = sampling[i]
    random_node_aux = rd.randrange(0, len(infected_nodes)) 
    random_node = infected_nodes[random_node_aux]

    #Delete the previous path from G
    #Recall that all nodes in T_current[random_node_aux][1:-1] are unobserved nodes
    for node in T_current[random_node_aux][1:-1]:
      G.nodes[node]['inf_time'] = math.inf

    #Find a new path for
    new_path = find_k_length_path(G, random_node, 0, G.nodes[random_node]['inf_time'], flag)

    # Modify the current state
    T_current[random_node_aux] = new_path

    # Append a unique copy of the current state to the sampling list
    sampling.append(copy.deepcopy(T_current))

  return sampling

def metropolis_hastings_approach(G, T_initial, n, infected_nodes, flag=0):
    #First choose an arbitrary state for X_0
    sampling = [T_initial]
    beta = 0.4

    G_aux = copy.deepcopy(G)
    
    for i in range(n):
        T_current = sampling[i]
        
        #Simulate a value X_prop
        random_node_aux = rd.randrange(0, len(infected_nodes)) 
        random_node = infected_nodes[random_node_aux]

        #Delete the previous path from G_aux
        #Recall that all nodes in T_current[random_node_aux][1:-1] are unobserved nodes
        for node in T_current[random_node_aux][1:-1]:
            G_aux.nodes[node]['inf_time'] = math.inf

        #Find a new path for the random node
        new_path = find_k_length_path(G_aux, random_node, 0, G_aux.nodes[random_node]['inf_time'], flag)

        T_aux = copy.deepcopy(T_current)
        T_aux[random_node_aux] = new_path

        #Compute the acceptance probability 
        prob_tree = prob_tree_log(G_aux, T_aux, beta) - prob_tree_log(G, T_current, beta) 
        prob_path = prob_path_log(G_aux, new_path) - prob_path_log(G, T_current[random_node])
        alpha = min(0, prob_tree + prob_path)


        p_uniform = math.log(np.random.uniform())

        # We accept the proposed state
        if p_uniform < alpha:
            # Modify the current state
            T_current[random_node_aux] = new_path
            G = G_aux

            # Append a unique copy of the current state to the sampling list
            sampling.append(copy.deepcopy(T_current))

    return sampling