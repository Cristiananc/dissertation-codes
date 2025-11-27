
"""
 This file contains the modified DFS algorithm for sampling the feasible trees.  
"""

import math
import random as rd
import numpy as np
import copy
from functools import reduce 


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



def prob_tree_log(G, T, beta):
    """
    Returns the log probability of a transmission tree.
    """

    succes_events = reduce(lambda count, l: count + len(l) - 1, T, 0)
    total_events = G.degree[0]

    for lis in T[1:]:
        for node in lis[0:-1]:
            total_events += G.degree[node] - 1
    
    failed_events = total_events - succes_events

    prob_log = math.log(beta**(succes_events)*(1 - beta)**failed_events)

    return prob_log


def prob_path_log(G, path):
    """
    This is based on the Degree-Biased Random Walk
    Returns the log probability of a path according to the degree of the nodes.
    It should return Inf if a path does not exist.
    Disclaimer: This code was refactored with AI assistance.
    """
    log_prob = 0

    if len(path) < 2:
        return log_prob
  
    #Iterating using a slide window
    for i in range(len(path) - 1):
      u = path[i]
      v = path[i + 1]

    #Check if nodes exist in the graph:
    if u not in G or v not in G:
      return math.inf

    #Check if edge exists in the graph
    if u not in G[v]:
      return math.inf
      
    neighbors = list(G.neighbors(u))
    norm_factor = sum([G.degree[i] for i in neighbors]) 

    if norm_factor == 0:
      return math.inf 

    numerator = G.degree(v)
    log_prob += math.log(numerator) - math.log(norm_factor) 
      
    return log_prob


def metropolis_hastings_approach(G, T_initial, n, infected_nodes, flag=0):
    #Initialize Current State
    current_T = copy.deepcopy(T_initial)
    current_G = copy.deepcopy(G)

    accepted_count = 0

    #Initialize Sampling 
    sampling = [current_T]
    beta = 0.2
    
    for i in range(n):
        #Create proposal objects based on current state
        proposal_T = copy.deepcopy(current_T)
        proposal_G = copy.deepcopy(current_G)
        
        #Simulate a x_prop (proposal step)
        random_node_aux = rd.randrange(0, len(infected_nodes)) 
        random_node = infected_nodes[random_node_aux]

        #Modify the proposal graph (delete previous path)
        #Recall that all nodes in T_current[random_node_aux][1:-1] are unobserved nodes
        for node in current_T[random_node_aux][1:-1]:
            proposal_G.nodes[node]['inf_time'] = math.inf

        #Find a new path for the random node using the proposal graph
        new_path = find_k_length_path(
            proposal_G, 
            random_node, 
            0, 
            proposal_G.nodes[random_node]['inf_time'], 
            flag
        )

        # Update the proposal time 
        proposal_T[random_node_aux] = new_path

        #Compute the acceptance probability 
        prob_tree_prop = prob_tree_log(proposal_G, proposal_T, beta)
        prob_tree_curr = prob_tree_log(current_G, current_T, beta)

        prob_path_prop = prob_path_log(proposal_G, new_path)
        prob_path_curr = prob_path_log(current_G, current_T[random_node_aux])
        
        log_alpha = (prob_tree_prop - prob_tree_curr) + (prob_path_prop - prob_path_curr)
        
        # Acceptance threshold
        alpha = min(0, log_alpha)
        p_uniform = math.log(np.random.uniform())

        # We accept the proposed state
        if p_uniform < alpha:
            # Modify the current state
            current_T = proposal_T
            current_G = proposal_G

            #Counter to track acceptance
            accepted_count += 1

        # Append a unique copy of the current state to the sampling list
        sampling.append(copy.deepcopy(current_T))

    print(f"Final Acceptance Rate: {accepted_count / n:.2%}")

    return sampling