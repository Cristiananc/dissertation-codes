import math
import random as rd
import numpy as np
import networkx as nx
import copy
from tqdm import tqdm
from time import sleep

#Importing from local files
from .helpers import prob_path_log, prob_tree_log
from search_on_graphs import *
from ..epidemic_simulation.sir_simulation import fast_SIR
from .helpers import check_feasibility_graphs

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

def naive_sampling(G, sampling_number, observed_nodes, initial_infecteds):

    samplings = []
    G_mutable = copy.deepcopy(G)

    while len(samplings) < sampling_number:

        p = np.random.uniform()

        #In place modification of G
        fast_SIR(G_mutable, initial_infecteds, p)

        if check_feasibility_graphs(G, G_mutable, observed_nodes):
            all_nodes = nx.get_node_attributes(G_mutable, "inf_time")
            nodes_infected = [node for node, inf_time in all_nodes.items() if inf_time < math.inf]

            samplings.append(nodes_infected)

    return samplings

