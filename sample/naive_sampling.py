"""
This file performs a naive approach to the problem. 
Given a partial information observed on a node G, it repeats the 
"""

import copy
from helpers import check_feasibility_graphs
from sampling_trees import feasible_tree, bernoulli_process

def naive_sampling(G, sampling_number, infected_nodes, p):

    samplings = []
    G_mutable = copy.deepcopy(G)

    while len(samplings) < sampling_number:

        #Bernoulli process 
        bernoulli_process(G, p)

        if check_feasibility_graphs(G, G_mutable):
            current_T = feasible_tree(G_mutable, infected_nodes)

            while any(None in sublist for sublist in current_T):
                current_T = feasible_tree(G_mutable, infected_nodes)
            
            samplings.append(current_T)

