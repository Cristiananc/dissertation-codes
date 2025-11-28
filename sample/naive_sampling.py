"""
This file performs a naive approach to the problem. 
"""

import copy
from .helpers import check_feasibility_graphs
from .sampling_trees import feasible_tree
from epidemic_simulation.sir_simulation import fast_SIR

def naive_sampling(G, sampling_number, infected_nodes, initial_infecteds, p, n):

    samplings = []
    G_mutable = copy.deepcopy(G)

    while len(samplings) < sampling_number:

        #In place modification of G
        fast_SIR(G_mutable, initial_infecteds, p)

        if check_feasibility_graphs(G, G_mutable, infected_nodes):
            current_T = feasible_tree(G_mutable, infected_nodes)

            while any(None in sublist for sublist in current_T):
                current_T = feasible_tree(G_mutable, infected_nodes)
            
            samplings.append(current_T)

    return samplings