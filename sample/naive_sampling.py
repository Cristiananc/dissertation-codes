"""
This file performs a naive approach to the problem. 
Given a partial information observed on a node G, it repeats the 
"""

import copy
from helpers import check_feasibility_graphs
from sampling_trees import feasible_tree
import numpy as np

def bernoulli_trial(p):
    u = np.random.uniform()

    if u < p:
        return True

    else: 
        return False

def bernoulli_process(G, p):
    curr_time = 1
    curr_infected = {'node': 0, 'time': curr_time}
    G_mutable = copy.deepcopy()

    while curr_infected:
        for node in curr_infected:
            for v in G.neighbors(node):
                if bernoulli_process(p):
                    G_mutable.nodes[v]['inf_time'] = curr_time + 1
                    curr_infected.append(v)

    return None


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

