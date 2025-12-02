"""
This file performs a naive approach to the problem. 
"""

import numpy as np
import networkx as nx
import copy
import math
from .helpers import check_feasibility_graphs
from epidemic_simulation.sir_simulation import fast_SIR

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