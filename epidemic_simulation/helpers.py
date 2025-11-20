#This file contains the auxiliary functions to perform the epidemic simulations 
import math
import numpy as np
import networkx as nx

#This function deletes the infection time information from a fraction p of nodes
# INPUT: A graph G in networkx format and a decimal p which stands for the
# desired fraction to be deleted.
def excludeInfTime(G, p):
    N = round(p*len(G.nodes))

    #We remove the node 0 from the possible removed nodes.
    excluded_nodes = np.random.choice(list(G.nodes)[1:], N, replace = False)

    for node in excluded_nodes:
        G.nodes[node]['inf_time'] = math.inf
        G.nodes[node]['status'] = ''

    obs_infected_nodes = [k for k,v in nx.get_node_attributes(G, 'status').items() if v.startswith('r')]


    return excluded_nodes, obs_infected_nodes


# Deleting nodes we know were not infected from the graph
def delete_susceptibles(G):
    node_statuses = nx.get_node_attributes(G, "status")
    removable_indices = [i for i, x in node_statuses.items() if x == 'susceptible']
    
    for node in removable_indices:
        G.remove_node(node)