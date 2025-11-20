# Importing the required modules and packages
import networkx as nx
from epidemic_simulation.sir_simulation import fast_SIR
from epidemic_simulation.helpers import *
from sample.helpers import *
from sample.sampling_tress import feasible_tree, sampling_trees
import scipy as sp

#Creating our initial example of a random network
n = 20
p = 5/(n - 1)
G = nx.erdos_renyi_graph(n, p)

beta = 0.4
fast_SIR(G, [0], beta, n)

#Selecting a fraction of nodes that will not be observed.
p_excluded = 0.2
excluded, infected_nodes = excludeInfTime(G, p_excluded)
#print(nx.get_node_attributes(G, "inf_time"))
#print(nx.adjacency_matrix(G))

#Deleting nodes that we known were not infected from the graph.
delete_susceptibles(G)

T_initial = feasible_tree(G, infected_nodes)

samplings_number = 100
samplings = sampling_trees(G, T_initial, samplings_number, infected_nodes, flag=2)

#Proportion of nodes found
nodes_prop = nodes_proportion(G, samplings, samplings_number)
print(infected_nodes)
print(nodes_prop)


"""
Tasks to do: Exclude a tree if it has a None path on it.
"Error: 'NoneType' object is not subscriptable
"""
