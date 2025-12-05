# Importing the required modules and packages
import networkx as nx
import pickle
from epidemic_simulation.sir_simulation import fast_SIR
from epidemic_simulation.helpers import *
from sample.helpers import *
from sample.sampling_trees import *
from sample.naive_sampling import naive_sampling
import time
from sample.sampling_trees_complete import TreeSampler

#Loading our example of a random network
#with open('data/graphs/20/G_erdos_renyi0_20.pkl', mode='rb') as f:
#    G = pickle.load(f)

#Creating initial graph example
n = 20
p = 5/(n - 1)
G = nx.erdos_renyi_graph(n, p)

beta = 0.4
fast_SIR(G, [0], beta)
print(f"Real infection times: {nx.get_node_attributes(G, "inf_time")}")

#Selecting a fraction of nodes that will not be observed.
p_excluded = 0.4
excluded, infected_nodes = excludeInfTime(G, p_excluded)

#Deleting nodes that we known were not infected from the graph.
delete_susceptibles(G)

T_initial = feasible_tree(G, infected_nodes)

#Repeat until a tree doesn't have a None path on it.
#"Error: 'NoneType' object is not subscri table
while None in T_initial:
    T_initial = feasible_tree(G, infected_nodes)
print("--------------------------------------------------------------------------------------------")
print(T_initial)
print("--------------------------------------------------------------------------------------------")
#print(check_feasibility_tree(G,T_initial))

samplings_number = 10000
#samplings = sampling_trees(G, T_initial, samplings_number, infected_nodes, flag=2)

#Proportion of nodes found
#nodes_prop = nodes_proportion(G, samplings)
#print(nodes_prop)

#samplings = metropolis_hastings_approach(G, T_initial, samplings_number, infected_nodes, flag=2)
#nodes_prop = nodes_proportion(G, samplings)
#print(nodes_prop)

#Naive sampling
#start_time = time.time()
#naive_sampling = naive_sampling(G, samplings_number, infected_nodes, [0])
#print("--- %s seconds ---" % (time.time() - start_time))
#print(nodes_proportion_list(G, naive_sampling))

#Adding new operations to the sampling 
#Initialize
sampler = TreeSampler(G, T_initial, infected_nodes,flag=1)

#Run
sampling = sampler.run(n_iterations=samplings_number)
print(f"Observed infection times: {nx.get_node_attributes(G, "inf_time")}")
print("--------------------------------------------------------------------------------------------")
print(f"Frequency of nodes: {nodes_proportion(G, sampling)}")
