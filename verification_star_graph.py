#Disclaimer: this was moved to the tests folder for organization
# but the file should be run in the main folder
import networkx as nx
import copy
import matplotlib.pyplot as plt
import math
import pickle

#Importing from the function I created
from sample.search_on_graphs import *
from sample.sampling_trees_degree_change_path_addition import TreeSampler
from sample.helpers import nodes_proportion

#Start graph
e = [(0,1), (0,2), (0,3)] #list of edges
G = nx.Graph(e)

pos = {0: [0,0], 1: [1,0], 2: [0,0.5], 3: [-1,0]}
fig = plt.figure(1, figsize=(5, 2.5))
nx.draw(G, with_labels= True, pos=pos, node_size= 800)

#Setting attribute to account for the real epidemics in the graph
attrs = {0: {"inf_time": 1, "status": 'recovered'}, 1: {"inf_time": 2, "status": 'recovered'},
         2: {"inf_time": 2, "status": 'recovered'}, 3: {"inf_time": math.inf, "status": 'susceptible'}}
nx.set_node_attributes(G, attrs)

#Visualizing the graph with their respectives infection times
color_state_map = {'recovered': 'lime', 'susceptible': 'cyan'}
node_color = [color_state_map[node[1]['status']] for node in G.nodes(data=True)]

infection_times = nx.get_node_attributes(G, "inf_time")
state_pos = {n: (x + 0.15, y + 0.02) for n, (x,y) in pos.items()}

fig = plt.figure(1, figsize=(6, 3))
nx.draw_networkx(G, pos, node_size = 800, node_color = node_color,)

nx.draw_networkx_labels(G, state_pos, labels= infection_times, font_color='blue')
plt.show()

#Partial information available
#First we make a copy of G with the real infection times
G_real = copy.deepcopy(G)

#We exclude the information for nodes 1 and 3
nx.set_node_attributes(G, {1:{"inf_time": math.inf, "status": None}, 3: {"inf_time": math.inf, "status": None}})

#Visualizing the graph with only the partial information available
color_state_map = {'recovered': 'lime', 'susceptible': 'cyan', None: 'magenta'}
node_color = [color_state_map[node[1]['status']] for node in G.nodes(data=True)]

infection_times = nx.get_node_attributes(G, "inf_time")
state_pos = {n: (x + 0.15, y + 0.02) for n, (x,y) in pos.items()}

fig = plt.figure(1, figsize=(6, 3))
nx.draw_networkx(G, pos, node_size = 800, node_color = node_color,)

nx.draw_networkx_labels(G, state_pos, labels= infection_times, font_color='blue')
plt.show()

#Set an initial feasible tree for G given the observed values
T_initial, t_children = feasible_tree(G, [0,2], flag=1)
print(T_initial)
print(t_children)

#Copying the partial information to use different sample sizes
G_10000 = copy.deepcopy(G)
G_50000 = copy.deepcopy(G)

#Start the sampling algorithm
infected_nodes = [0,2]

print(f"Real infection times: {nx.get_node_attributes(G_real, "inf_time")}")
print("--------------------------------------------------------------------------------------------")

#Initialize class
sampler_1000 = TreeSampler(G, T_initial, t_children, infected_nodes)

#Run
print("Test with star graph example")
sampling1 = sampler_1000.run(n_iterations=1000)
print(f"Frequency of nodes for 1000 iterations: {nodes_proportion(G, sampling1)}")

#ITERATIONS = 10000
sampler_10000 = TreeSampler(G_10000, T_initial,t_children, infected_nodes)
sampling2 = sampler_1000.run(n_iterations=10000)
print(f"Frequency of nodes for 10000 iterations: {nodes_proportion(G_10000, sampling2)}")

#ITERATIONS = 50000
sampler_50000 = TreeSampler(G_50000, T_initial, t_children, infected_nodes)
sampling3 = sampler_50000.run(n_iterations=50000)
print(f"Frequency of nodes for 50000 iterations: {nodes_proportion(G_50000, sampling3)}")

#Saving the sampling obtained
sample_star_graph = []
sample_star_graph.append(sampling1)
sample_star_graph.append(sampling2)
sample_star_graph.append(sampling3)

#Saving the sampling obtained using pickle
file_path = 'data/samples/verification_test_star_graph'
pickle.dump(sample_star_graph, open(file_path + '.pickle', 'wb'))