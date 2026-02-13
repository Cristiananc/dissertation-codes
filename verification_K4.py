import networkx as nx
import copy
import matplotlib.pyplot as plt
import math

#Importing from the function I created
from sample.search_on_graphs import *
from sample.sampling_trees_degree_change_path_addition import TreeSampler
from sample.helpers import nodes_proportion

#Complete graph K_4
e = [(0,1), (1,2), (2,3), (3,0), (3,1), (2,0)] #list of edges
G = nx.Graph(e)

pos = {0: [0,0], 1: [0,1], 2: [1,1], 3: [1,0]}
fig = plt.figure(1, figsize=(3, 2.5))
nx.draw(G, with_labels= True, pos=pos, node_size= 800)

#The real epidemics
attrs = {0: {"inf_time": 1, "status": 'recovered'}, 1: {"inf_time": 2, "status": 'recovered'},
         2: {"inf_time": 2, "status": 'recovered'}, 3: {"inf_time": 3, "status": 'recovered'}}
nx.set_node_attributes(G, attrs)

#Visualizing the graph
color_state_map = {'recovered': 'lime', 'susceptible': 'cyan'}
node_color = [color_state_map[node[1]['status']] for node in G.nodes(data=True)]

infection_times = nx.get_node_attributes(G, "inf_time")
state_pos = {n: (x + 0.08, y + 0.02) for n, (x,y) in pos.items()}

fig = plt.figure(1, figsize=(6, 3))
nx.draw_networkx(G, pos, node_size = 800, node_color = node_color,)

nx.draw_networkx_labels(G, state_pos, labels= infection_times, font_color='blue')
plt.show()

#First we make a copy of G with the real infection times
G_real = copy.deepcopy(G)

#We exclude the information for nodes 1 and 2
nx.set_node_attributes(G, {1:{"inf_time": math.inf, "status": None}, 2: {"inf_time": math.inf, "status": None}})

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
T_initial, t_children = feasible_tree(G, [0,3], flag=1)
print(T_initial)
print(t_children)

#Copying the partial information to use different sample sizes
G_10000 = copy.deepcopy(G)
G_100000 = copy.deepcopy(G)

#Start the sampling algorithm
infected_nodes = [0,3]

print(f"Real infection times: {nx.get_node_attributes(G_real, "inf_time")}")
print("--------------------------------------------------------------------------------------------")

sampler_1000 = TreeSampler(G, T_initial, t_children, infected_nodes)
sampling = sampler_1000.run(n_iterations=1000)
print(f"Frequency of nodes: {nodes_proportion(G, sampling[500:])}")

sampler_10000 = TreeSampler(G_10000, T_initial,t_children, infected_nodes)
sampling = sampler_1000.run(n_iterations=10000)
print(f"Frequency of nodes: {nodes_proportion(G_10000, sampling[500:])}")

#sampler_100000 = TreeSampler(G_100000, T_initial, t_children, infected_nodes)
#sampling = sampler_100000.run(n_iterations=100000)
#print(f"Frequency of nodes: {nodes_proportion(G_100000, sampling)}")