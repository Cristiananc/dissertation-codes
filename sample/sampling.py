
import copy
import math
import random as rd
from feasible_tress import find_k_length_path

def sampling_trees(G,T_initial,n, infected_nodes):
  sampling = [T_initial]

  for i in range(n):
    T_current = sampling[i]
    _
    random_node_aux = rd.randrange(1, len(infected_nodes)) #Excludes the node 0, since it's the root of the tree
    random_node = infected_nodes[random_node_aux]

    #Delete the previous path from G
    #Recall that all nodes in T_current[random_node_aux][1:-1] are unobserved nodes
    for node in T_current[random_node_aux][1:-1]:
      G.nodes[node]['inf_time'] = math.inf

    #Find a new path for
    new_path = find_k_length_path(G, random_node, 0, G.nodes[random_node]['inf_time'])

    # Modify the current state
    T_current[random_node_aux] = new_path

    # Append a unique copy of the current state to the sampling list
    sampling.append(copy.deepcopy(T_current))

  return sampling

def nodes_proportion(G, samplings, samplings_number):
    nodes_prop = {x: 0 for x in list(G.nodes)}
    for node in G.nodes:
        for lis in samplings:
            is_present = any(node in sublist for sublist in lis[1:])
            nodes_prop[node] += is_present

    for node in nodes_prop:
        nodes_prop[node] = nodes_prop[node]/(samplings_number + 1)

    return nodes_prop