import math
from functools import reduce

def nodes_proportion(G, samplings):
    """
    Returns the proportion of each node in a sampling of feasible trees in the format
    of list of lists.
    """

    if samplings == None:
        return 
    
    nodes_prop = {x: 0 for x in list(G.nodes)}
    for node in G.nodes:
        for lis in samplings:
            is_present = any(node in sublist for sublist in lis)
            nodes_prop[node] += is_present

    n = len(samplings)
    
    for node in nodes_prop:
        nodes_prop[node] = nodes_prop[node]/ n
    return nodes_prop

def nodes_proportion_list(G, samplings):
    """
    Returns the proportion of each node in a sampling.
    """
    nodes_prop = {x: 0 for x in list(G.nodes)}
    for node in G.nodes:
        for lis in samplings:
            is_present = node in lis
            nodes_prop[node] += is_present

    n = len(samplings)
    
    for node in nodes_prop:
        nodes_prop[node] = nodes_prop[node]/n
    return nodes_prop

def check_feasibility_graphs(G, T, infected_nodes):
    """
    Checks if a given Tree embeeded in a graph is feasible and returns a boolean 
    accordingly.
    """
   
    for node in infected_nodes:

        if G.nodes[node]['inf_time'] == T.nodes[node]['inf_time']:
            continue
        else:
            return False
    
    return True

def check_feasibility_tree(G, T):
    """
    Checks if a given tree in list of lists format is feasible.
    """
    for sublist in T:
        if not G.nodes[sublist[0]]['inf_time'] - G.nodes[sublist[-1]]['inf_time'] == len(sublist) - 1:
            return False

    return True

def prob_tree_log(G, T, beta):
    """
    Returns the log probability of a transmission tree.
    """

    succes_events = reduce(lambda count, l: count + len(l) - 1, T, 0)
    total_events = G.degree[0]

    for lis in T[1:]:
        for node in lis[0:-1]:
            total_events += G.degree[node] - 1
    
    failed_events = total_events - succes_events

    prob_log = math.log(beta**(succes_events)*(1 - beta)**failed_events)

    return prob_log


def prob_path_log(G, path):
    """
    This is based on the Degree-Biased Random Walk
    Returns the log probability of a path according to the degree of the nodes.
    It should return Inf if a path does not exist.
    Disclaimer: This code was refactored with AI assistance.
    """
    log_prob = 0

    if len(path) < 2:
        return log_prob
  
    #Iterating using a slide window
    for i in range(len(path) - 1):
      u = path[i]
      v = path[i + 1]

    #Check if nodes exist in the graph:
    if u not in G or v not in G:
      return math.inf

    #Check if edge exists in the graph
    if u not in G[v]:
      return math.inf
      
    neighbors = list(G.neighbors(u))
    norm_factor = sum([G.degree[i] for i in neighbors]) 

    if norm_factor == 0:
      return math.inf 

    numerator = G.degree(v)
    log_prob += math.log(numerator) - math.log(norm_factor) 
      
    return log_prob
