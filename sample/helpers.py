import networkx as nx
from functools import reduce 
#from operator import mul
import math

def nodes_proportion(G, samplings, samplings_number):
    """
    Returns the proportion of each node in a sampling.
    """
    nodes_prop = {x: 0 for x in list(G.nodes)}
    for node in G.nodes:
        for lis in samplings:
            is_present = any(node in sublist for sublist in lis)
            nodes_prop[node] += is_present

    for node in nodes_prop:
        nodes_prop[node] = nodes_prop[node]/(samplings_number + 1)

    return nodes_prop

def prob_tree(G, T, beta):
    """
    Returns the probability of a transmission tree.
    """

    succes_events =reduce(lambda count, l: count + len(l) - 1, T, 0)
    total_events = len(list(G.degree[0]))

    for lis in T[1:]:
        for node in lis[0:-1]:
            total_events += G.degree[node] - 1
    
    failed_events = total_events - succes_events

    prob = beta**(succes_events)*(1 - beta)**failed_events

    return prob


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
