from functools import reduce 

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
    total_events = len(list(G.neighbors(0)))

    for lis in T[1:]:
        for node in lis[0:-1]:
            total_events += len(list(G.neighbors(node))) - 1
    
    failed_events = total_events - succes_events

    prob = beta**(succes_events)*(1 - beta)**failed_events

    return prob
