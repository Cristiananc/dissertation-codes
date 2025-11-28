
def nodes_proportion(G, samplings):
    """
    Returns the proportion of each node in a sampling.
    """
    nodes_prop = {x: 0 for x in list(G.nodes)}
    for node in G.nodes:
        for lis in samplings:
            is_present = any(node in sublist for sublist in lis)
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
    return None