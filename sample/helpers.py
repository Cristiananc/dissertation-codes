
def nodes_proportion(G, samplings, samplings_number):
    nodes_prop = {x: 0 for x in list(G.nodes)}
    for node in G.nodes:
        for lis in samplings:
            is_present = any(node in sublist for sublist in lis)
            nodes_prop[node] += is_present

    for node in nodes_prop:
        nodes_prop[node] = nodes_prop[node]/(samplings_number + 1)

    return nodes_prop