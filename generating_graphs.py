import networkx as nx
import pickle

"""
In this file, we generate all the graphs that will be used in the analysis. 
We have selected the following models: random graphs, scale-free and small-world
(the three canonical network topologies). 
The sizes of the graphs are 100, 1000, 10000 and 100000 (stress test).
Additionally, for every size and model, we generate 10 instances to average the results later.
This is justified by the fact that these models are randomly generated and one graph of 
1000 nodes might behave significantly different from another.
"""

sizes = [100, 1000, 10000, 100000]

#First we generate the random graphs
for i in sizes:
    #For each size we generate 10 instances of the graph.
    for j in range(10):
        #The average degree <k> of a node is five.
        p = 5/(i - 1)
        G_erdos_renyi = nx.erdos_renyi_graph(i, p)

        # Saving graph objects to file
        pickle.dump(G_erdos_renyi, open('data/G' + str(j) + 'size' + str(i) + 'erdos_renyi.pickle', 'wb'))

