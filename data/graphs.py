import networkx as nx
import pickle

"""
In this file, we generate all the graphs that will be used in my analysis. 
I have selected the following models: random graphs, scale-free and small-world. 
The sizes of the graphs are 100, 1000, 10000 and 100000.
Additionally, for every size and model, we generate 10 instances and average the results.
This is justified by the fact that these models are randomly generated and one graph of 
1000 might behave different from another.
"""
n = 10
p = 5/(n - 1)
G10_erdos_renyi = nx.erdos_renyi_graph(n, p)

# Saving graph objects to file
pickle.dump(G10_erdos_renyi, open('data/G10_erdos_renyi.pickle', 'wb'))
