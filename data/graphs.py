import networkx as nx
import pickle

#Generating all the graphs that will be used in my analysis
n = 10
p = 5/(n - 1)
G10_erdos_renyi = nx.erdos_renyi_graph(n, p)

# Saving graph objects to file
pickle.dump(G10_erdos_renyi, open('data/G10_erdos_renyi.pickle', 'wb'))
