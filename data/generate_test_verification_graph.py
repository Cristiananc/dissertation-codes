import pickle 
import networkx as nx

file_path = 'data/graphs/verification_test_graph_20_nodes'

#Creating the verification graph
n = 20
p = 5/(n - 1)
G = nx.erdos_renyi_graph(n, p)

pickle.dump(G, open(file_path + '.pickle', 'wb'))