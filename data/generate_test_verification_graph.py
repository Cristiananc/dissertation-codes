import pickle 
import networkx as nx

file_path = 'data/graphs/verification_test_graph'

#Creating the verification graph
n = 10
p = 5/(n - 1)
G = nx.erdos_renyi_graph(n, p)

pickle.dump(G, open(file_path + '.pickle', 'wb'))