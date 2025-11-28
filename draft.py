# Importing the required modules and packages
import networkx as nx
from epidemic_simulation.sir_simulation import fast_SIR
from epidemic_simulation.helpers import *
from sample.helpers import *
from sample.sampling_trees import *
from sample.naive_sampling import naive_sampling
#import scipy as sp

#Creating our initial example of a random network
n = 10
p = 5/(n - 1)
G = nx.erdos_renyi_graph(n, p)

beta = 0.4
fast_SIR(G, [0], beta)

infection_times = nx.get_node_attributes(G, "inf_time")
print(infection_times)