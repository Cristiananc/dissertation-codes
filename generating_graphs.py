"""
In this file, we generate all the graphs that will be used in the analysis. 
We have selected the following models: random graphs, scale-free and small-world
(the three canonical network topologies). 
The sizes of the graphs are 20, 50 and 100.
Additionally, for every size and model, we generate 5 instances of the graph.
This is justified by the fact that these models are randomly generated and one graph of 
100 nodes might behave significantly different from another.
"""

import networkx as nx
import pickle
from enum import Enum

class Model(Enum):
    RANDOM = 'erdos_renyi'
    SCALE_FREE = 'scale_free'
    SMALL_WORLD = 'small_world'

sizes = [20, 50, 100]
instances = 5

def generate_graphs(sizes, model, instances):

    for size in sizes:

        file_path = 'data/graphs/' + str(size) + '/G_' + model 

        #For each size we generate 10 instances of the graph.
        for j in range(instances):

            if model == Model.RANDOM.value:
                #The average degree <k> of a node is five.
                p = 5/(size - 1)
                
                G = nx.erdos_renyi_graph(size, p)

            elif model == Model.SCALE_FREE.value:
                G = nx.scale_free_graph(size)

            elif model == Model.SMALL_WORLD.value:
                G = nx.watts_strogatz_graph(size)

            # Saving graph objects to file
            pickle.dump(G, open(file_path + str(j) + "_" + str(size) + '.pickle', 'wb'))


#First we generate the random graphs
generate_graphs(sizes, Model.RANDOM.value, instances)

#Generating small-world graphs

#Generating scale-free graphs
