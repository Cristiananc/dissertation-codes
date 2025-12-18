"""
In this file, we load the generated graphs and perform a Network SIR discrete-time epidemics.
From the set of nodes, we randomly select nodes for which their infection time information 
will be deleted. With that we generate a graph with the obseverd partial information.
Later these files with the true course of the epidemic will be compared to the truth.
"""
import pickle
from enum import Enum

class Model(Enum):
    RANDOM = 'erdos_renyi'
    SCALE_FREE = 'scale_free'
    SMALL_WORLD = 'small_world'

sizes = [20, 50, 100]

for i in sizes:
    file = open("file_name",'r')
    object_file = pickle.load(file)
