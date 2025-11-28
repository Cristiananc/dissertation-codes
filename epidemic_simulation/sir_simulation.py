"""
This file contains the code to perform simulations for the Network SIR discrete-time model.
The code is based on the one provided by the book by (Kiss, Miller, Simon) on page 386.
Disclaimer: The code was refactored with AI assistance (more speciffically 
on suggestions of the heapq usage and enum addition).
"""
import networkx as nx
import numpy as np
import heapq
import math
from enum import Enum

class State(Enum):
    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    RECOVERED = "recovered"

class Action(Enum):
    TRANSMIT = 1
    RECOVER = 2

def fast_SIR(G: nx.Graph, initial_infecteds: list, p: float):
    """
    Simulates a discrete-time SIR outbreak on a static network using a priority queue. 
    We assume the infection lasts one time step and transmission occurs with probability p.

    Input:  
        G : a networkx graphs.
        initial_infecteds: list of node indices starting the infection. 
        p: transmission probability.
    
    Output:
    tuple: (times, S, I, R) lists tracking the history of the outbreak.
    """

    #Queue structure: (time, action_type, node)
    event_queue = [] 

    #Initialize graph attributes
    nx.set_node_attributes(G, "susceptible", "status")
    nx.set_node_attributes(G, math.inf, "inf_time")

    #Local tracker (reading dict is faster than G.nodes(v)[attr])
    status = {node : State.SUSCEPTIBLE.value for node in G.nodes()}
    inf_times = {node : math.inf for node in G.nodes()}

    for u in initial_infecteds:
        status[u] = State.INFECTED.value
        inf_times[u] = 1

        #Updating infection time for initial infected nodes
        G.nodes[u]['inf_time'] = 1

        #Schedule next transmission events
        heapq.heappush(event_queue, (1, Action.TRANSMIT.value, u))
    

    while event_queue:
        t, action, u = heapq.heappop(event_queue)

        if action == Action.RECOVER.value:
            status[u] = State.RECOVERED.value
            G.nodes[u]["status"] = State.RECOVERED.value

        elif action == Action.TRANSMIT.value:
            #Schedule recovery for node u at t+1
            heapq.heappush(event_queue, (t+1, Action.RECOVER.value, u))

            for v in G.neighbors(u):

                if status[v] == State.SUSCEPTIBLE.value:
                    new_inf_time = t + 1

                    if new_inf_time < inf_times[v]:
                        #Bernoulli trial
                        if np.random.uniform() < p:
                            #Update infection time (local and graph)
                            inf_times[v] = new_inf_time
                            G.nodes[v]['inf_time'] = new_inf_time

                            #Schedule transmission
                            heapq.heappush(event_queue, (new_inf_time, Action.TRANSMIT.value, v))

                            status[v] = State.INFECTED.value
                            G.nodes[v]['status'] = State.INFECTED.value
                
    return G
