# This file contains the code to perform a simulation for the Network SIR in discrete time
#The code is based on the one provided by the book by (Kiss, Miller, Simon) on page 386

import numpy as np
import networkx as nx
import math

"""
The following algorithm calculates the events occuring in a outbreak 
modeled as a discrete-time SIR in a static network. Here, we assume the 
infection lasts one time step and transmission occurs with probability p.
"""

#INPUT: A graph G created with networkx, a list called initial_infecteds 
# which represents the index-cases, a decimal p which represents the infection rate
# and an integer n, that is the population size.
#OUTPUT: It returns a list with the times, an another three lists for the number of
# individuals in each compartment for each time. 
def fast_SIR(G, initial_infecteds, p, n):
    #S,I and R keep track of number of nodes in each state
    S = [n]
    I = [0]
    R = [0]
    Q = [] #Empty priority queue

    times = [0]

    nx.set_node_attributes(G, "susceptible", "status")

    nx.set_node_attributes(G, math.inf, "inf_time")

    for u in initial_infecteds:
        event = {'node': u, 'time': 1, 'action': 'transmit'}
        Q.append(event)

    #Updating infection time for initial infected nodes
    G.nodes[u]['inf_time'] = 1

    #While the list of events is not empty we continue the transmission process
    while Q:
        Q = sorted(Q, key=lambda d: d['time'])
        event = Q[0]  #earliest remaining event in Q


        if event['action'] == 'transmit':
            process_trans_SIR(G, event['node'], S, I, R, times, event['time'], p, Q)

        else:
            process_rec_SIR(G, event['node'], event['time'], times, S, I, R)

        #Once we are acessing this event we remove it from priority queue
        #of course we need to check what is the best data structure to do this
        Q.pop(0)


    return times, S, I, R

def process_rec_SIR(G, u, t, times, S, I, R):

    #updating the number of nodes in each compartment
    S.append(S[-1])
    R.append(R[-1] + 1)
    I.append(I[-1] - 1)

    times.append(t)

    G.nodes[u]["status"] = 'recovered'

def process_trans_SIR(G, u, S, I, R, times, t, p, Q):
    #updating the number of nodes in each compartment
    S.append(S[-1] - 1)
    R.append(R[-1])
    I.append(I[-1] + 1)

    times.append(t)

    G.nodes[u]["status"] = 'infected'

    #Creating an event of recovery
    newEvent = {'node': u, 'time': t + 1, 'action': 'recover'}
    Q.append(newEvent)

    for v in G.neighbors(u):

        if G.nodes[v]["status"] == 'susceptible':
            inf_time = t + 1

            if inf_time < G.nodes[v]['inf_time']:
                aux = np.random.uniform()

                if aux < p:
                    newEvent = {'node': v, 'time': inf_time, 'action': 'transmit'}
                    Q.append(newEvent)
                    G.nodes[v]['inf_time'] = inf_time
