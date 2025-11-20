#This file contains the functions used to color the graphs according to the 
# transmission process

import networkx as nx
import matplotlib as plt

def coloring_graph_by_status(G):
    color_state_map = {'recovered': 'lime', 'susceptible': 'cyan'}
    node_color = [color_state_map[node[1]['status']] for node in G.nodes(data=True)]
    
    nx.draw(G,
        with_labels=True,
        node_color = node_color,
        node_size=800)

#Drawing graph with the infection times for each node 
# (inf means the node has never been infected)
def coloring_graph_inf_times(G):
    color_state_map = {'recovered': 'lime', 'susceptible': 'cyan'}
    node_color = [color_state_map[node[1]['status']] for node in G.nodes(data=True)]
    
    infection_times = nx.get_node_attributes(G, "inf_time")
    pos = nx.spring_layout(G, seed = 0)
    state_pos = {n: (x + 0.14, y + 0.05) for n, (x,y) in pos.items()}

    nx.draw_networkx(G,
                 pos,
                 node_size = 800,
                 node_color = node_color,)

    nx.draw_networkx_labels(G, state_pos, labels= infection_times, font_color='blue')
    plt.show()
    