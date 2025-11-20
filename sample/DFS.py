"""
 This file contains the modified DFS algorithm for sampling the feasible trees.  
"""

import math
import random as rd
import numpy as np

# INPUT:
# A flag that states which version of the DFS modified we would like to use. 
# OUTPUT:
def DFS(G, v, t,k, path, visited, flag=0):
    #Code refactored with AI assistance.
    
    # 1. Base Case: Sucess
    #If the current node is the target node and the length of the path - 1 is k, we have found
    #a desired path 
    if v == t and len(path) - 1 == k:
        return path
    
    # 2. Base Case: Path too long
    if len(path) - 1 > k:
        return None
    
    # 3. Base Case: Target reached too early 
    #If the target node is reached before the length of the desired path, then no path is returned
    if len(path) - 1 < k + 1 and v == t:
        return None
    
    # 4. Observed Node Check (Existing "Infection Time" Logic)
    # If we hit a node that was already visited in a previous valid context
    # and the timing matches, we consider this path valid
    if G.nodes[v]['inf_time'] != math.inf:
        #If timings match, return path
        if k -  G.nodes[v]['inf_time'] == len(path) - 1 and len(path) > 1:
            return path
        else:
            return None
    
    # 5. Set Infection Time
    # Only set this if it hasn't been set (implied by passing step 4)
    # Each unobserved node we pass in the path receives a infection time.
    if G.nodes[v]['inf_time'] == math.inf:
        G.nodes[v]['inf_time'] = k - len(path) + 1
        reset_inf_time = True #A flag to remember we need to reset it later
    
    else:
        reset_inf_time = False # We didn't change it
    
    visited[v] = True

    # 6. Neighbor Handling
    neighbors  = list(G.neighbors(v))

    # Check in case the node doesn't have neighbors:
    if not neighbors:
        visited[v] = False

        #In case we changed the inf time, reset it back to infinite
        if reset_inf_time:
            G.nodes[v]['inf_time']
        
        return None

    # 7. The flag determines in which ordering the neighbors will be consider
    # The default ordering is 0.
    if flag == 1:
        rd.shuffle(neighbors) #rd.shuffle is an in-place function

    # We shuffle the list, considering the nodes with higher degree
    # may appear first in the list. 
    elif flag == 2:
        neighbors_degree = []
        
        for node in neighbors:
            neighbors_degree.append(G.degree(node))

        norm = sum(neighbors_degree)
        probs = [d/norm for d in neighbors_degree]
        shuffled_list = np.random.choice(neighbors, len(neighbors), replace = False, p=probs)
        neighbors = shuffled_list.tolist()

    # 7. Recursive Step
    for neighbor in neighbors:
        if not visited[neighbor]:
            result = DFS(G, neighbor, t, k, path + [neighbor], visited, flag)
            #If a desired path is returned we return it
            if result:
                return result
    
    # 8. Backtracking
    #The lines below deals with the process of backtracking in the search
    visited[v] = False
    # Only reset inf_time if We changed it in this specific call
    if reset_inf_time:
        G.nodes[v]['inf_time'] = math.inf

    return None
