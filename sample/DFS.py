"""
 This file contains the different versions of the DFS for sampling the feasible trees.  
"""

import math

# INPUT:
# A flag that states which version of the DFS modified search we would like to use. 
# OUTPUT:
def DFS(G, v, t,k, path, visited, flag):
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
        if k -  G.nodes[v]['inf_time'] == len(path) - 1:
            return path
        else:
            return None
    
    #We have found an observed node that does not relate to the current path size
    if G.nodes[v]['inf_time'] != math.inf and (k - G.nodes[v]['inf_time'] != len(path) - 1):
        return None
    
    #Each unobserved node we pass in the path receives a infection time
    G.nodes[v]['inf_time'] = k - len(path) + 1
    visited[v] = True
    
    for neighbor in G.neighbors(v):
        if not visited[neighbor]:
            result = DFS(G, neighbor, t, k, path + [neighbor], visited, flag)
            #If a desired path is returned we return it
            if result:
                return result
    
    #The two lines above deals with the process of backtracking in the search
    visited[v] = False
    G.nodes[v]['inf_time'] = math.inf