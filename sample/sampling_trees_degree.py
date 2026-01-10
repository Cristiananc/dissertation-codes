"""
Disclaimer: Code refactored with AI assistance, more specifically on code modularization and 
verification of edge cases. All final implementation logic, and resulting analysis remain the 
original work and responsability of the author.
"""

import math
import random as rd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm 
from .search_on_graphs import find_k_length_path
from epidemic_simulation.sir_simulation import fast_SIR
from .helpers import check_feasibility_graphs

#Set output prints to a txt file
f = open("output.txt", "a")

class TreeSampler:
    """
    MCMC Sampler for inferring infected nodes on a network with partial 
    information available.
    """
    
    def __init__(self, G, T_initial, children_of, infected_nodes):
        self.G = G
        self.T_current = copy.deepcopy(T_initial)
        self.children_of_curr = copy.deepcopy(children_of)
        self.infected_nodes = list(infected_nodes) 

        self.nodes_to_sample = list(infected_nodes)
        self.unobserved_leaves = []
        self.samplings_trees = [copy.deepcopy(T_initial)]
        self.trees_degre = {}

        # Adding intermediate nodes to our list of possible nodes to sample
        for key,val in T_initial.items():
            if key not in self.infected_nodes and key not in self.nodes_to_sample:
                    self.nodes_to_sample.append(key)

        sum_of_edges = 0
        for i in self.G.degree():
            sum_of_edges += i[1]
        self.avg_degree = math.ceil((sum_of_edges / len(self.G.nodes)) - .5)
                
    def run(self, n_iterations):
        """
        This function executes the Metropolis-Hastings sampling loop.

        Args:
            n_iterations (int): The number of iterations of the loop.

        Returns:
            self.samplings_trees (list): List of states sampled.
        """

        if self.T_current == None: return None
        
        accepted_count = 0
        self.beta = 0.3 #Initial value for Beta
        
        for _ in tqdm(range(n_iterations), desc="Sampling trees"):
            if not self.nodes_to_sample: break 
            
            # Capture full state for reversion
            previous_children_of = copy.deepcopy(self.children_of_curr)
            previous_T = copy.deepcopy(self.T_current)
            previous_G = copy.deepcopy(self.G)
            previous_nodes_list = list(self.nodes_to_sample)
            previous_leaves_list = list(self.unobserved_leaves)
        
            valid_proposal, q_ratio = self._propose_next_state()

            current_ll = self._prob_tree_log(previous_G, previous_T, self.beta)

            if valid_proposal:
                alpha = self._compute_acceptance_prob(q_ratio, self.beta, previous_G, previous_T)
                p_uniform = math.log(np.random.uniform())

                if p_uniform < alpha:
                    #ACCEPT
                    accepted_count += 1
                    self.samplings_trees.append(copy.deepcopy(self.T_current))

                else:
                    #REJECT
                    self.G = previous_G
                    self.T_current = previous_T
                    self.nodes_to_sample = previous_nodes_list
                    self.unobserved_leaves = previous_leaves_list
                    self.children_of_curr = previous_children_of

                    self.samplings_trees.append(copy.deepcopy(self.T_current))

            else:
                self.G = previous_G
                self.T_current = previous_T
                self.nodes_to_sample = previous_nodes_list
                self.unobserved_leaves = previous_leaves_list
                self.children_of_curr = previous_children_of
                
                self.samplings_trees.append(copy.deepcopy(self.T_current))
            
            #"""
            print(file=f)
            print(f"Current Tree: {self.T_current}", file=f)
            print(file=f)
            print(f"Nodes to sample: {self.nodes_to_sample}",file=f)
            print(file=f)
            print(f"Unobserved leaves: {self.unobserved_leaves}",file=f)
            print(file=f)
            print(f"Current infection times: {nx.get_node_attributes(self.G, "inf_time")}",file=f)
            print(file=f)
            print(f"Children: {self.children_of_curr}", file=f)
            #"""

        print()
        print(f"Final Acceptance Rate: {accepted_count / n_iterations:.2%}")
        return self.samplings_trees

    # --------- Helper methods ------------ #

    def _propose_next_state(self):
        """
        Handles the logic for proposing a move.

        Returns:
            (bool, float) (tuple) -> (valid_proposal, q_ratio).
        """
        
        q_ratio = 0 # Default (log(1) = 0)
        valid_proposal = True

        #Calculating the degree of T_curr in the state space graph
        curr_degree_approx = self._calculate_degree_curr_tree()
        rd_idx = rd.randrange(1, curr_degree_approx + 1)

        print(f"curr_degree: {curr_degree_approx}", file = f)
        print(f"rd_idx: {rd_idx}", file = f)

        if rd_idx <= len(self.unobserved_leaves):
            #Deletion 
            node_delete = self._choose_random_node(self.unobserved_leaves)
            self._delete_node(node_delete)

        elif rd_idx > len(self.unobserved_leaves) and rd_idx <= len(self.unobserved_leaves) + (len(self.infected_nodes) - 1)*self.avg_degree:
            node_change_path = self._choose_random_node(self.infected_nodes[1:])
            self._change_path(node_change_path)           
            
        else:
            #Addition
            node_addition = self._choose_random_node(self.nodes_to_sample)
            len_neigh = self._add_neighbor(node_addition)

            if len_neigh is None or len_neigh == 0:
                valid_proposal = False

        #Calculating the degree of T_prop in the state space graph
        prop_degree_approx = self._calculate_degree_curr_tree()
        q_ratio = math.log(curr_degree_approx / prop_degree_approx)
        print(f"q_ratio: {q_ratio}", file = f)

        return valid_proposal, q_ratio

    def _calculate_degree_curr_tree(self):
        curr_degree_approx = len(self.unobserved_leaves) + (len(self.infected_nodes) - 1)*self.avg_degree  + len(self.nodes_to_sample)*self.avg_degree

        return curr_degree_approx

    def _choose_random_node(self, list_of_nodes):
        """
        Chooses a random element uniformly from a list.

        Args:
            list_of_nodes (list): List with nodes (integers).

        Returns:
            random_node (int): Random element from list_of_nodes.
        """

        rand_idx = rd.randrange(0, len(list_of_nodes))
        random_node = list_of_nodes[rand_idx]

        return random_node

    def _calculate_new_path(self,source_node):
        """
        Finds a path of fixed length from a source node to a target node.

        Args:
            source_node (int): Source node for the s-t k-path search.

        Returns:
            new_path (list): s-t k-path found.
        """

        new_path = find_k_length_path(
            self.G, source_node, 0, self.G.nodes[source_node]['inf_time'],flag=1)

        return new_path

    # ---------- Operations function --------------- #
    def _change_path(self, target_node):
        """
        This function performs a change of path operation.

        Args:
            target_node (int): The node for which the path in the current tree will be changed.
        """
        print("Changing path",file=f)        

        old_parent = self.T_current[target_node]
        new_path = self._calculate_new_path(target_node)

        if new_path is not None:
            #Adding the new path
            for i in range(len(new_path) - 1):
                self.T_current[new_path[i]] = new_path[i + 1]

            for j in range(len(new_path) - 1, 0, -1):
                if new_path[j] not in self.children_of_curr:
                    self.children_of_curr[new_path[j]] = [new_path[j - 1]]
                else:
                    self.children_of_curr[new_path[j]].append(new_path[j-1])
            
            for n in new_path:
                #If a leaf becomes part of a path for the new_node, it is no longer a leaf
                if n in self.unobserved_leaves:
                    self.unobserved_leaves.remove(n)
            
                if n not in self.nodes_to_sample:
                    self.nodes_to_sample.append(n)

            self.children_of_curr[old_parent].remove(target_node)

            if len(self.children_of_curr[old_parent]) == 0:
                self.unobserved_leaves.append(old_parent)

    def _add_neighbor(self, node):
        """
        Performs an addition of a node operation.

        Args:
            node (int): The parent of the new node that will be added.

        Returns:
            len_neighb_available (int): The size of the list of available neighbors of the 
            parent node that can be added in the tree.
        """

        neighbors = list(self.G.neighbors(node))
        neighb_available = [node for node in neighbors if self.G.nodes[node]['inf_time'] == math.inf]        
        
        if not neighb_available: return None

        new_node =  neighb_available[rd.randrange(0, len( neighb_available))]

        print(f"New node added: {new_node}",file=f)
        self.G.nodes[new_node]['inf_time'] = self.G.nodes[node]['inf_time'] + 1

        #Update state
        self.T_current[new_node] = node
        
        if node in self.children_of_curr: self.children_of_curr[node].append(new_node)
        else: self.children_of_curr[node] = [new_node]

        self.nodes_to_sample.append(new_node)
        self.unobserved_leaves.append(new_node)

        #If the source was a leaf, it is not anymore
        if node in self.unobserved_leaves:
            self.unobserved_leaves.remove(node)

        len_neighb_available = len(neighb_available)

        return len_neighb_available

    def _delete_node(self, node):
        """
        Performs a deletion of a node operation.

        Args:
            node (int): The node which will be deleted from the current feasible tree.

        Returns:
            parent_node (int or None): The parent of the deleted node in the current feasible tree.
        """

        parent_node = self.T_current[node]

        #Reset infection time for deleted node
        self.G.nodes[node]['inf_time'] = math.inf
        print({f"Node deleted: {node}"},file=f)

        if node in self.nodes_to_sample:
            self.nodes_to_sample.remove(node)
        
        if node in self.unobserved_leaves:
            self.unobserved_leaves.remove(node)

        del self.T_current[node]
        self.children_of_curr[parent_node].remove(node)
        
        #Handling the case where the parent becomes a leaf
        if len(self.children_of_curr[parent_node]) == 0:
            if parent_node not in self.unobserved_leaves:
                if parent_node not in self.infected_nodes:
                    self.unobserved_leaves.append(parent_node)

        return parent_node

    # ---------- Compute probabilities -------------------#
    def _prob_tree_log(self, G, T, beta):
        """
        This function calculates the log-likelihood of a tree for a given infection rate.

        Args:
            G (networkx graph): Graph for which we will calculate the failed infection events.
            T (list): The transmission tree for which we will compute the log-likelihood.
            beta (float): Infection rate.

        Returns:
            prob_log (float): The log-likelihood for T for a fixed beta.
        """

        #Identify all nodes in the tree
        nodes_in_tree = set()
        nodes_in_tree.add(0)
        for node,value in T.items():
            nodes_in_tree.add(node)

        if len(nodes_in_tree) <= 1:
            return 0
        
        # Succes Events (V_T - 1)
        succes_events = len(nodes_in_tree) - 1
        failed_events = 0

        for u in nodes_in_tree:
            for v in G.neighbors(u):
                if v not in nodes_in_tree:
                    failed_events += 1
                else:
                    if G.nodes[v]['inf_time'] > G.nodes[u]['inf_time'] and T[v] != u:
                        failed_events +=1
                
        
        log_beta = math.log(beta)
        log_beta_aux = math.log(1-beta)
        prob_log = succes_events * log_beta + failed_events * log_beta_aux
        
        return prob_log
    
    def _compute_acceptance_prob(self, q_ratio, beta, previous_G, previous_T):
        """
        Calculates alpha using the log-likelihoods

        Args:
            q_ratio (float): Proposal distribution ratio.
            beta (float): Infection rate.
            previous_G (networkx graph): Graph G before proposal.
            previos_T (list): Feasible tree before proposal.

        Returns:
            alpha (float): Acceptance probability for proposal.
        """

        prob_tree_prop = self._prob_tree_log(self.G, self.T_current, beta)
        prob_tree_curr = self._prob_tree_log(previous_G, previous_T, beta)

        # Alpha = (Log P_new - Log P_old) + Log Q_ratio
        alpha_aux = prob_tree_prop - prob_tree_curr + q_ratio

        alpha = min(0, alpha_aux) 
        return alpha
    
    # ---------------- Visualise results ------------------------- #
    def _trace_plot_log_likelihood(self):
        """
        Plots the trace plot for the log likelihood of the samplings.
        """
        if len(self.log_likelihood_history) <= 1:
            print("No valid sampling has been performed yet!")
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(self.log_likelihood_history)
            plt.xlabel("Iteration")
            plt.ylabel("Log-Likelihood")
            plt.show()
    
    # ------------- Performs Naive Sampling ---------------- #
    def naive_sampling(G, sampling_number, observed_nodes, initial_infecteds):
        samplings = []
        G_mutable = copy.deepcopy(G)

        while len(samplings) < sampling_number:

            p = np.random.uniform()

            #In place modification of G
            fast_SIR(G_mutable, initial_infecteds, p)

            if check_feasibility_graphs(G, G_mutable, observed_nodes):
                all_nodes = nx.get_node_attributes(G_mutable, "inf_time")
                nodes_infected = [node for node, inf_time in all_nodes.items() if inf_time < math.inf]

                samplings.append(nodes_infected)

        return samplings