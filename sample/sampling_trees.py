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
    
    def __init__(self, G, T_initial, children_of, infected_nodes, seed=None):
        self.G = G
        self.T_current = copy.deepcopy(T_initial)
        self.children_of_curr = copy.deepcopy(children_of)
        self.infected_nodes = list(infected_nodes) 

        self.nodes_to_sample = list(infected_nodes)
        self.unobserved_leaves = []
        self.samplings_trees = [copy.deepcopy(T_initial)]
        self.log_likelihood_history = []

        if seed is not None:
          import random as rd
          rd.seed(seed)
          self.seed = seed

        # Initialization logic
        # Adding intermediate nodes to our list of possible nodes to sample
        for key,val in T_initial.items():
            if key not in self.infected_nodes and key not in self.nodes_to_sample:
                    self.nodes_to_sample.append(key)
        
        self.nodes_to_sample.append(0)
        
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
            
            #self._update_beta_gibbs()
            beta = self.beta

            # Capture full state for reversion
            previous_children_of = copy.deepcopy(self.children_of_curr)
            previous_T = copy.deepcopy(self.T_current)
            previous_G = copy.deepcopy(self.G)
            previous_nodes_list = list(self.nodes_to_sample)
            previous_leaves_list = list(self.unobserved_leaves)
        
            valid_proposal, q_ratio = self._propose_next_state()

            current_ll = self._prob_tree_log(previous_G, previous_T, beta)

            if valid_proposal:
                alpha = self._compute_acceptance_prob(q_ratio, beta, previous_G, previous_T)
                p_uniform = math.log(np.random.uniform())

                if p_uniform < alpha:
                    #ACCEPT
                    accepted_count += 1
                    self.samplings_trees.append(copy.deepcopy(self.T_current))
                    self.log_likelihood_history.append(self._prob_tree_log(self.G, self.T_current, beta))

                else:
                    #REJECT
                    self._revert_state(previous_G, previous_T, previous_nodes_list, previous_leaves_list, previous_children_of)
                    self.samplings_trees.append(copy.deepcopy(self.T_current))
                    self.log_likelihood_history.append(current_ll)

            else:
                self._revert_state(previous_G, previous_T, previous_nodes_list, previous_leaves_list, previous_children_of)
                self.samplings_trees.append(copy.deepcopy(self.T_current))
                self.log_likelihood_history.append(current_ll)
            
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
        
        p = np.random.uniform()
        q_ratio = 0 # Default (log(1) = 0)
        valid_proposal = True

        if p < 1/3:
            #Addition
            node_addition = self._choose_random_node(self.nodes_to_sample)
            len_neigh = self._add_neighbor(node_addition)

            if len_neigh is not None and len_neigh > 0:
                # Foward: 1/N_sample_old * 1/len_neigh | Reverse: 1/Leaves_new
                n_sample = len(self.nodes_to_sample) - 1 #The list already includes the new node added
                n_leaves_new = len(self.unobserved_leaves)

                if n_leaves_new > 0:
                    q_ratio = math.log((n_sample *len_neigh)) - math.log(n_leaves_new)

                else:
                    valid_proposal = False
            
            else:
                valid_proposal = False

        elif p < 2/3:
            #Changing path
            node_change_path = self._choose_random_node(self.infected_nodes)
            self._change_path(node_change_path)

            #We assume the number of available paths doesn't change much between states
            q_ratio = 0 #Assumed symmetric

        else:
            #Deletion 
            if len(self.unobserved_leaves) > 0:
                node_delete = self._choose_random_node(self.unobserved_leaves)
                parent_node = self._delete_node(node_delete)
                valid_proposal, q_ratio = self._compute_q_ratio_deletion(parent_node)
            
            else:
                valid_proposal = False

        return valid_proposal, q_ratio

    def _revert_state(self, prev_G, prev_T, prev_nodes, prev_leaves, previous_children_of):
        """
        Clean helper to revert all state components.
        """
        self.G = copy.deepcopy(prev_G)
        self.T_current = copy.deepcopy(prev_T)
        self.nodes_to_sample = list(prev_nodes)
        self.unobserved_leaves = list(prev_leaves)
        self.children_of_curr = copy.deepcopy(previous_children_of)

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

    def _find_path_intermediate(self, source_node, path):

        if source_node not in self.infected_nodes:
            path.append(self.T_current[source_node])
            self._find_path_intermediate(self.T_current[source_node], path)
        else:
            return path
    
        return path
    
    # ---------- Operations function --------------- #
    def _change_path(self, target_node):
        """
        This function performs a change of path operation.

        Args:
            target_node (int): The node for which the path in the current tree will be changed.
        """
        print("Changing path",file=f)        

        old_parent = self.T_current[target_node]
        #old_path = self._find_path_intermediate(target_node)

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

        #""" 
        print(f"Succes events: {succes_events}",file=f)
        print(f"Failed events: {failed_events}",file=f)
        print(f"prob_log: {prob_log}",file=f)
        #"""

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
    
    def _compute_q_ratio_deletion(self, parent_node):
        """
        Calculates the Hastings ratio for deletion

        Args:
            parent_node (int): The parent node of the deleted node.

        Returns:
            (bool, float) -> (valid_proposal, q_ratio)
        """

        valid_proposal = False
        q_ratio = 0

        if parent_node is not None:

            neigh_available = 0

            # Recalculate available neighbors for the reverse move (Addition)
            for neighbor in self.G.neighbors(parent_node):

                if self.G.nodes[neighbor]['inf_time'] == math.inf:
                    neigh_available += 1

            n_sample_new = len(self.nodes_to_sample)
            
            # Check for the Math Domain Error condition (Denominator must be non-zero)
            if n_sample_new == 0 or neigh_available == 0:
                valid_proposal = False

            else:
                # q(T|T') / q(T'|T) = N_leaves_old / (N_sample_new * neigh_available)
                n_leaves_old = len(self.unobserved_leaves) + 1
                q_ratio = math.log(n_leaves_old) - math.log(n_sample_new) - math.log(neigh_available)
                valid_proposal = True

        else:
            # Deletion failed
            valid_proposal = False     

        return valid_proposal, q_ratio

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

    def _trace_plot_beta(self):
        """
        Plots the trace plot for the beta variable.
        """        
        if len(self.beta_history) <= 1:
            print("No valid sampling has been performed yet!")
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(self.beta_history)
            plt.xlabel("Iteration")
            plt.ylabel(r"$\beta$")
            plt.show()

    def _size_of_the_tree(self, T):
        """
        Calculates the size of a tree.

        Args:
            T (list): A list of lists with the paths of the tree.

        Returns:
           tree_size (int): Returns the size of the tree |V_T|
        """
        #Identify all nodes in the tree
        nodes_in_tree = set()
        for path in T:
            nodes_in_tree.update(path)
        
        tree_size = len(nodes_in_tree)

        return tree_size
    
    def _trace_plot_tree_size(self):
        return None

    # ----------------- Beta (Gibbs update) ------------- #
    def _get_tree_statistics(self, G, T):
        """
        Computes the N_success and N_fail in a given tree for the Gibbs step.

        Args:
            G (networkx graph): A graph G that we are sampling from.
            T (list): A list of lists with the paths of the tree.

        Returns:
           n_success, n_fail  (int, int): Number of sucessfull and failed 
           transmission events for T.
        """

        nodes_in_tree = set()
        for path in T:
            for node in path:
                nodes_in_tree.add(node)

        if len(nodes_in_tree) <= 1:
            return 0
        
        # Succes Events (V_T - 1)
        n_success = len(nodes_in_tree) - 1
        n_fail = 0
        
        for u in nodes_in_tree:
            for v in G.neighbors(u):
                if v not in nodes_in_tree:
                    n_fail += 1
        
        #"""
        print("Tree statistics:",file=f)
        print(f"n_sucess: {n_success}" ,file=f)
        print(f"n_fail: {n_fail}",file=f)
        #"""

        return n_success, n_fail
    
    def _update_beta_gibbs(self):
        """
        Performs the Gibbs update step for the beta parameter.
        P(beta | T) ~ Beta(alpha_prior + N_success, beta_prior + N_fail)
        """

        n_success, n_fail = self._get_tree_statistics(self.G, self.T_current)
        c_posterior = self.c_prior + n_success
        d_posterior = self.d_prior + n_fail

        new_beta = np.random.beta(c_posterior, d_posterior)
        self.beta = max(1e-9, min(new_beta, 1 - 1e-9))

    
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
