### Markov Chains on Trees in Networks of Labeled Nodes

This repository contains all the code necessary to reproduce the results in the dissertation "Markov Chains on Trees in Networks of Labeled Nodes" submitted to the Instituto de
Ciências Matemáticas e de Computação – ICMC- USP, in partial fulfillment of the requirements for the degree of the Master Program in Computer Science and Computational Mathematics. 

Advisor: Prof. Dr. Tiago Pereira

### Abstract:
During an infectious disease epidemic, some infected individuals in a population may go unidentified for various reasons: either due to difficulties accessing the healthcare system, or
because they are asymptomatic or oligosymptomatic cases. A health surveillance system would like to be able to identify these individuals in order to formulate more efficient control measures.
Thus, assuming a network of contacts and a set of marked nodes where their respective infection times are known, our interest is to make inferences from the partially observed data to estimate
the probability of an unobserved node having been infected. Our main strategy is to use Markov Chain Monte Carlo simulation algorithms. As is known, a major difficulty with these algorithms
is demonstrating the ergodicity of the underlying Markov chain. To address this, we define operations on the feasible trees, such as joining and removing an unmarked node, and swapping
unmarked paths. Once it has been proven that, given two feasible trees, one can map one tree onto the other using the operations described above, it is possible to develop a Monte Carlo
algorithm using Markov chains to obtain inferences about the data. In this work, examples are presented, such as the simple path and the star graph, in which the local operations of adding
and removing a node are sufficient to make the chain irreducible and, with this, estimate the probabilities of interest of an unobserved node having been infected.

#### Keywords: Markov Chain Monte Carlo, Epidemic network inference, Partially observed data, Contact networks.

#### Getting Started 
All the required packages can be found in the file "requirements.txt".

#### Repository Structure
`data/`: Contains the sampling outputs for each graph used in the tests.

`epidemic_simulation/`: Contains the implementation of the SIR (Susceptible-Infected-Recovered) network model.

`notebooks/`: Contains the Jupyter notebooks used to generate some of the figures presented in the dissertation.

`sample/`: Contains the `TreeSampler` class, which implements the Metropolis-Hastings algorithm for sampling trees.

`tests/`: Contains the tests for the Metropolis-Hastings implementation.

`visualization/`: Contains auxiliary functions to visualize the graphs with partial information on infection times, organized by a color scheme.
