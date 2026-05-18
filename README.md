### Markov Chains on Trees in Networks of Labeled Nodes

This repository contains all the code necessary to reproduce the results in the dissertation "Markov Chains on Trees in Networks of Labeled Nodes" submitted to the Instituto de
Ciências Matemáticas e de Computação – ICMC- USP, in partial fulfillment of the requirements for the degree of the Master Program in Computer Science and Computational Mathematics. 
Advisor: Prof. Dr. Tiago Pereira

### Abstract:
During an epidemic of an infectious disease, some individuals in a population may not be identified for different reasons, whether due to difficulties in accessing the healthcare system or
because they are asymptomatic cases. The health surveillance system would like to be able to identify these individuals to formulate more efficient control measures. Thus, assuming a network
of contacts and a set of marked nodes for which some observed infection times are known, our interest lies in making inferences from partially observed data to estimate the probability of an
unobserved node having been infected. Our main strategy is to use Monte Carlo Markov Chain simulation algorithms. As is known, a major difficulty with these algorithms is demonstrating
the ergodicity of the underlying Markov Chain. To address this, we define operations on the feasible trees, such as joining and removing an unmarked node or swapping unmarked paths.
Once it was proven that given two feasible trees, it is possible to map one to the other using the operations described above, a Monte Carlo algorithm using Markov Chains was developed to
obtain inferences about the data. 
Keywords: Markov Chain Monte Carlo, Epidemic network inference, Partially observed data, Contact networks.

#### Getting Started 
All the required packages can be found in the file "requirements.txt".

#### Repository Structure
`data/`: Contains the sampling outputs for each graph used in the tests.

`epidemic_simulation/`: Contains the implementation of the SIR (Susceptible-Infected-Recovered) network model.

`notebooks/`: Contains the Jupyter notebooks used to generate some of the figures presented in the dissertation.

`sample/`: Contains the `TreeSampler` class, which implements the Metropolis-Hastings algorithm for sampling trees.

`tests/`: Contains the tests for the Metropolis-Hastings implementation.

`visualization/`: Contains auxiliary functions to visualize the graphs with partial information on infection times, organized by a color scheme.
