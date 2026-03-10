### Markov Chains on Trees in Networks of Labeled Nodes

This repository contains all the code necessary to reproduce the results in the dissertation "Markov Chains on Trees in Networks of Labeled Nodes" for the master's program in Computational Mathematics at ICMC/USP. 

#### Getting Started 
All the required packages can be found in the file "requirements.txt".

#### Repository Structure
`data/`: Contains the sampling outputs for each graph used in the tests.

`epidemic_simulation/`: Contains the implementation of the SIR (Susceptible-Infected-Recovered) network model.

`notebooks/`: Contains the Jupyter notebooks used to generate some of the figures presented in the dissertation.

`sample/`: Contains the `TreeSampler` class, which implements the Metropolis-Hastings algorithm for sampling trees.

`tests/`: Contains the tests for the Metropolis-Hastings implementation.

`visualization/`: Contains auxiliary functions to visualize the graphs with partial information on infection times, organized by a color scheme.
