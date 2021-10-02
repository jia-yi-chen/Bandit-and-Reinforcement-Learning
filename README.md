# Reinforcement Learning

Author: [Jiayi Chen](https://github.com/jia-yi-chen)

Time: Oct 2020

## Implemented Algorithms:

* **Bandit**:
  - **Multi-arm Bandit**: epsilon-greedy; upper confidence bound (UCB); Thompsom Sampling (TS); Perturbed-history Exploration (PHE)
  - **Contextual Linear Bandit**: LinUCB; LinTS; LinPHE
* **Reinforcement Learning**: 
  - **Dynamic programming** solution for Markov Decision Process (known environment): value iteration; policy iteration
  - **Model-free control**: off-policy Monte Carlo (MC) control; off-policy Temporal Difference (TD) control (i.e., Q-learning)

## Requirements

* python 3



## Getting Started

### Bandit

**Simulation environment**: 
- Action: articles
- User: users
- In each time step, we will iterate over each user, make recommendation to it and receive an reward of the recommended article.


```
run "/bandit/SimulationComparison.py"
```
See "/bandit/lib/$ALGOTHISNAME$.py" for each algorithm.



### Reinforcement Learning

**Simulation environment**: 4-by-4 grid world. The goal of the agent is to get to the goal (cell 15) as soon as possible, while avoid the pits (cell grid\[1\]\[1\] and grid\[2\]\[1\]).

#### Dynamic programming for Markov Decision Process
```
run "/rl/runDP.py"
```
#### Model-free MC/TD control
```
run "/rl/runRL.py"
```

