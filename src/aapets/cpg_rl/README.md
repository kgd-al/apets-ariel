# Parameter scalability in Evolutionary Robotics

This folder contains the code used to investigate the impact of controller type (CPG/MLP) and architecture
(neighborhood, number of layers ...) on multiple performance metrics including speed and stability.

## Training algorithms

Two training protocols are available: black-box evolution (via CMA-ES) and gradient descent (via PPO).
In the following $n$ is the number of optimizable parameters of the controller being trained.

### CMA-ES

The experiment relies on the [cma](https://pypi.org/project/cma/) Python library for the Covariant Matrix Adaptation
Evolutionary Strategy with minimal changes from defaults. Specifically:

| Field             | Value                |
|-------------------|----------------------|
| initial mean      | $(0.5)_{0\le i < n}$ |
| initial deviation | 0.5                  | 
| tolfun            | 0                    |
| tolflatfitness    | 10                   | 

All other parameters are kept to their defaults. The fist two rows provide a valid starting point for the exploration,
based of the expected parameter ranges. The last two try to reduce premature convergence when faced with many parameters
(e.g. an MLP with > 1000 weights).

### PPO

The Reinforcement Learning portion of the experiment uses the implementation of Proximal Policy Optimization (PPO) from
[Stable Baselines3](https://stable-baselines3.readthedocs.io/), again with minimal changes.

| Field               | Value  |
|---------------------|--------|
| normalize_advantage | True   |
| n_steps             | 256    |
| batch_size          | 32     |
| gae_lambda          | 0.95   |
| gamma               | 0.99   |
| learning_rate       | 2.5e-4 |

These values are derived from recommended parameters used for
[ant-v0](https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml#L201)

## Controllers

Two controller types are used in this experiment: Central Pattern Generators (CPGs) and Multi-Layer Perceptrons (MLPs).
The first uses the implementation from [Revolve2](https://github.com/ci-group/revolve2), loosely based on Hopf
oscillators. These additionally have customizable neighborhood, compared to traditional use of $k=2$ (see `common.controllers.neighborhood_cpg`)
The second relies on PyTorch networks either as provided by Stable Baselines3 or reimplemented (see `common.controllers.mlp_tensor`).

## Task

Robots are expected to move forward along the X axis. Rewards depend on the specific training regime.
