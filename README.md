[ARIEL](https://github.com/kgd-al/ariel)-powered version of the APets [project](https://arxiv.org/abs/2407.21357)

The core idea behind this repository is to evolve body and brains (not always together) 
*in* an interactive settings *for* human-machine interaction

# Project structure

The main code lives under src with additional folders providing persistent storage or helper bash scripts

## Assets
The `assets` folder contains GIFs of champions from select experiments referenced in the related README

## Scripts
Various general-purpose bash scripts for persistent calls to some experiments as well as small utilities
to communicate with the in-house cluster. Not relevant for reproducibility.

## Sources

Every folder under `src/aapets` either provides the context for an experiment or generic helper code:
- `bin` contains the single general-purpose entry-point of the project where any properly saved
individual can be seen in action in its environment. See the help for details on how to use it.
- `common` is a collection of classes providing the common backbone of the project
(configuration, controllers, persistent storage)
- `misc` contains things with no place elsewhere

The remaining folders are dedicated to a specific experiment/environment with their dedicated README:
- `cpg_rl` studies parameter scalability with MLPs and CPGs of varying scale
- `fetch` defines an environment and higher-level controller for a robot having to fetch a ball and bring it back to a
human
- `g_cpg` focuses on general-control CPGs for multi-layer policies
- `miel` is an abandoned formulation of the Multi-level Interactive Evolution with Learning
- `watchmaker` provides a GUI for the interactive evolution of a robot's controller (CPG)
- `zoo` trains controllers for all canonical morphologies in ARIEL
