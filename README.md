# Generalizing while preserving monotonicity in comparison-based preference learning models

This repository contains the code for the experiments presented in the paper [Generalizing while preserving monotonicity in comparison-based preference learning models](https://arxiv.org/abs/2506.08616v2).

# Repository organization

The repository is organized as follows
- `src/GBTLab2.jl` is a module containing the main data structures and methods.
- `src/Experiment{N}.jl` contains code for experiment number N.
- `data/experiment{N}` is the data folder for experiment number N. 

We committed the data of our own runs in the repository. Be careful when cloning the repository,
as some files may be relatively large (up to 100MB).

This code has been tested with julia version `1.11.7`.

# Running the experiments

In the Julia repl, load the code.
```julia
using DrWatson
@quickactivate "GBTLab2"
```

Every experiment script is organized same pattern. Run the experiment using
```julia
include("src/Experiment1.jl")
run(xp1)
```
This produces results in the data folder, e.g. `data/measure-1.obj`.
To get a description of the experiment
```julia
data = deserialize("data/measure-1.obj")
data.description
```
To plot the results, from within the Julia repl, type
```julia
do_plot(data)
```
