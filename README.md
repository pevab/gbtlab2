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
include(srcdir("GBTLab2.jl"))
```

Every experiment script is organized same pattern. 

## Experiment 1 - nMSE as a function of the embedding dimension D

Run the experiment using
```julia
include(srcdir("Experiment1.jl"))
run(xp1)
```
This produces results in the data folder. To plot the results, from within the Julia repl, type
```julia
data = deserialize(datadir("experiment1/measure-1.obj"))
do_plot(data)
```

## Experiment 2 - nMSE as a function of the number N of comparisons

Run the experiment using
```julia
include(srcdir("Experiment2.jl"))
run(xp2)
```
This produces results in the data folder. To plot the results, from within the Julia repl, type
```julia
data = deserialize(datadir("experiment2/measure-1.obj"))
do_plot(data)
```

## Experiment 3 - Two cliques
Run the experiment using
```julia
include(srcdir("Experiment3.jl"))
run(xp3)
```
This produces results in the data folder. To plot the results, from within the Julia repl, type
```julia
data = deserialize(datadir("experiment3/measure-1.obj"))
do_plot(data)
```

## Experiment 4 - Real-world dataset
Run the experiment using
```julia
include(srcdir("Experiment4.jl"))
run(xp4)
```
Warning, the experiment can take up to 2 hours.
This produces results in the data folder. To plot the results, from within the Julia repl, type
```julia
data = deserialize(datadir("experiment4/blueberry.obj"))
do_plot(data)
```
