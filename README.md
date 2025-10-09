# GBTLab2

Open a terminal
```bash
cd GBTLab2.jl
julia
```
Then, in the Julia repl, load the code.
```julia
using DrWatson
@quickactivate "GBTLab2"
```

Each experiment code is located in a separate file, e.g. `Experiment1.jl`.
Run the experiment using
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
