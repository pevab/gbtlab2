"""
EXPERIMENT 1 

Normalized MSE with respect to number of embedding features
for three LGBTD models
"""

using .GBTLab2
using LinearAlgebra
using Distributions
using Plots
using Serialization
using PrettyTables
using Printf

struct Experiment1
    root_law :: RootLaw
    n_alternatives :: Int
    n_samples :: Int
    n_seeds :: Int
    sampler :: Sampler
    metric :: ScoreMetric
    d_max :: Int
end

struct ResultsExperiment1
    description :: Experiment1
    gbt_plus_embedding :: Vector{MeasureResult}
    gbt :: Vector{MeasureResult}
    embedding_only :: Vector{MeasureResult}
end

function init_experiment1(description :: Experiment1) :: ResultsExperiment1
    d_max = description.d_max
    ResultsExperiment1(
        description,
        Vector{MeasureResult}(undef, d_max),
        Vector{MeasureResult}(undef, d_max),
        Vector{MeasureResult}(undef, d_max),
    )
end

function do_plot(data :: ResultsExperiment1)
    d = range(1, data.description.d_max)
    n_seeds = data.description.n_seeds

    gbt_plus_embedding_mean = map(r -> r.mean, data.gbt_plus_embedding)
    gbt_plus_embedding_err = map(r -> 1.96 * r.std / sqrt(n_seeds), data.gbt_plus_embedding)

    gbt_mean = map(r -> r.mean, data.gbt)
    gbt_err = map(r -> 1.96 * r.std / sqrt(n_seeds), data.gbt)

    embedding_only_mean = map(r -> r.mean, data.embedding_only)
    embedding_only_err = map(r -> 1.96 * r.std / sqrt(n_seeds), data.embedding_only)

    plt = plot()
    plot!(
        d, 
        gbt_plus_embedding_mean, 
        ribbon=gbt_plus_embedding_err,
        label="gbt & embedding",
        linewidth=2,
    )
    plot!(
        d, 
        gbt_mean, 
        ribbon=gbt_err,
        label="gbt",
        linewidth=2,
    )
    plot!(
        d, 
        embedding_only_mean, 
        ribbon=embedding_only_err,
        label="embedding only",
        linewidth=2,
    )
    xlabel!("D")
    ylabel!("nMSE")
    plot!(plt;
        linewidth=2,            # increase line thickness
        titlefontsize=18,       
        guidefontsize=16,       
        tickfontsize=14,        
        legendfontsize=12       
    )
    savefig(plt, datadir("experiment1/fig-2a.png"))
    plt
end


function run(experiment :: Experiment1)
    # hyper parameters
    root_law = experiment.root_law
    metric = experiment.metric
    n_seeds = experiment.n_seeds
    n_samples = experiment.n_samples
    sampler = experiment.sampler
    n_alternatives = experiment.n_alternatives

    results = init_experiment1(experiment)
    d_max = experiment.d_max

    # for each number d of embedding features
    for d in range(1, d_max)
        identity = Matrix(I, n_alternatives, n_alternatives)
        mvn = MvNormal(zeros(d), Matrix(I,d,d))
        x = rand(mvn, n_alternatives)
        true_x = vcat(identity, x)
        zero_laplacian = zeros(n_alternatives, n_alternatives)
        true_model = LGBTDModel(
            root_law,
            true_x,
            1.0,
            zero_laplacian,
        )

        k_gbt_plus_embedding = "gbt_plus_embedding"
        k_gbt = "gbt"
        k_embedding_only = "embedding_only"
        measure = Measure(
            true_model,
            Dict(
                k_gbt_plus_embedding => LGBTDModel(root_law, true_x, 1.0, zero_laplacian),
                k_gbt => LGBTDModel(root_law, identity, 1.0, zero_laplacian),
                k_embedding_only => LGBTDModel(root_law, x, 1.0, zero_laplacian),
            ),
            metric,
            n_seeds,
            n_samples,
            sampler,
        )
        raw_results = GBTLab2.run(measure)
        results.gbt_plus_embedding[d] = raw_results[k_gbt_plus_embedding]
        results.gbt[d] = raw_results[k_gbt]
        results.embedding_only[d] = raw_results[k_embedding_only]
    end
    
    # save
    folder = "data/experiment1"
    n_files = count(isfile, readdir(folder, join=true))
    idx = n_files + 1
    filename = "$folder/measure-$idx.obj"
    serialize(filename, results)
end

xp1 = Experiment1(
    RootLawUniform(1.0),
    25, # alternatives
    500, # samples
    100, # seeds
    SimpleSampler(25),
    NormalizedCenteredScoreMetric(),
    25, # maximum feature dimension
)

function print_table(data :: ResultsExperiment1)
    header = [
        "Dimension D",
        "GBT & encoding",
        "GBT",
        "Encoding",
    ]
    n_seeds = data.description.n_seeds
    function ge(d :: Int)
        v = data.gbt_plus_embedding
        m = @sprintf("%.3g", v[d].mean)
        s = @sprintf("%.3g", 1.96 * v[d].std / sqrt(n_seeds))
        "$m ± $s"
    end
    function g(d :: Int)
        v = data.gbt
        m = @sprintf("%.3g", v[d].mean)
        s = @sprintf("%.3g", 1.96 * v[d].std / sqrt(n_seeds))
        "$m ± $s"
    end
    function e(d :: Int)
        v = data.embedding_only
        m = @sprintf("%.3g", v[d].mean)
        s = @sprintf("%.3g", 1.96 * v[d].std / sqrt(n_seeds))
        "$m ± $s"
    end
    t = vcat([
        [d ge(d) g(d) e(d)]
        for d in [2 4 8 16 20]
    ]...)
    conf = set_pt_conf(tf = tf_markdown, alignment = :c)
    pretty_table_with_conf(
        conf, t;
        header = header
    )
end

"""

Figure 2a. nMSE as a function of D for A=25 alternatives and N=500 comparisons over 100 seeds.
The following table reports mean ± 1.96 * stdev / sqrt(n_seeds).

| Dimension D |  GBT & encoding  |       GBT        |     Encoding     |
|-------------|------------------|------------------|------------------|
|      2      | 0.0497 ± 0.00605 | 0.0801 ± 0.00676 |  0.39 ± 0.0432   |
|      8      | 0.0269 ± 0.00249 |  0.102 ± 0.0079  |  0.125 ± 0.0124  |
|     20      | 0.0183 ± 0.00169 |   0.2 ± 0.0112   | 0.0271 ± 0.00274 |

"""

xp1000 = Experiment1(
    RootLawUniform(1.0),
    25, # alternatives
    500, # samples
    1000, # seeds
    SimpleSampler(25),
    NormalizedCenteredScoreMetric(),
    25, # maximum feature dimension
)