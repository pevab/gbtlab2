"""
EXPERIMENT 2

Normalized MSE with respect to number of samples in the dataset
for three LGBTD models
"""

using .GBTLab2
using LinearAlgebra
using Distributions
using Plots
using Serialization
using Printf
using PrettyTables

struct Experiment2
    root_law :: RootLaw
    n_alternatives :: Int
    n_features :: Int
    n_samples :: StepRange{Int, Int}
    n_seeds :: Int
    sampler :: Sampler
    metric :: ScoreMetric
end

struct ResultsExperiment2
    description :: Experiment2
    gbt_plus_embedding :: Vector{MeasureResult}
    gbt :: Vector{MeasureResult}
    embedding_only :: Vector{MeasureResult}
end

function init(description :: Experiment2) :: ResultsExperiment2
    n = length(description.n_samples)
    ResultsExperiment2(
        description,
        Vector{MeasureResult}(undef, n),
        Vector{MeasureResult}(undef, n),
        Vector{MeasureResult}(undef, n),
    )
end

function do_plot(data :: ResultsExperiment2)
    n_samples = data.description.n_samples
    n_seeds = data.description.n_seeds

    gbt_plus_embedding_mean = map(r -> r.mean, data.gbt_plus_embedding)
    gbt_plus_embedding_err = map(r -> 1.96 * r.std / sqrt(n_seeds), data.gbt_plus_embedding)

    gbt_mean = map(r -> r.mean, data.gbt)
    gbt_err = map(r -> 1.96 * r.std / sqrt(n_seeds), data.gbt)

    embedding_only_mean = map(r -> r.mean, data.embedding_only)
    embedding_only_err = map(r -> 1.96 * r.std / sqrt(n_seeds), data.embedding_only)

    plt = plot()
    plot!(
        n_samples, 
        gbt_plus_embedding_mean, 
        ribbon=gbt_plus_embedding_err,
        label="gbt & embedding",
        linewidth=2,            # increase line thickness
    )
    plot!(
        n_samples, 
        gbt_mean, 
        ribbon=gbt_err,
        label="gbt",
        linewidth=2,            # increase line thickness
    )
    plot!(
        n_samples, 
        embedding_only_mean, 
        ribbon=embedding_only_err,
        label="embedding only",
        linewidth=2,            # increase line thickness
    )
    xlabel!("N")
    ylabel!("nMSE")
    plot!(plt;
        linewidth=3,            # increase line thickness
        titlefontsize=18,       
        guidefontsize=16,       
        tickfontsize=14,        
        legendfontsize=12       
    )
    savefig(plt, datadir("experiment2/fig-2b.png"))
    plt
end

function run(experiment :: Experiment2)
    # hyper parameters
    root_law = experiment.root_law
    metric = experiment.metric
    n_seeds = experiment.n_seeds
    n_samples = experiment.n_samples
    sampler = experiment.sampler
    n_features = experiment.n_features
    n_alternatives = experiment.n_alternatives
    results = init(experiment)

    # for each dataset size
    for (idx, n) in enumerate(n_samples)
        identity = Matrix(I, n_alternatives, n_alternatives)
        mvn = MvNormal(zeros(n_features), Matrix(I, n_features, n_features))
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
            n,
            sampler,
        )
        raw_results = GBTLab2.run(measure)
        results.gbt_plus_embedding[idx] = raw_results[k_gbt_plus_embedding]
        results.gbt[idx] = raw_results[k_gbt]
        results.embedding_only[idx] = raw_results[k_embedding_only]
    end

    # save
    folder = "data/experiment2"
    n_files = count(isfile, readdir(folder, join=true))
    idx = n_files + 1
    filename = "$folder/measure-$idx.obj"
    serialize(filename, results)
end

function print_table(data :: ResultsExperiment2)
    header = [
        "N",
        "GBT & encoding",
        "GBT",
    ]
    n_seeds = data.description.n_seeds
    ns = data.description.n_samples
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
    t = vcat([
        [ns[d] ge(d) g(d)]
        for d in [1 8 15 20 30]
    ]...)
    conf = set_pt_conf(tf = tf_markdown, alignment = :c)
    pretty_table_with_conf(
        conf, t;
        header = header
    )
end

xp2 = Experiment2(
    RootLawUniform(1.0),
    20, # alternatives
    10, # features
    1:10:301, # samples
    1000, # seeds
    SimpleSampler(20),
    NormalizedCenteredScoreMetric(),
)

"""

Figure 2b. nMSE with respect to the number of comparisons N for A=20, D=10, and 1000 seeds.
The following table reports mean ± 1.96 * stdev / sqrt(n_seeds).

|  N  |  GBT & encoding   |       GBT       |
|-----|-------------------|-----------------|
|  1  |  0.934 ± 0.00974  | 0.983 ± 0.00125 |
| 71  |  0.113 ± 0.00348  | 0.454 ± 0.00444 |
| 141 |  0.0636 ± 0.0019  | 0.29 ± 0.00426  |
| 191 | 0.0508 ± 0.00145  | 0.231 ± 0.0039  |
| 291 | 0.0301 ± 0.000989 | 0.182 ± 0.00378 |

"""

xp2_100 = Experiment2(
    RootLawUniform(1.0),
    20, # alternatives
    10, # features
    1:10:301, # samples
    100, # seeds
    SimpleSampler(20),
    NormalizedCenteredScoreMetric(),
)