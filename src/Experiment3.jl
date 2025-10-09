"""
EXPERIMENT 3

Normalized MSE with respect to number of comparisons
between two cliques.

Consider a true embedding that is a one-hot encoding
among two features. Hence, true_x is a 2 x A matrix.

"""

using .GBTLab2
using LinearAlgebra
using Distributions
using Plots
using Serialization

struct Experiment3
    root_law :: RootLaw
    n_alternatives :: Int
    split_at :: Int
    n_samples :: StepRange{Int, Int}
    n_seeds :: Int
    metric :: ScoreMetric
end

struct ResultsExperiment3
    description :: Experiment3
    gbt :: Vector{MeasureResult}
    gbt_plus_embedding :: Vector{MeasureResult}
    embedding_only :: Vector{MeasureResult}
    gbt_laplacian_regularization :: Vector{MeasureResult}
end

function init(description :: Experiment3) :: ResultsExperiment3
    n = length(description.n_samples)
    ResultsExperiment3(
        description,
        Vector{MeasureResult}(undef, n),
        Vector{MeasureResult}(undef, n),
        Vector{MeasureResult}(undef, n),
        Vector{MeasureResult}(undef, n),
    )
end

function do_plot(data :: ResultsExperiment3)
    n_samples = data.description.n_samples

    gbt_plus_embedding_mean = map(r -> r.mean, data.gbt_plus_embedding)
    gbt_plus_embedding_lo = map(r -> r.lo, data.gbt_plus_embedding)
    gbt_plus_embedding_hi = map(r -> r.hi, data.gbt_plus_embedding)

    gbt_mean = map(r -> r.mean, data.gbt)
    gbt_lo = map(r -> r.lo,    data.gbt)
    gbt_hi = map(r -> r.hi,    data.gbt)

    embedding_only_mean = map(r -> r.mean, data.embedding_only)
    embedding_only_lo = map(r -> r.lo,    data.embedding_only)
    embedding_only_hi = map(r -> r.hi,    data.embedding_only)

    gbt_laplacian_regularization_mean = map(r -> r.mean, data.gbt_laplacian_regularization)
    gbt_laplacian_regularization_lo = map(r -> r.lo,    data.gbt_laplacian_regularization)
    gbt_laplacian_regularization_hi = map(r -> r.hi,    data.gbt_laplacian_regularization)

    plt = plot()
    plot!(
        n_samples, 
        gbt_plus_embedding_mean, 
        ribbon=(gbt_plus_embedding_lo, gbt_plus_embedding_hi),
        label="gbt & embedding",
    )
    plot!(
        n_samples, 
        gbt_mean, 
        ribbon=(gbt_lo, gbt_hi),
        label="gbt",
    )
    plot!(
        n_samples, 
        embedding_only_mean, 
        ribbon=(embedding_only_lo, embedding_only_hi),
        label="embedding only",
    )
    # plot!(
    #     n_samples, 
    #     gbt_laplacian_regularization_mean, 
    #     ribbon=(gbt_laplacian_regularization_lo, gbt_laplacian_regularization_hi),
    #     label="gbt with laplacian regularization",
    # )
    plt
end

function mk_true_embedding(n_alternatives :: Int, split_at :: Int) :: Matrix{Float64}
    result = zeros(2, n_alternatives)
    for a in 1:split_at
        result[1,a] = 1.0
    end
    for b in (split_at + 1):n_alternatives
        result[2,b] = 1.0
    end
    result
end

function mk_laplacian(n_alternatives :: Int, split_at :: Int) :: Matrix{Float64}
    result = zeros(n_alternatives, n_alternatives)
    n_a = split_at
    n_b = n_alternatives - split_at
    for a in 1:n_alternatives
        if a <= split_at
            result[a,a] = n_a - 1
        else
            result[a,a] = n_b - 1
        end
        for b in (a + 1):n_alternatives
            if a <= split_at && b <= split_at
                result[a,b] = -1.0
                result[b,a] = -1.0
            elseif a > split_at && b > split_at
                result[a,b] = -1.0
                result[b,a] = -1.0
            else
            end 
        end
    end
    result
end


function run(experiment :: Experiment3)
    # hyper parameters
    root_law = experiment.root_law
    metric = experiment.metric
    n_seeds = experiment.n_seeds
    n_samples = experiment.n_samples
    n_alternatives = experiment.n_alternatives
    split_at = experiment.split_at
    sampler = BipartiteSampler(n_alternatives, split_at)
    n_features = 2
    results = init(experiment)

    k_gbt = "gbt"
    k_gbt_laplacian_regularization = "gbt_laplacian_regularization"
    k_embedding_only = "embedding_only"
    k_gbt_plus_embedding = "gbt_plus_embedding"

    # for number of samples between the two cliques
    for (idx, n) in enumerate(n_samples)
        identity = Matrix{Float64}(I, n_alternatives, n_alternatives)
        true_x = mk_true_embedding(n_alternatives, split_at)
        zero_laplacian = zeros(n_alternatives, n_alternatives)
        true_model = LGBTDModel(
            root_law,
            true_x,
            1.0,
            zero_laplacian,
        )
        laplacian = mk_laplacian(n_alternatives, split_at)
        measure = Measure(
            true_model,
            Dict(
                k_gbt => LGBTDModel(root_law, identity, 1.0, zero_laplacian),
                k_gbt_plus_embedding => LGBTDModel(root_law, vcat(0.8*identity, 0.2*true_x), 1.0, zero_laplacian),
                k_embedding_only => LGBTDModel(root_law, true_x, 1.0, zero_laplacian),
                k_gbt_laplacian_regularization => LGBTDModel(root_law, identity, 1.0, laplacian),
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
        results.gbt_laplacian_regularization[idx] = raw_results[k_gbt_laplacian_regularization]
    end

    # save
    folder = "data/experiment3"
    n_files = count(isfile, readdir(folder, join=true))
    idx = n_files + 1
    filename = "$folder/measure-$idx.obj"
    serialize(filename, results)
end

xp3 = Experiment3(
    RootLawUniform(1.0),
    50,
    25,
    10:10:300,
    100,
    CenteredScoreMetric(),
)

xp3_b = Experiment3(
    RootLawUniform(1.0),
    50,
    25,
    10:10:80,
    20,
    CenteredScoreMetric(),
)
