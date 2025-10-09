module GBTLab2

using Base.Threads
using Dates
using LinearAlgebra
using Distributions
using Optim
using Statistics

export RootLaw, RootLawUniform, eval
export LGBTDModel, LGBTDResult, infer
export Measure, MeasureResult, run
export Sampler, SimpleSampler, ErdosRenyiSampler, BipartiteSampler, ComparisonPair, ComparisonSample
export ScoreMetric, NormalizedCenteredScoreMetric, CenteredScoreMetric

"""
ROOT LAWS
"""

abstract type RootLaw end

struct RootLawUniform <: RootLaw 
    max_score :: Float64
end

function eval(root_law :: RootLawUniform, score_difference :: Float64) :: Float64
    epsilon = 1e-6
    if abs(score_difference) <= epsilon * root_law.max_score
        return 0.0
    end
    log(sinh(score_difference) / score_difference)
end

"""
COMPARISONS
"""

struct ComparisonPair
    a :: Int64
    b :: Int64
end

struct ComparisonSample
    a :: Int64
    b :: Int64
    r :: Float64
end

function compare(_ :: RootLawUniform, score_difference::Float64) :: Float64
    if abs(score_difference) <= 1e-6
        return rand(Uniform(-1.0, 1.0))
    end

    param = abs(score_difference)
    exponential = Exponential(1.0 / param)

    sample = 3.0

    while sample > 2.0
        sample = rand(exponential)
    end

    return -(sample - 1.0) * sign(score_difference)
end

"""
SAMPLER
"""

abstract type Sampler end

struct SimpleSampler <: Sampler
    n_alternatives :: Int
end

function sample(sampler :: SimpleSampler, n_samples :: Int) :: Vector{ComparisonPair}
    n_alternatives = sampler.n_alternatives
    result = Vector{ComparisonPair}()
    for _ in 1:n_samples
        a = rand(1:n_alternatives)
        b = a
        while a == b
            b = rand(1:n_alternatives)
        end
        push!(result, ComparisonPair(a,b))
    end
    result
end

struct ErdosRenyiSampler <: Sampler
    n_alternatives :: Int
    p_c :: Float64
end

# weird: this one does not consider the requested number of samples
function sample(sampler :: ErdosRenyiSampler,  _ :: Int) :: Vector{ComparisonPair}
    n_alternatives = sampler.n_alternatives
    result = Vector{ComparisonPair}()
    for a in 1:n_alternatives
        for b in 1:n_alternatives
            if a != b &&  rand() < sampler.p_c
                push!(result, ComparisonPair(a,b))
            end
        end
    end
    result
end

struct BipartiteSampler <: Sampler
    n_alternatives :: Int
    split_at :: Int
end

function sample(sampler :: BipartiteSampler, n_samples :: Int) :: Vector{ComparisonPair}
    n_alternatives = sampler.n_alternatives
    s = sampler.split_at
    result = Vector{ComparisonPair}()
    group_a = 1:s
    group_b = (s+1):n_alternatives
    for _ in 1:n_samples
        a = rand(group_a)
        b = rand(group_b)
        push!(result, ComparisonPair(a,b))
    end
    result
end


"""
MODEL
"""

struct LGBTDModel
    root_law :: RootLaw
    embedding :: Array{Float64}
    sigma :: Float64
    laplacian :: Array{Float64}
end

function beta_covariance_matrix(model :: LGBTDModel) :: Array{Float64}
    n_features, _ = size(model.embedding)
    model.sigma^(-2) * Matrix(I, n_features, n_features) + model.embedding * model.laplacian * transpose(model.embedding)
end

struct LGBTDResult
    beta :: Array{Float64}
    scores :: Array{Float64}
end


"""
DATASETS
"""

struct SyntheticDataset
    samples :: Vector{ComparisonSample}
    true_beta :: Vector{Float64}
    true_scores :: Vector{Float64}
end

abstract type SyntheticDatasetGenerator end

struct LGBTDSyntheticDatasetGenerator <: SyntheticDatasetGenerator
    model :: LGBTDModel
    n_samples :: Int
    sampler :: Sampler
end

function generate(g :: LGBTDSyntheticDatasetGenerator) :: SyntheticDataset
    root_law = g.model.root_law
    embedding = g.model.embedding
    laplacian = g.model.laplacian
    n_features, _ = size(embedding)
    sigma2 = g.model.sigma^2
    i = Matrix(I, n_features, n_features)


    bcm = (1.0/sigma2) * i + embedding * laplacian * transpose(embedding)

    true_beta = rand(MvNormal(zeros(n_features), bcm))
    true_scores = transpose(embedding) * true_beta

    comparison_pairs = sample(g.sampler,g.n_samples)
    samples = map(pair -> 
        ComparisonSample(
            pair.a, 
            pair.b, 
            compare(root_law, true_scores[pair.a] - true_scores[pair.b])
            ), comparison_pairs)
    SyntheticDataset(samples, true_beta, true_scores)
end

"""
INFERENCE
"""

function mk_loss(
    samples :: Vector{ComparisonSample},
    embedding :: Array{Float64},
    bcm :: Array{Float64},
    root_law :: RootLaw,
) :: Function
    (beta :: Array{Float64}) -> begin
        theta = transpose(embedding) * beta
        regularization = 0.5 * transpose(beta) * bcm * beta
        fitness = sum([eval(root_law, theta[sample.a] - theta[sample.b]) - sample.r * (theta[sample.a] - theta[sample.b]) for sample in samples ])
        regularization + fitness
    end
end

function infer(model :: LGBTDModel, samples :: Vector{ComparisonSample}) :: LGBTDResult
    n_features, _ = size(model.embedding)
    bcm = beta_covariance_matrix(model)
    beta_prior = rand(MvNormal(zeros(n_features), bcm))
    loss = mk_loss(samples, model.embedding, bcm, model.root_law)
    inferred_beta = Optim.minimizer(optimize(loss, beta_prior, LBFGS())) # variants: LBFGS, BFGS
    inferred_scores = transpose(model.embedding) * inferred_beta
    LGBTDResult(
        inferred_beta,
        inferred_scores,
    )
end

"""
EVALUATION
"""

abstract type ScoreMetric end

struct NormalizedCenteredScoreMetric <: ScoreMetric end
struct CenteredScoreMetric <: ScoreMetric end

function eval(_ :: NormalizedCenteredScoreMetric, theta_star :: Vector{Float64}, theta_true :: Vector{Float64}) :: Float64
    theta_star_mean = mean(theta_star)
    theta_true_mean = mean(theta_true)
    theta_star_centered = theta_star .- theta_star_mean
    theta_true_centered = theta_true .- theta_true_mean

    difference = theta_star_centered - theta_true_centered
    truth = theta_true_centered
    dot(difference, difference) / dot(truth, truth)
end

function eval(_ :: CenteredScoreMetric, theta_star :: Vector{Float64}, theta_true :: Vector{Float64}) :: Float64
    theta_star_mean = mean(theta_star)
    theta_true_mean = mean(theta_true)
    theta_star_centered = theta_star .- theta_star_mean
    theta_true_centered = theta_true .- theta_true_mean

    difference = theta_star_centered - theta_true_centered
    dot(difference, difference)
end

"""
MEASURE
"""

struct Measure 
    true_model :: LGBTDModel
    models :: Dict{String, LGBTDModel}
    metric :: ScoreMetric
    n_seeds :: Int
    n_samples :: Int
    sampler :: Sampler
end

struct MeasureResult
    mean :: Float64
    std :: Float64
    min :: Float64
    max :: Float64
    q25 :: Float64
    q50 :: Float64
    q75 :: Float64
    lo :: Float64
    hi :: Float64
end

function run(measure :: Measure) :: Dict{String, MeasureResult}
    n_seeds = measure.n_seeds
    metric =  measure.metric
    g = LGBTDSyntheticDatasetGenerator(
        measure.true_model,
        measure.n_samples,
        measure.sampler,
    )

    errors = Dict{String, Vector{Float64}}()
    for (k, _) in measure.models
        errors[k] = Vector{Float64}(undef, n_seeds)
    end
    @threads for i in 1:n_seeds
        dataset = generate(g)
        true_scores = dataset.true_scores
    
        for (k,model) in measure.models
            result = infer(model, dataset)
            inferred_scores = result.scores
            e = eval(metric, inferred_scores, true_scores)
            errors[k][i] = e
        end
    end
    Dict(
        k => MeasureResult(
            mean(v),
            std(v),
            minimum(v),
            maximum(v),
            quantile(v, 0.25),
            quantile(v, 0.50),
            quantile(v, 0.75),
            mean(v) - 1.96*std(v)/sqrt(n_seeds),
            mean(v) + 1.96*std(v)/sqrt(n_seeds),
        ) for (k,v) in errors
    )
end

end