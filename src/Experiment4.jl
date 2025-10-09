"""
EXPERIMENT 4 

Compare GBT and LGBTD on real-world data.
    - load tournesol data
    - GBT/GBTD setup
        - common prior (identity)
        - GBTD 
            - one-hot encoding of channels, concatenated with lambda * identity
            - zero laplacian
    - split in training and test set
    - infer beta and scores
        - for GBT
        - for GBTD
    - prediction function
        Let s be a score difference. The GBT model states that
            P( c | s ) = f(c) exp( cs - Phi(s) ) = exp( cs - F(c) - Phi(s))
            where F(c) = -ln f(c)
        output c = expected value  ∫ c⋅P(c|s) dc
    - metric = empirical risk of the prediction function on the test set
            quadratic loss |c_actual - c_predicted|^2
"""

using DrWatson
@quickactivate "GBTLab2"

# include(srcdir("GBTLab2.jl"))
using .GBTLab2

using Base.Threads
using LinearAlgebra
using CSV
using JSON
using DataFrames
using DataStructures
using Bijections
using Statistics
using Random
using StatsBase
using Serialization
using Plots
using StatsPlots
using MLBase
using Setfield

"""
    Video metadata
"""
struct VideoYoutubeId
    value :: String
end

struct ChannelId
    value :: String
end

struct VideoMetadata
    name :: String
    youtube_id :: String
    channel_name :: String
    channel_id :: String
end

function video_id(m :: VideoMetadata) :: VideoYoutubeId
    VideoYoutubeId(m.youtube_id)
end

function channel_id(m :: VideoMetadata) :: ChannelId
    ChannelId(m.channel_id)
end

"""
    Tournesol dataset
"""

struct TournesolDataset
    comparisons :: Vector{ComparisonSample}
    video_key_map :: Bijection{VideoYoutubeId, Int}
    metadata :: Dict{VideoYoutubeId, VideoMetadata}
end

function n_videos(dataset :: TournesolDataset) :: Int
    length(dataset.video_key_map)
end

function video_key_map(dataset :: TournesolDataset) :: Bijection{VideoYoutubeId, Int}
    dataset.video_key_map
end

function channel_key_map(dataset :: TournesolDataset) :: Bijection{ChannelId, Int}
    channel_ids = Set(map(md -> channel_id(md), values(dataset.metadata)))
    channel_key_map = Bijection{ChannelId, Int}()
    for (k,v) in enumerate(channel_ids)
        channel_key_map[v] = k
    end
    channel_key_map
end

function load_raw_comparisons(
    comparisons_file :: String,
    username :: String,
    criteria :: String;
    limit :: Union{Int, Nothing}
) :: DataFrame
    df = CSV.read(comparisons_file, DataFrame; drop=["score_max", "week_date"])
    df = df[df.public_username .== username .&& df.criteria .== criteria, Not(:public_username, :criteria)]
    if limit !== nothing
        rows = sample(1:nrow(df), limit; replace=false)
        df = df[rows, :]
    end
    df
end

function load_raw_metadata(
    metadata_file :: String,
) :: DataFrame
    metadata = CSV.read(metadata_file, DataFrame)

    extract_uploader(s::String) = JSON.parse(s)["uploader"]
    extract_channel_id(s::String) = JSON.parse(s)["channel_id"]
    extract_name(s::String) = JSON.parse(s)["name"]
    extract_video_id(s::String15) = s[4:end]

    metadata[!, "name"] = extract_name.(metadata[!, "metadata"])
    metadata[!, "channel_name"] = extract_uploader.(metadata[!, "metadata"])
    metadata[!, "channel_id"] = extract_channel_id.(metadata[!, "metadata"])
    metadata[!, "id"] = extract_video_id.(metadata[!, "uid"])

    metadata
end

function video_key_map_from_raw_comparisons(raw_comparisons :: DataFrame) :: Bijection{VideoYoutubeId, Int}
    video_ids = unique(vcat(raw_comparisons.video_a, raw_comparisons.video_b))
    key_map = Bijection{VideoYoutubeId, Int}()
    for (k,v) in enumerate(video_ids)
        key_map[VideoYoutubeId(v)] = k
    end
    key_map
end

function metadata_from_raw_data(video_ids :: Vector{VideoYoutubeId}, raw_metadata :: DataFrame) :: Dict{VideoYoutubeId, VideoMetadata}
    metadata = Dict{VideoYoutubeId, VideoMetadata}()
    
    metadata_key_map = Dict{VideoYoutubeId, Int}()
    for (key, row) in enumerate(eachrow(raw_metadata))
        video_id = VideoYoutubeId(row.id)
        metadata_key_map[video_id] = key
    end

    missing_keys = setdiff(video_ids, keys(metadata_key_map))
    if length(missing_keys) > 0
        error("missing keys ($(length(missing_keys))): ", missing_keys)
    end

    for video_id in video_ids
        video_key = metadata_key_map[video_id]
        row = raw_metadata[video_key, :]
        metadata[video_id] = VideoMetadata(
            row[:name],
            row[:id],
            row[:channel_name],
            row[:channel_id],
        )
    end
    metadata
end

function samples_from_raw_comparisons(
    raw_comparisons :: DataFrame,
    video_key_map :: Bijection{VideoYoutubeId, Int}
) :: Vector{ComparisonSample}
    comparisons = ComparisonSample[]
    for row in eachrow(raw_comparisons)
        video_key_a = video_key_map[VideoYoutubeId(row[:video_a])]
        video_key_b = video_key_map[VideoYoutubeId(row[:video_b])]
        score = row[:score] / 10
        push!(comparisons, ComparisonSample(
            video_key_a,
            video_key_b,
            score,
        ))
    end
    comparisons
end

function load_tournesol_dataset(
    comparisons_file :: String,
    metadata_file :: String,
    username :: String;
    limit :: Union{Int, Nothing}
) :: TournesolDataset
    raw_comparisons = load_raw_comparisons(
        comparisons_file,
        username,
        "largely_recommended";
        limit=limit
    )
    video_key_map = video_key_map_from_raw_comparisons(raw_comparisons)
    raw_metadata = load_raw_metadata(metadata_file)
    metadata = metadata_from_raw_data(collect(keys(video_key_map)), raw_metadata)
    comparisons = samples_from_raw_comparisons(raw_comparisons, video_key_map)
    TournesolDataset(
        comparisons, 
        video_key_map,
        metadata,
    )
end

"""
    Embeddings
"""

abstract type Embeddings end

struct ChannelEmbeddings <: Embeddings
    embeddings :: Matrix{Float64}
    channel_key_map :: Bijection{ChannelId, Int}
    video_key_map :: Bijection{VideoYoutubeId, Int}
end

function channel_embedding(dataset :: TournesolDataset) :: Embeddings
    vkm = video_key_map(dataset)
    ckm = channel_key_map(dataset)

    embeddings = zeros(length(ckm), length(vkm))
    for m in values(dataset.metadata)
        video_key = vkm[video_id(m)]
        channel_key = ckm[channel_id(m)]
        embeddings[channel_key, video_key] = 1
    end

    ChannelEmbeddings(
        embeddings,
        ckm,
        vkm,
    )
end

function matrix(channel_embedding :: ChannelEmbeddings) :: Matrix{Float64}
    channel_embedding.embeddings
end

struct IdentityEmbeddings <: Embeddings
    n_videos :: Int
end

function identity_embeddings(dataset :: TournesolDataset) :: Embeddings
    vkm = video_key_map(dataset)
    IdentityEmbeddings(length(vkm))
end

function matrix(identity_embeddings :: IdentityEmbeddings) :: Matrix{Float64}
    n = identity_embeddings.n_videos
    Matrix(I, n, n)
end

"""
    Model builders
"""

function gbt_model(dataset :: TournesolDataset, sigma :: Float64) :: LGBTDModel
    n = n_videos(dataset)
    LGBTDModel(
        RootLawUniform(1.0),
        matrix(identity_embeddings(dataset)),
        sigma,
        zeros(n, n)
    )
end

function gbtd_channel_model(dataset :: TournesolDataset, sigma :: Float64, lambda :: Float64) :: LGBTDModel
    n = n_videos(dataset)
    LGBTDModel(
        RootLawUniform(1.0),
        vcat(matrix(channel_embedding(dataset)), lambda * Matrix(I, n, n)),
        sigma,
        zeros(n,n),
    )
end

"""
    Validation
    with MLBase.jl <https://mlbasejl.readthedocs.io/en/latest/crossval.html>
"""

function compare_from(
    _ :: RootLawUniform,
    score_difference :: Float64,
    ) :: Float64
    """
        P(c|s) = (1/2) 1{-1 ≤ c ≤ 1} exp(cs) exp(-Phi(s))
        Phi(s) = log ∫ (1/2) 1{-1 ≤ c ≤ 1} exp(cs) dc
               = log sinh(s)/s
        E(c | s) = Phi'(s)
                 = (cosh(s)/s - sinh(s)/s^2) / (sinh(s) / s)
                 = coth(s) - 1/s
                 = s/3 - s^3/45 + (2 s^5)/945 + O(s^7)
    """
    if abs(score_difference) < 1e-1
        score_difference/3 - score_difference^3/45 + 2 * score_difference^5 / 945
    else
        coth(score_difference) - (1/score_difference)
    end
end

function empirical_risk(root_law :: RootLaw, result :: LGBTDResult, comparisons :: Vector{ComparisonSample}) :: Float64
    predicted = [
        compare_from(
            root_law,
            result.scores[c.a] - result.scores[c.b]
            )
        for c in comparisons
    ]
    actual = [
        c.r
        for c in comparisons
    ]
    mean((predicted - actual).^2)
end


function val_mk_estfun(model :: LGBTDModel, dataset :: TournesolDataset)
    train_indices -> begin
        result = infer(model, dataset.comparisons[train_indices])
        result
    end
end

function val_mk_evalfun(model :: LGBTDModel, dataset :: TournesolDataset)
    (result, test_indices) -> begin
        empirical_risk(model.root_law, result, dataset.comparisons[test_indices])
    end
end

"""
    Experiment
"""

struct Experiment4
    # a unique name for this experiment
    xp_id :: String

    # tournesol user to consider
    username :: String
    limit :: Union{Int, Nothing} # number of user comparisons to consider

    # filepaths to data
    comparisons_file :: String
    metadata_file :: String
    
    # Cross validation Kfold(limit, k)
    n_folds :: Int
    
    # GBTD model hyperparameters
    sigma :: Float64
    lambda :: Float64
end

function Base.show(io :: IO, description :: Experiment4)
    println(io, "username: $(description.username)")
    println(io, "limit: $(description.limit)")
    println(io, "comparisons_file: $(description.comparisons_file)")
    println(io, "metadata_file: $(description.metadata_file)")
    println(io, "n_folds: $(description.n_folds)")
    println(io, "sigma: $(description.sigma)")
    println(io, "lambda: $(description.lambda)")
end

struct Experiment4Results
    description :: Experiment4
    models :: Dict{String, LGBTDModel}
    result :: Dict{String, Vector{Float64}}
end

function run(xp :: Experiment4)
    # prepare
    Random.seed!(42) # fix the seed
    dataset = load_tournesol_dataset(
        xp.comparisons_file,
        xp.metadata_file,
        xp.username;
        limit=xp.limit
    )
    models = Dict(
            "gbt" => gbt_model(dataset, xp.sigma),
            "gbtd" => gbtd_channel_model(dataset, xp.sigma, xp.lambda),
        )
    xp_id = xp.xp_id

    # run
    result = Dict{String, Vector{Float64}}()
    for (model_name, model) in models
        result[model_name] = cross_validate(
            val_mk_estfun(model, dataset),
            val_mk_evalfun(model, dataset),
            xp.limit,
            Kfold(xp.limit, xp.n_folds),
        )
    end
    serialize(datadir("experiment4/$xp_id.obj"), Experiment4Results(
        xp,
        models,
        result,
    ))
end

function do_plot(data :: Experiment4Results)
    points = data.result
    n_labels = length(keys(points))
    labels = reshape([k for (k,_) in points], (1, n_labels))
    values = [v for (_,v) in points]
    plt = plot()
    ylims = (
        1.1 * min(0, minimum(minimum(values))),
        1.1 * max(0, maximum(maximum(values))),
    )
    boxplot!(labels, values; legend=false, outliers=false, ylims=ylims)
    ylabel!("empirical risk")
    plt
end

"""
On my machine

for one pass (n_folds=1)
    limit   time (s)
    100     2.8
    500     13
    1000    138
"""

xp4 = Experiment4(
    "blueberry",
    "emmanuel.chambost",
    1000,
    datadir("tournesol/comparisons.csv"),
    datadir("entity_metadata_2024_01_16.csv"),
    10,
    1.0,
    1.0,
)

