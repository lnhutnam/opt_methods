using LinearAlgebra, SparseArrays, Statistics
# using SpecialFunctions  # Optional - install with: using Pkg; Pkg.add("SpecialFunctions")

include("loss_oracle.jl")
include("utils.jl")

# Helper functions for logistic regression
function sigmoid(x::Real)
    return 1.0 / (1.0 + exp(-x))
end

function sigmoid(x::AbstractArray)
    return sigmoid.(x)
end

"""
Compute the log-sigmoid function component-wise.
See http://fa.bianp.net/blog/2019/evaluate_logistic/ for more details.
"""
function logsig(x::AbstractArray)
    out = similar(x)
    for i in eachindex(x)
        if x[i] < -33
            out[i] = x[i]
        elseif -33 <= x[i] < -18
            out[i] = x[i] - exp(x[i])
        elseif -18 <= x[i] < 37
            out[i] = -log1p(exp(-x[i]))
        else  # x[i] >= 37
            out[i] = -exp(-x[i])
        end
    end
    return out
end

logsig(x::Real) = logsig([x])[1]

"""
Logistic regression oracle that returns loss values, gradients, Hessians,
their stochastic analogues as well as smoothness constants. Supports both
sparse and dense iterates.

The loss is defined as f(x) = 1/n sum_{i=1}^n log(cᵢ s(<aᵢ,x>)) + l₂/2 ||x||².
"""
mutable struct LogisticRegression <: Oracle
    # Inherit from Oracle
    l1::Float64
    l2::Float64
    l2_in_prox::Bool
    x_opt::Union{Vector{Float64}, Nothing}
    f_opt::Float64
    regularizer::Union{Regularizer, Nothing}
    seed::Int
    rng::AbstractRNG

    # Cached properties
    _smoothness::Union{Float64, Nothing}
    _max_smoothness::Union{Float64, Nothing}
    _ave_smoothness::Union{Float64, Nothing}
    _importance_probs::Union{Vector{Float64}, Nothing}
    _individ_smoothness::Union{Vector{Float64}, Nothing}
    _hessian_lipschitz::Union{Float64, Nothing}

    # Specific to LogisticRegression
    A::AbstractMatrix
    b::Vector{Float64}
    n::Int
    dim::Int
    store_mat_vec_prod::Bool
    x_last::Union{Vector{Float64}, Float64}
    _mat_vec_prod::Vector{Float64}

    function LogisticRegression(A, b; store_mat_vec_prod=true, kwargs...)
        # Create base oracle
        base = BaseOracle(; kwargs...)

        b = Vector{Float64}(b)
        b_unique = unique(b)

        # Handle label transformations
        if length(b_unique) == 1
            @warn "The labels have only one unique value."
        elseif length(b_unique) > 2
            error("The number of classes must be no more than 2 for binary classification.")
        end

        if length(b_unique) == 2 && !issetequal(b_unique, [0, 1])
            if issetequal(b_unique, [1, 2])
                println("The passed labels have values in the set {1, 2}. Changing them to {0, 1}")
                b = b .- 1
            elseif issetequal(b_unique, [-1, 1])
                println("The passed labels have values in the set {-1, 1}. Changing them to {0, 1}")
                b = (b .+ 1) ./ 2
            else
                println("Changing the labels from $(b[1]) to 1s and the rest to 0s")
                b = Float64.(b .== b[1])
            end
        end

        n, dim = size(A)
        x_last = Float64[]
        _mat_vec_prod = zeros(n)

        new(base.l1, base.l2, base.l2_in_prox, base.x_opt, base.f_opt,
            base.regularizer, base.seed, base.rng,
            base._smoothness, base._max_smoothness, base._ave_smoothness,
            base._importance_probs, base._individ_smoothness, base._hessian_lipschitz,
            A, b, n, dim, store_mat_vec_prod, x_last, _mat_vec_prod)
    end
end

function _value(oracle::LogisticRegression, x::Vector{Float64})
    Ax = mat_vec_product(oracle, x)
    regularization = oracle.l2 == 0 ? 0.0 : oracle.l2 / 2 * safe_sparse_norm(x)^2
    return mean(safe_sparse_multiply(1 .- oracle.b, Ax) .- logsig(Ax)) + regularization
end

function gradient(oracle::LogisticRegression, x::Vector{Float64})
    Ax = mat_vec_product(oracle, x)
    activation = sigmoid.(Ax)  # sigmoid function

    if oracle.l2 == 0
        grad = oracle.A' * (activation .- oracle.b) ./ oracle.n
    else
        grad = safe_sparse_add(oracle.A' * (activation .- oracle.b) ./ oracle.n, oracle.l2 .* x)
    end

    if issparse(x)
        grad = sparse(grad)
    end
    return grad
end

function stochastic_gradient(oracle::LogisticRegression, x::Vector{Float64};
                           idx=nothing, batch_size=1, replace=false, normalization=nothing,
                           importance_sampling=false, p=nothing, rng=nothing, return_idx=false)
    if batch_size === nothing || batch_size == oracle.n
        result = gradient(oracle, x)
        return return_idx ? (result, collect(1:oracle.n)) : result
    end

    if idx === nothing
        if rng === nothing
            rng = oracle.rng
        end
        if p === nothing && importance_sampling
            if oracle._importance_probs === nothing
                oracle._importance_probs = individ_smoothness(oracle)
                oracle._importance_probs ./= sum(oracle._importance_probs)
            end
            p = oracle._importance_probs
        end
        idx = rand(rng, 1:oracle.n, batch_size)  # Simplified - would need proper weighted sampling for p
    else
        batch_size = length(idx)
    end

    if normalization === nothing
        normalization = p === nothing ? batch_size : batch_size * p[idx] * oracle.n
    end

    A_idx = oracle.A[idx, :]
    Ax = A_idx * x
    if issparse(Ax)
        Ax = Array(Ax)
    end
    Ax = vec(Ax)

    activation = sigmoid.(Ax)
    error = (activation .- oracle.b[idx]) ./ normalization

    if length(error) == 1
        grad = oracle.l2 .* x .+ error[1] .* A_idx'
    else
        grad = oracle.l2 .* x .+ A_idx' * error
    end

    return return_idx ? (grad, idx) : grad
end

function hessian(oracle::LogisticRegression, x::Vector{Float64})
    Ax = mat_vec_product(oracle, x)
    activation = sigmoid.(Ax)
    weights = activation .* (1 .- activation)
    A_weighted = oracle.A' .* weights'
    return A_weighted * oracle.A ./ oracle.n .+ oracle.l2 .* I(oracle.dim)
end

function mat_vec_product(oracle::LogisticRegression, x::Vector{Float64})
    if oracle.store_mat_vec_prod && is_equal(x, oracle.x_last)
        return oracle._mat_vec_prod
    end

    Ax = oracle.A * x
    if issparse(Ax)
        Ax = Array(Ax)
    end
    Ax = vec(Ax)

    if oracle.store_mat_vec_prod
        oracle._mat_vec_prod = Ax
        oracle.x_last = copy(x)
    end

    return Ax
end

function smoothness(oracle::LogisticRegression)
    if oracle._smoothness !== nothing
        return oracle._smoothness
    end

    if oracle.dim > 20000 && oracle.n > 20000
        @warn "The matrix is too large to compute the smoothness constant, so an estimate is used instead."
        if issparse(oracle.A)
            estimate1 = norm(oracle.A, "fro")^2
            estimate2 = norm(oracle.A, 1) * norm(oracle.A, Inf)
        else
            estimate1 = norm(oracle.A, "fro")^2
            estimate2 = norm(oracle.A, 1) * norm(oracle.A, Inf)
        end
        oracle._smoothness = 0.25 * min(estimate1, estimate2) / oracle.n + oracle.l2
    else
        # Use largest singular value
        if issparse(oracle.A)
            sing_val_max = svds(oracle.A, nsv=1)[2][1]
        else
            sing_val_max = svdvals(oracle.A)[1]
        end
        oracle._smoothness = 0.25 * sing_val_max^2 / oracle.n + oracle.l2
    end

    return oracle._smoothness
end

function max_smoothness(oracle::LogisticRegression)
    if oracle._max_smoothness !== nothing
        return oracle._max_smoothness
    end
    max_squared_sum = maximum(sum(oracle.A.^2, dims=2))
    oracle._max_smoothness = 0.25 * max_squared_sum + oracle.l2
    return oracle._max_smoothness
end

function average_smoothness(oracle::LogisticRegression)
    if oracle._ave_smoothness !== nothing
        return oracle._ave_smoothness
    end
    ave_squared_sum = mean(sum(oracle.A.^2, dims=2))
    oracle._ave_smoothness = 0.25 * ave_squared_sum + oracle.l2
    return oracle._ave_smoothness
end

function individ_smoothness(oracle::LogisticRegression)
    if oracle._individ_smoothness !== nothing
        return oracle._individ_smoothness
    end
    oracle._individ_smoothness = [norm(oracle.A[i, :]) for i in 1:oracle.n]
    return oracle._individ_smoothness
end

function batch_smoothness(oracle::LogisticRegression, batch_size::Int)
    # Smoothness constant of stochastic gradients sampled in minibatches
    L = smoothness(oracle)
    L_max = max_smoothness(oracle)
    L_batch = oracle.n / (oracle.n - 1) * (1 - 1/batch_size) * L +
              (oracle.n / batch_size - 1) / (oracle.n - 1) * L_max
    return L_batch
end