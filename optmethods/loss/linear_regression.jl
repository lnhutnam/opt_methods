using LinearAlgebra, SparseArrays, Statistics

include("loss_oracle.jl")
include("utils.jl")

"""
    LinearRegression(A, b; store_mat_vec_prod=true, kwargs...)

Linear regression oracle implementing least squares loss.

Loss function: f(x) = (1/2n)||Ax - b||² + regularization

# Arguments
- `A::AbstractMatrix`: Design matrix (n × d)
- `b::Vector`: Target vector (n,)
- `store_mat_vec_prod::Bool=true`: Cache matrix-vector products for efficiency
- `kwargs...`: Regularization parameters (l1, l2, l2_in_prox) passed to BaseOracle

# Example
```julia
lr = LinearRegression(A, b, l2=0.01)
loss_val = value(lr, x)
grad = gradient(lr, x)
```
"""
mutable struct LinearRegression <: Oracle
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

    # Specific to LinearRegression
    A::AbstractMatrix
    b::Vector{Float64}
    n::Int
    dim::Int
    store_mat_vec_prod::Bool
    x_last::Union{Vector{Float64}, Float64}
    _mat_vec_prod::Vector{Float64}

    function LinearRegression(A, b; store_mat_vec_prod=true, kwargs...)
        # Create base oracle
        base = BaseOracle(; kwargs...)

        b = Vector{Float64}(b)
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

function _value(oracle::LinearRegression, x::Vector{Float64})
    z = mat_vec_product(oracle, x)
    residual = z .- oracle.b
    mse = 0.5 * mean(residual.^2)
    regularization = oracle.l2 == 0 ? 0.0 : oracle.l2 / 2 * safe_sparse_norm(x)^2
    return mse + regularization
end

function gradient(oracle::LinearRegression, x::Vector{Float64})
    z = mat_vec_product(oracle, x)
    residual = z .- oracle.b

    if oracle.l2 == 0
        grad = oracle.A' * residual ./ oracle.n
    else
        grad = safe_sparse_add(oracle.A' * residual ./ oracle.n, oracle.l2 .* x)
    end

    if issparse(x)
        grad = sparse(grad)
    end
    return grad
end

function hessian(oracle::LinearRegression, x::Vector{Float64})
    # For linear regression, Hessian is constant: A'A/n + λI
    hess = oracle.A' * oracle.A ./ oracle.n
    if oracle.l2 > 0
        hess += oracle.l2 * I(oracle.dim)
    end
    return hess
end

function stochastic_gradient(oracle::LinearRegression, x::Vector{Float64};
                           idx=nothing, batch_size=1, replace=false, normalization=nothing,
                           rng=nothing, return_idx=false)
    if batch_size === nothing || batch_size == oracle.n
        result = gradient(oracle, x)
        return return_idx ? (result, collect(1:oracle.n)) : result
    end

    if idx === nothing
        if rng === nothing
            rng = oracle.rng
        end
        idx = rand(rng, 1:oracle.n, batch_size)
    else
        batch_size = length(idx)
    end

    if normalization === nothing
        normalization = batch_size
    end

    A_idx = oracle.A[idx, :]
    z = A_idx * x
    residual = z .- oracle.b[idx]

    grad = oracle.l2 .* x .+ A_idx' * residual ./ normalization
    return return_idx ? (grad, idx) : grad
end

function mat_vec_product(oracle::LinearRegression, x::Vector{Float64})
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

function smoothness(oracle::LinearRegression)
    if oracle._smoothness !== nothing
        return oracle._smoothness
    end

    covariance = oracle.A' * oracle.A ./ oracle.n
    if issparse(covariance)
        covariance = Array(covariance)
    end

    oracle._smoothness = maximum(eigvals(covariance)) + oracle.l2
    return oracle._smoothness
end

function max_smoothness(oracle::LinearRegression)
    if oracle._max_smoothness !== nothing
        return oracle._max_smoothness
    end
    max_squared_sum = maximum(sum(oracle.A.^2, dims=2))
    oracle._max_smoothness = max_squared_sum + oracle.l2
    return oracle._max_smoothness
end

function average_smoothness(oracle::LinearRegression)
    if oracle._ave_smoothness !== nothing
        return oracle._ave_smoothness
    end
    ave_squared_sum = mean(sum(oracle.A.^2, dims=2))
    oracle._ave_smoothness = ave_squared_sum + oracle.l2
    return oracle._ave_smoothness
end

function individ_smoothness(oracle::LinearRegression)
    if oracle._individ_smoothness !== nothing
        return oracle._individ_smoothness
    end
    oracle._individ_smoothness = [norm(oracle.A[i, :]) for i in 1:oracle.n]
    return oracle._individ_smoothness
end

function batch_smoothness(oracle::LinearRegression, batch_size::Int)
    # Smoothness constant of stochastic gradients sampled in minibatches
    L = smoothness(oracle)
    L_max = max_smoothness(oracle)
    L_batch = oracle.n / (oracle.n - 1) * (1 - 1/batch_size) * L +
              (oracle.n / batch_size - 1) / (oracle.n - 1) * L_max
    return L_batch
end