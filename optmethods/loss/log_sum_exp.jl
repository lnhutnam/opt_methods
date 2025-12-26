using LinearAlgebra, Statistics, Random
# using SpecialFunctions  # Optional - install with: using Pkg; Pkg.add("SpecialFunctions")

include("loss_oracle.jl")

"""
Logarithm of the sum of exponentials plus two optional quadratic terms:
    log(sum_{i=1}^n exp(<a_i, x>-b_i)) + 1/2*||Ax - b||^2 + l2/2*||x||^2
See, for instance,
    https://arxiv.org/pdf/2002.00657.pdf
    https://arxiv.org/pdf/2002.09403.pdf
for examples of using the objective to benchmark second-order methods.

Due to the potential under- and overflow, log-sum-exp and softmax
functions might be unstable. This implementation has not been tested
for stability and an alternative choice of functions may lead to
more precise results. See
    https://academic.oup.com/imajna/advance-article/doi/10.1093/imanum/draa038/5893596
for a discussion of possible ways to increase stability.

Sparse matrices are currently not supported as it is not clear if it is relevant to practice.

Arguments:
    max_smoothing (float, optional): the smoothing constant of the log-sum-exp term
    least_squares_term (bool, optional): add term 0.5*||Ax-b||^2 to the objective (default: false)
"""
mutable struct LogSumExp <: Oracle
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

    # Specific to LogSumExp
    max_smoothing::Float64
    least_squares_term::Bool
    A::Matrix{Float64}
    b::Vector{Float64}
    n::Int
    dim::Int
    store_mat_vec_prod::Bool
    store_softmax::Bool
    x_last_mv::Union{Vector{Float64}, Float64}
    x_last_soft::Union{Vector{Float64}, Float64}
    _mat_vec_prod::Vector{Float64}
    _softmax::Vector{Float64}

    function LogSumExp(; max_smoothing=1.0, least_squares_term=false, A=nothing, b=nothing,
                      n=nothing, dim=nothing, store_mat_vec_prod=true, store_softmax=true, kwargs...)
        # Create base oracle
        base = BaseOracle(; kwargs...)

        if b === nothing && n !== nothing
            b = randn(base.rng, n) .- 1.0
        else
            b = Vector{Float64}(b)
        end

        if A === nothing && n !== nothing && dim !== nothing
            A_temp = 2.0 * rand(base.rng, n, dim) .- 1.0
            # Need to create temporary instance to compute gradient at zero
            temp_lse = new(base.l1, base.l2, base.l2_in_prox, base.x_opt, base.f_opt,
                          base.regularizer, base.seed, base.rng,
                          base._smoothness, base._max_smoothness, base._ave_smoothness,
                          base._importance_probs, base._individ_smoothness, base._hessian_lipschitz,
                          max_smoothing, least_squares_term, A_temp, b, n, dim,
                          false, false, 0.0, 0.0, zeros(n), zeros(n))

            # Adjust A by subtracting gradient at zero
            grad_zero = gradient(temp_lse, zeros(dim))
            A = A_temp .- grad_zero'

            # Compute value at zero to set internal state
            _value(temp_lse, zeros(dim))
        else
            A = Matrix{Float64}(A)
        end

        n, dim = size(A)
        x_last_mv = 0.0
        x_last_soft = 0.0
        _mat_vec_prod = zeros(n)
        _softmax = zeros(n)

        new(base.l1, base.l2, base.l2_in_prox, base.x_opt, base.f_opt,
            base.regularizer, base.seed, base.rng,
            base._smoothness, base._max_smoothness, base._ave_smoothness,
            base._importance_probs, base._individ_smoothness, base._hessian_lipschitz,
            max_smoothing, least_squares_term, A, b, n, dim,
            store_mat_vec_prod, store_softmax, x_last_mv, x_last_soft,
            _mat_vec_prod, _softmax)
    end
end

function _value(oracle::LogSumExp, x::Vector{Float64})
    Ax = mat_vec_product(oracle, x)
    regularization = oracle.l2 == 0 ? 0.0 : oracle.l2 / 2 * norm(x)^2

    if oracle.least_squares_term
        regularization += 0.5 * norm(Ax)^2
    end

    # Stable log-sum-exp computation
    z = (Ax .- oracle.b) ./ oracle.max_smoothing
    return oracle.max_smoothing * logsumexp(z) + regularization
end

function gradient(oracle::LogSumExp, x::Vector{Float64})
    Ax = mat_vec_product(oracle, x)
    softmax_vals = softmax(oracle; x=x, Ax=Ax)

    if oracle.least_squares_term
        grad = (softmax_vals .+ Ax)' * oracle.A
    else
        grad = softmax_vals' * oracle.A
    end

    if oracle.l2 == 0
        return vec(grad)
    else
        return vec(grad) .+ oracle.l2 .* x
    end
end

function hessian(oracle::LogSumExp, x::Vector{Float64})
    Ax = mat_vec_product(oracle, x)
    softmax_vals = softmax(oracle; x=x, Ax=Ax)

    hess1 = oracle.A' * Diagonal(softmax_vals ./ oracle.max_smoothing) * oracle.A
    grad = softmax_vals' * oracle.A
    hess2 = -vec(grad) * vec(grad)' ./ oracle.max_smoothing

    return hess1 + hess2 + oracle.l2 * I(oracle.dim)
end

function mat_vec_product(oracle::LogSumExp, x::Vector{Float64})
    if oracle.store_mat_vec_prod && is_equal(x, oracle.x_last_mv)
        return oracle._mat_vec_prod
    end

    Ax = oracle.A * x

    if oracle.store_mat_vec_prod
        oracle._mat_vec_prod = Ax
        oracle.x_last_mv = copy(x)
    end

    return Ax
end

function softmax(oracle::LogSumExp; x=nothing, Ax=nothing)
    if x === nothing && Ax === nothing
        error("Either x or Ax must be provided to compute softmax.")
    end

    if oracle.store_softmax && x !== nothing && is_equal(x, oracle.x_last_soft)
        return oracle._softmax
    end

    if Ax === nothing
        Ax = mat_vec_product(oracle, x)
    end

    # Stable softmax computation
    z = (Ax .- oracle.b) ./ oracle.max_smoothing
    z_max = maximum(z)
    exp_z = exp.(z .- z_max)
    softmax_vals = exp_z ./ sum(exp_z)

    if oracle.store_softmax && x !== nothing
        oracle._softmax = softmax_vals
        oracle.x_last_soft = copy(x)
    end

    return softmax_vals
end

function smoothness(oracle::LogSumExp)
    if oracle._smoothness !== nothing
        return oracle._smoothness
    end

    matrix_coef = 1 + oracle.least_squares_term

    if oracle.dim > 20000 && oracle.n > 20000
        @warn "The matrix is too large to estimate the smoothness constant, so Frobenius estimate is used instead."
        oracle._smoothness = matrix_coef * norm(oracle.A, "fro")^2 + oracle.l2
    else
        if oracle.n == 1 || oracle.dim == 1
            sing_val_max = norm(oracle.A)
        else
            # Use SVD for small matrices, or estimate for large ones
            sing_val_max = svdvals(oracle.A)[1]
        end
        oracle._smoothness = matrix_coef * sing_val_max^2 + oracle.l2
    end

    return oracle._smoothness
end

function hessian_lipschitz(oracle::LogSumExp)
    if oracle._hessian_lipschitz === nothing
        row_norms = [norm(oracle.A[i, :]) for i in 1:oracle.n]
        max_row_norm = maximum(row_norms)
        oracle._hessian_lipschitz = 2 * max_row_norm / oracle.max_smoothing * smoothness(oracle)
    end
    return oracle._hessian_lipschitz
end

# Utility function for stable log-sum-exp
function logsumexp(x::Vector{Float64})
    x_max = maximum(x)
    return x_max + log(sum(exp.(x .- x_max)))
end