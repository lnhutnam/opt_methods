using Random, LinearAlgebra
include("regularizer.jl")

"""
    Oracle

Abstract type for all optimization objectives. Subtypes should implement
methods for computing objective values, gradients, and Hessians.
Supports l1 and l2 regularization.
"""
abstract type Oracle end

mutable struct BaseOracle <: Oracle
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

    function BaseOracle(; l1=0.0, l2=0.0, l2_in_prox=false, regularizer=nothing, seed=42)
        if l1 < 0.0
            error("Invalid value for l1 regularization: $l1")
        end
        if l2 < 0.0
            error("Invalid value for l2 regularization: $l2")
        end
        if l2 == 0.0 && l2_in_prox
            @warn "The value of l2 is set to 0, so l2_in_prox is changed to false."
            l2_in_prox = false
        end

        l2_actual = l2_in_prox ? 0.0 : l2

        if (l1 > 0 || (l2 > 0 && l2_in_prox)) && regularizer === nothing
            l2_prox = l2_in_prox ? l2 : 0.0
            regularizer = Regularizer(l1=l1, l2=l2_prox)
        end

        rng = MersenneTwister(seed)

        new(l1, l2_actual, l2_in_prox, nothing, Inf, regularizer, seed, rng,
            nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

function set_seed!(oracle::Oracle, seed::Int)
    oracle.seed = seed
    oracle.rng = MersenneTwister(seed)
end

function value(oracle::Oracle, x::Vector{Float64})
    val = _value(oracle, x)
    if oracle.regularizer !== nothing
        val += oracle.regularizer(x)
    end
    if val < oracle.f_opt
        oracle.x_opt = copy(x)
        oracle.f_opt = val
    end
    return val
end

# These functions need to be implemented by concrete types
function _value(oracle::Oracle, x::Vector{Float64})
    error("_value method must be implemented by $(typeof(oracle))")
end

function gradient(oracle::Oracle, x::Vector{Float64})
    error("gradient method must be implemented by $(typeof(oracle))")
end

function hessian(oracle::Oracle, x::Vector{Float64})
    error("hessian method must be implemented by $(typeof(oracle))")
end

function hess_vec_prod(oracle::Oracle, x::Vector{Float64}, v::Vector{Float64};
                       grad_dif=false, eps=nothing)
    error("hess_vec_prod method must be implemented by $(typeof(oracle))")
end

function smoothness(oracle::Oracle)
    error("smoothness property must be implemented by $(typeof(oracle))")
end

function max_smoothness(oracle::Oracle)
    error("max_smoothness property must be implemented by $(typeof(oracle))")
end

function average_smoothness(oracle::Oracle)
    error("average_smoothness property must be implemented by $(typeof(oracle))")
end

function batch_smoothness(oracle::Oracle, batch_size::Int)
    error("batch_smoothness method must be implemented by $(typeof(oracle))")
end

# Static utility functions
function inner_prod(x::Vector{Float64}, y::Vector{Float64})
    return dot(x, y)
end

function outer_prod(x::Vector{Float64}, y::Vector{Float64})
    return x * y'
end

function is_equal(x::Vector{Float64}, y::Vector{Float64})
    # Check if sizes match first
    if length(x) != length(y)
        return false
    end
    return x ≈ y
end