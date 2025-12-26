using LinearAlgebra, SparseArrays

include("utils.jl")

"""
    Regularizer(; l1=0.0, l2=0.0, coef=nothing)

Regularization oracle with proximal operator support.
Implements combined L1 and L2 regularization: l1*||x||₁ + (l2/2)*||x||₂²

# Arguments
- `l1::Float64=0.0`: L1 regularization coefficient
- `l2::Float64=0.0`: L2 regularization coefficient
- `coef::Union{Vector{Float64},Nothing}=nothing`: Optional element-wise coefficients
"""
mutable struct Regularizer
    l1::Float64
    l2::Float64
    coef::Union{Vector{Float64}, Nothing}

    function Regularizer(; l1=0.0, l2=0.0, coef=nothing)
        new(l1, l2, coef)
    end
end

# Make Regularizer callable
function (reg::Regularizer)(x::Vector{Float64})
    return value(reg, x)
end

function value(reg::Regularizer, x::Vector{Float64})
    return reg.l1 * safe_sparse_norm(x, 1) + reg.l2/2 * safe_sparse_norm(x, 2)^2
end

function prox_l1(reg::Regularizer, x::AbstractVector, lr::Float64)
    abs_x = abs.(x)
    if issparse(x)
        prox_res = abs_x .- min.(abs_x, reg.l1 * lr)
        dropzeros!(prox_res)
        prox_res = prox_res .* sign.(x)
    else
        prox_res = abs_x .- min.(abs_x, reg.l1 * lr)
        prox_res .*= sign.(x)
    end
    return prox_res
end

function prox_l2(reg::Regularizer, x::AbstractVector, lr::Float64)
    return x ./ (1 + lr * reg.l2)
end

"""
    prox(reg::Regularizer, x::AbstractVector, lr::Float64)

Compute the proximal operator for combined L1 and L2 regularization.

The proximal operator of l1||x||₁ + (l2/2)||x||² is computed by
sequentially applying prox_l1 followed by prox_l2.
"""
function prox(reg::Regularizer, x::AbstractVector, lr::Float64)
    prox_l1_result = prox_l1(reg, x, lr)
    return prox_l2(reg, prox_l1_result, lr)
end