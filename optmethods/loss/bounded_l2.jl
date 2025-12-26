using LinearAlgebra, SparseArrays
include("regularizer.jl")

"""
The bounded l2 regularization is equal to
    R(x) = sum_{i=1}^d x_i^2 / (x_i^2 + 1)
where x=(x_1, ..., x_d) is from R^d. This penalty is attractive for benchmarking
purposes since it is smooth (has Lipschitz gradient) and nonconvex.

See
    https://arxiv.org/pdf/1905.05920.pdf
    https://arxiv.org/pdf/1810.10690.pdf
for examples of using this penalty for benchmarking.
"""
mutable struct BoundedL2Regularizer
    coef::Float64

    function BoundedL2Regularizer(coef::Float64)
        new(coef)
    end
end

function value(reg::BoundedL2Regularizer, x::AbstractVector, x2=nothing)
    if !issparse(x)
        if x2 === nothing
            x2 = x .* x
        end
        return reg.coef * 0.5 * sum(x2 ./ (x2 .+ 1))
    else
        if x2 === nothing
            x2 = x .* x
        end
        ones_where_nonzero = sign.(x2)
        return reg.coef * 0.5 * sum(x2 ./ (x2 .+ ones_where_nonzero))
    end
end

function prox(reg::BoundedL2Regularizer, x::AbstractVector, lr=nothing)
    error("Exact proximal operator for bounded l2 does not exist. Consider using gradients.")
end

function grad(reg::BoundedL2Regularizer, x::AbstractVector)
    if !issparse(x)
        return reg.coef .* x ./ (x.^2 .+ 1).^2
    else
        ones_where_nonzero = abs.(sign.(x))
        x2_plus_one = x .* x .+ ones_where_nonzero
        denominator = x2_plus_one .* x2_plus_one
        return reg.coef .* x .* (ones_where_nonzero ./ denominator)
    end
end

function smoothness(reg::BoundedL2Regularizer)
    return reg.coef
end

# Make it callable like the original Regularizer
function (reg::BoundedL2Regularizer)(x::AbstractVector)
    return value(reg, x)
end