using LinearAlgebra
include("base_line_search.jl")

"""
This line search estimates the Hessian Lipschitz constant for the Global Regularized Newton.
See the following paper for the details and convergence proof:
    "Regularized Newton Method with Global O(1/k^2) Convergence"
    https://arxiv.org/abs/2112.02089
For consistency with other line searches, 'lr' parameter is used to denote the inverse of regularization.
Arguments:
    decrease_reg (boolean, optional): multiply the previous regularization parameter by 1/backtracking (default: true)
    backtracking (float, optional): constant by which the current regularization is divided (default: 0.5)
"""
mutable struct RegularizedNewtonLineSearch <: LineSearch
    # Inherit from BaseLineSearch
    lr0::Float64
    lr::Float64
    count_first_it::Bool
    count_last_it::Bool
    it::Int
    it_max::Int
    tolerance::Float64

    # Set during optimization
    optimizer
    loss
    use_prox::Bool

    # RegNewtonLS-specific parameters
    decrease_reg::Bool
    backtracking::Float64
    H0::Union{Float64, Nothing}
    H::Float64
    attempts::Int
    f_prev::Union{Float64, Nothing}
    f_new::Float64

    function RegularizedNewtonLineSearch(; decrease_reg=true, backtracking=0.5,
                                        H0=nothing, kwargs...)
        base = BaseLineSearch(; kwargs...)
        H = H0 === nothing ? 1.0 : H0
        new(base.lr0, base.lr, base.count_first_it, base.count_last_it,
            base.it, base.it_max, base.tolerance, base.optimizer, base.loss, base.use_prox,
            decrease_reg, backtracking, H0, H, 0, nothing, 0.0)
    end
end

function condition(ls::RegularizedNewtonLineSearch, x_new::Vector{Float64}, x::Vector{Float64},
                  grad::Vector{Float64}, identity_coef::Float64)
    if ls.f_prev === nothing
        ls.f_prev = value(ls.loss, x)
    end

    ls.f_new = value(ls.loss, x_new)
    r = norm(x_new .- x)
    condition_f = ls.f_new <= ls.f_prev - 2/3 * identity_coef * r^2

    grad_new = gradient(ls.loss, x_new)
    condition_grad = norm(grad_new) <= 2 * identity_coef * r

    ls.attempts = (condition_f && condition_grad) ? 0 : ls.attempts + 1
    return condition_f && condition_grad
end

function (ls::RegularizedNewtonLineSearch)(x::Vector{Float64}, grad::Vector{Float64}, hess::Matrix{Float64})
    if ls.decrease_reg
        ls.H *= ls.backtracking
    end

    grad_norm = norm(grad)
    identity_coef = sqrt(ls.H * grad_norm)

    # Solve (H + σI)p = -∇f where σ = identity_coef
    regularized_hess = hess + identity_coef * I(size(hess, 1))
    x_new = x .- regularized_hess \ grad

    condition_met = condition(ls, x_new, x, grad, identity_coef)
    ls.it += it_per_call(ls)
    it_extra = 0
    it_max = min(ls.it_max, ls.optimizer.ls_it_max - ls.it)

    while !condition_met && it_extra < it_max
        ls.H /= ls.backtracking
        identity_coef = sqrt(ls.H * grad_norm)
        regularized_hess = hess + identity_coef * I(size(hess, 1))
        x_new = x .- regularized_hess \ grad
        condition_met = condition(ls, x_new, x, grad, identity_coef)
        it_extra += 1

        if ls.backtracking / ls.H == 0
            break
        end
    end

    ls.f_prev = ls.f_new
    ls.it += it_extra
    ls.lr = 1 / identity_coef
    return x_new
end

function reset!(ls::RegularizedNewtonLineSearch, optimizer)
    reset!(ls, optimizer)  # Call base reset
    ls.f_prev = nothing
end