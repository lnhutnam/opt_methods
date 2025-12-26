include("base_line_search.jl")

"""
Goldstein line search with optional resetting of the initial stepsize
at each iteration. Combines Armijo condition with an extra condition to make sure
that the stepsize is not too small. If resetting is used, the previous value
is used as the first stepsize to try at this iteration. Otherwise,
it starts with the maximal stepsize.
Arguments:
    goldstein_const (float, optional): proportionality constant for both conditions (default: 0.05)
    start_with_prev_lr (boolean, optional): sets the reset option from (default: true)
    backtracking (float, optional): constant by which the current stepsize is multiplied (default: 0.5)
"""
mutable struct GoldsteinLineSearch <: LineSearch
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

    # Goldstein-specific parameters
    goldstein_const::Float64
    start_with_prev_lr::Bool
    backtracking::Float64
    x_prev::Union{Vector{Float64}, Nothing}
    val_prev::Union{Float64, Nothing}
    current_value::Float64

    function GoldsteinLineSearch(; goldstein_const=0.05, start_with_prev_lr=true,
                                backtracking=0.5, kwargs...)
        base = BaseLineSearch(; kwargs...)
        new(base.lr0, base.lr, base.count_first_it, base.count_last_it,
            base.it, base.it_max, base.tolerance, base.optimizer, base.loss, base.use_prox,
            goldstein_const, start_with_prev_lr, backtracking,
            nothing, nothing, 0.0)
    end
end

function armijo_condition(ls::GoldsteinLineSearch, gradient::Vector{Float64}, x::Vector{Float64}, x_new::Vector{Float64})
    value_new = value(ls.loss, x_new)
    ls.x_prev = copy(x_new)
    ls.val_prev = value_new
    descent = ls.goldstein_const * inner_prod(x .- x_new, gradient)
    return value_new <= ls.current_value - descent + ls.tolerance
end

function goldstein_condition(ls::GoldsteinLineSearch, gradient::Vector{Float64}, x::Vector{Float64}, x_new::Vector{Float64})
    value_new = value(ls.loss, x_new)
    ls.x_prev = copy(x_new)
    ls.val_prev = value_new
    descent = (1 - ls.goldstein_const) * inner_prod(x .- x_new, gradient)
    return value_new >= ls.current_value - descent
end

function (ls::GoldsteinLineSearch)(; x=nothing, x_new=nothing, gradient=nothing, direction=nothing)
    if gradient === nothing
        gradient = ls.optimizer.grad
    end
    if x === nothing
        x = ls.optimizer.x
    end
    if direction === nothing
        direction = (x_new .- x) ./ ls.lr
    end

    ls.lr = ls.start_with_prev_lr ? ls.lr : ls.lr0

    if x_new === nothing
        x_new = x .+ ls.lr .* direction
    end

    if ls.x_prev !== nothing && is_equal(x, ls.x_prev)
        ls.current_value = ls.val_prev
    else
        ls.current_value = value(ls.loss, x)
    end

    armijo_cond = armijo_condition(ls, gradient, x, x_new)
    goldstein_cond = goldstein_condition(ls, gradient, x, x_new)
    it_extra = 0
    it_max = min(ls.it_max, ls.optimizer.ls_it_max - ls.it)

    # First satisfy Armijo condition by backtracking
    while !armijo_cond && it_extra < it_max
        ls.lr *= ls.backtracking
        x_new = x .+ ls.lr .* direction
        armijo_cond = armijo_condition(ls, gradient, x, x_new)
        it_extra += 1
    end

    # If Armijo was satisfied from the start, try to satisfy Goldstein condition
    if it_extra == 0
        while !goldstein_cond && it_extra < ls.it_max
            ls.lr /= ls.backtracking
            x_new = x .+ ls.lr .* direction
            goldstein_cond = goldstein_condition(ls, gradient, x, x_new)
            it_extra += 1
        end
    end

    ls.it += it_per_call(ls) + it_extra
    return x_new
end