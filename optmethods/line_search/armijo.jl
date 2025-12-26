include("base_line_search.jl")

"""
Armijo line search with optional resetting of the initial stepsize
at each iteration. If resetting is used, the previous value is optionally
multiplied by 1/backtracking and used as the first stepsize to try at the
new iteration. Otherwise, it starts with the maximal stepsize.
Arguments:
    armijo_const (float, optional): proportionality constant for the armijo condition (default: 0.5)
    start_with_prev_lr (boolean, optional): initialize lr with the previous value (default: true)
    increase_lr (boolean, optional): multiply the previous lr by 1/backtracking (default: true)
    backtracking (float, optional): constant by which the current stepsize is multiplied (default: 0.5)
"""
mutable struct ArmijoLineSearch <: LineSearch
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

    # Armijo-specific parameters
    armijo_const::Float64
    start_with_prev_lr::Bool
    increase_lr::Bool
    backtracking::Float64
    x_prev::Union{Vector{Float64}, Nothing}
    val_prev::Union{Float64, Nothing}
    current_value::Float64

    function ArmijoLineSearch(; armijo_const=0.5, start_with_prev_lr=true,
                             increase_lr=true, backtracking=0.5, kwargs...)
        base = BaseLineSearch(; kwargs...)
        new(base.lr0, base.lr, base.count_first_it, base.count_last_it,
            base.it, base.it_max, base.tolerance, base.optimizer, base.loss, base.use_prox,
            armijo_const, start_with_prev_lr, increase_lr, backtracking,
            nothing, nothing, 0.0)
    end
end

function condition(ls::ArmijoLineSearch, gradient::Vector{Float64}, x::Vector{Float64}, x_new::Vector{Float64})
    value_new = value(ls.loss, x_new)
    ls.val_prev = value_new
    descent = ls.armijo_const * inner_prod(x .- x_new, gradient)
    return value_new <= ls.current_value - descent + ls.tolerance
end

function (ls::ArmijoLineSearch)(; x=nothing, x_new=nothing, gradient=nothing, direction=nothing)
    if gradient === nothing
        gradient = ls.optimizer.grad
    end
    if x === nothing
        x = ls.optimizer.x
    end
    if direction === nothing
        direction = (x_new .- x) ./ ls.lr
    end

    if ls.start_with_prev_lr
        ls.lr = ls.increase_lr ? ls.lr / ls.backtracking : ls.lr
    else
        ls.lr = ls.lr0
    end

    if x_new === nothing
        x_new = x .+ ls.lr .* direction
    end

    if ls.x_prev !== nothing && is_equal(x, ls.x_prev)
        ls.current_value = ls.val_prev
    else
        ls.current_value = value(ls.loss, x)
    end

    armijo_condition_met = condition(ls, gradient, x, x_new)
    it_extra = 0
    it_max = min(ls.it_max, ls.optimizer.ls_it_max - ls.it)

    while !armijo_condition_met && it_extra < it_max
        ls.lr *= ls.backtracking
        x_new = x .+ ls.lr .* direction
        armijo_condition_met = condition(ls, gradient, x, x_new)
        it_extra += 1
    end

    ls.x_prev = copy(x_new)
    ls.it += it_per_call(ls) + it_extra
    return x_new
end