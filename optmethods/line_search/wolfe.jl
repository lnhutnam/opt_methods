include("base_line_search.jl")

"""
Wolfe line search with optional resetting of the initial stepsize
at each iteration. If resetting is used, the previous value is used
as the first stepsize to try at this iteration. Otherwise, it starts
with the maximal stepsize.
Arguments:
    armijo_const (float, optional): proportionality constant for the armijo condition (default: 0.1)
    wolfe_const (float, optional): second proportionality constant for the wolfe condition (default: 0.9)
    strong (bool, optional): use strong Wolfe conditions (default: false)
    start_with_prev_lr (boolean, optional): sets the reset option from (default: true)
    backtracking (float, optional): constant by which the current stepsize is multiplied (default: 0.5)
"""
mutable struct WolfeLineSearch <: LineSearch
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

    # Wolfe-specific parameters
    armijo_const::Float64
    wolfe_const::Float64
    strong::Bool
    start_with_prev_lr::Bool
    backtracking::Float64
    x_prev::Union{Vector{Float64}, Nothing}
    val_prev::Union{Float64, Nothing}
    current_value::Float64

    function WolfeLineSearch(; armijo_const=0.1, wolfe_const=0.9, strong=false,
                            start_with_prev_lr=true, backtracking=0.5, kwargs...)
        base = BaseLineSearch(; kwargs...)
        new(base.lr0, base.lr, base.count_first_it, base.count_last_it,
            base.it, base.it_max, base.tolerance, base.optimizer, base.loss, base.use_prox,
            armijo_const, wolfe_const, strong, start_with_prev_lr, backtracking,
            nothing, nothing, 0.0)
    end
end

function armijo_condition(ls::WolfeLineSearch, gradient::Vector{Float64}, x::Vector{Float64}, x_new::Vector{Float64})
    value_new = value(ls.loss, x_new)
    ls.x_prev = copy(x_new)
    ls.val_prev = value_new
    descent = ls.armijo_const * inner_prod(x .- x_new, gradient)
    return value_new <= ls.current_value - descent + ls.tolerance
end

function curvature_condition(ls::WolfeLineSearch, grad_current::Vector{Float64}, x::Vector{Float64}, x_new::Vector{Float64})
    grad_new = gradient(ls.loss, x_new)
    curv_x = inner_prod(x .- x_new, grad_current)
    curv_x_new = inner_prod(x .- x_new, grad_new)

    if ls.strong
        curv_x, curv_x_new = abs(curv_x), abs(curv_x_new)
    end

    return curv_x_new <= ls.wolfe_const * curv_x + ls.tolerance
end

function (ls::WolfeLineSearch)(; x=nothing, x_new=nothing, gradient=nothing, direction=nothing)
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
    curvature_cond = curvature_condition(ls, gradient, x, x_new)
    it_extra = 0
    it_max = min(ls.it_max, ls.optimizer.ls_it_max - ls.it)

    # First satisfy Armijo condition by backtracking
    while !armijo_cond && it_extra < it_max
        ls.lr *= ls.backtracking
        x_new = x .+ ls.lr .* direction
        armijo_cond = armijo_condition(ls, gradient, x, x_new)
        it_extra += 1
    end

    # If Armijo was satisfied from the start, try to satisfy curvature condition
    if it_extra == 0
        while !curvature_cond && it_extra < it_max
            ls.lr /= ls.backtracking
            x_new = x .+ ls.lr .* direction
            curvature_cond = curvature_condition(ls, gradient, x, x_new)
            it_extra += 1
        end
    end

    ls.it += it_per_call(ls) + it_extra
    return x_new
end