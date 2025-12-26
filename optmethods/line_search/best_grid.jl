include("base_line_search.jl")

"""
Find the best stepsize either over values {lr_max * backtracking^pow: pow=0, 1, ...} or
    over {lr * backtracking^(1 - pow): pow=0, 1, ...} where lr is the previous value
Arguments:
    lr_max (float, optional): the maximal stepsize, useful for second-order
        and quasi-Newton methods (default: Inf)
    functional (boolean, optional): use functional values to check optimality.
        Otherwise, gradient norm is used (default: true)
    start_with_prev_lr (boolean, optional): initialize lr with the previous value (default: false)
    increase_lr (boolean, optional): multiply the previous lr by 1/backtracking (default: true)
    increase_many_times (boolean, optional): multiply the lr by 1/backtracking until it's the best (default: true)
    backtracking (float, optional): constant to multiply the estimated stepsize by (default: 0.5)
"""
mutable struct BestGridLineSearch <: LineSearch
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

    # BestGrid-specific parameters
    lr_max::Float64
    functional::Bool
    start_with_prev_lr::Bool
    increase_lr::Bool
    increase_many_times::Bool
    backtracking::Float64
    x_prev::Union{Vector{Float64}, Nothing}
    val_prev::Union{Float64, Nothing}
    current_value::Float64

    function BestGridLineSearch(; lr_max=Inf, functional=true, start_with_prev_lr=false,
                               increase_lr=true, increase_many_times=true,
                               backtracking=0.5, kwargs...)
        base = BaseLineSearch(; kwargs...)
        new(base.lr0, base.lr, base.count_first_it, base.count_last_it,
            base.it, base.it_max, base.tolerance, base.optimizer, base.loss, base.use_prox,
            lr_max, functional, start_with_prev_lr, increase_lr, increase_many_times,
            backtracking, nothing, nothing, 0.0)
    end
end

function condition(ls::BestGridLineSearch, proposed_value::Float64, proposed_next::Float64)
    return proposed_value <= proposed_next + ls.tolerance
end

function metric_value(ls::BestGridLineSearch, x::Vector{Float64})
    if ls.functional
        return value(ls.loss, x)
    else
        return norm(gradient(ls.loss, x))
    end
end

function (ls::BestGridLineSearch)(; x=nothing, x_new=nothing, direction=nothing, gradient=nothing)
    if x === nothing
        x = ls.optimizer.x
    end

    if direction === nothing
        direction = x_new .- x
        ls.lr = 1.0
    elseif ls.start_with_prev_lr
        ls.lr = ls.increase_lr ? ls.lr / ls.backtracking : ls.lr
        ls.lr = min(ls.lr, ls.lr_max)
    else
        ls.lr = ls.lr0
    end

    if x_new === nothing
        x_new = x .+ ls.lr .* direction
    end

    if ls.x_prev !== nothing && is_equal(x, ls.x_prev)
        ls.current_value = ls.val_prev
    else
        ls.current_value = metric_value(ls, x)
    end

    it_extra = 0
    proposed_value = metric_value(ls, x_new)
    need_to_decrease_lr = proposed_value > ls.current_value

    if !need_to_decrease_lr
        x_next = x .+ ls.lr .* ls.backtracking .* direction
        proposed_next = metric_value(ls, x_next)
        if !condition(ls, proposed_value, proposed_next)
            need_to_decrease_lr = true
            ls.lr *= ls.backtracking
            proposed_value = proposed_next
        end
        it_extra += 1
    end

    found_best = !need_to_decrease_lr && !ls.increase_many_times
    it_max = min(ls.it_max, ls.optimizer.ls_it_max - ls.it)

    while !found_best && it_extra < it_max
        if need_to_decrease_lr
            lr_next = ls.lr * ls.backtracking
        else
            lr_next = min(ls.lr / ls.backtracking, ls.lr_max)
        end

        x_next = x .+ lr_next .* direction
        proposed_next = metric_value(ls, x_next)
        found_best = condition(ls, proposed_value, proposed_next)
        it_extra += 1

        if !found_best || it_extra == ls.it_max
            ls.lr = lr_next
            proposed_value = proposed_next
        end
    end

    x_new = x .+ ls.lr .* direction
    ls.val_prev = proposed_value
    ls.x_prev = copy(x_new)
    ls.it += it_per_call(ls) + it_extra
    return x_new
end