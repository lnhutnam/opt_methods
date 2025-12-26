include("base_line_search.jl")

"""
This line search implements procedure (4.9) from the following paper by Nesterov:
    http://www.optimization-online.org/DB_FILE/2007/09/1784.pdf
Arguments:
    mu (float, optional): strong convexity constant (default: 0.0)
    start_with_prev_lr (boolean, optional): initialize lr with the previous value (default: true)
    backtracking (float, optional): constant by which the current stepsize is multiplied (default: 0.5)
    start_with_small_momentum (bool, optional): momentum gradually increases.
        Only used if mu>0 (default: true)
"""
mutable struct NesterovArmijoLineSearch <: LineSearch
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

    # NestArmijo-specific parameters
    mu::Float64
    start_with_prev_lr::Bool
    backtracking::Float64
    start_with_small_momentum::Bool
    global_calls::Int

    function NesterovArmijoLineSearch(; mu=0.0, start_with_prev_lr=true,
                                     backtracking=0.5, start_with_small_momentum=true, kwargs...)
        base = BaseLineSearch(; count_first_it=true, kwargs...)
        new(base.lr0, base.lr, base.count_first_it, base.count_last_it,
            base.it, base.it_max, base.tolerance, base.optimizer, base.loss, base.use_prox,
            mu, start_with_prev_lr, backtracking, start_with_small_momentum, 0)
    end
end

function condition(ls::NesterovArmijoLineSearch, y::Vector{Float64}, x_new::Vector{Float64})
    grad_new = gradient(ls.loss, x_new)
    return inner_prod(y .- x_new, grad_new) >= ls.lr * norm(grad_new)^2 - ls.tolerance
end

function (ls::NesterovArmijoLineSearch)(x::Vector{Float64}, v::Vector{Float64}, A::Float64)
    ls.global_calls += 1
    ls.lr = ls.start_with_prev_lr ? ls.lr / ls.backtracking : ls.lr0

    # Find `a` from quadratic equation a^2/(A+a) = 2*lr*(1 + mu*A)
    discriminant = (ls.lr * (1 + ls.mu * A))^2 + A * ls.lr * (1 + ls.mu * A)
    a = ls.lr * (1 + ls.mu * A) + sqrt(discriminant)

    if ls.start_with_small_momentum
        a_small = ls.lr + sqrt(ls.lr^2 + A * ls.lr)
        a = min(a, a_small)
    end

    y = (A .* x .+ a .* v) ./ (A + a)
    grad = gradient(ls.loss, y)
    x_new = y .- ls.lr .* grad
    nest_condition_met = condition(ls, y, x_new)

    it_extra = 0
    it_max = min(2 * ls.it_max, ls.optimizer.ls_it_max - ls.it)

    while !nest_condition_met && it_extra < it_max
        ls.lr *= ls.backtracking
        discriminant = (ls.lr * (1 + ls.mu * A))^2 + A * ls.lr * (1 + ls.mu * A)
        a = ls.lr * (1 + ls.mu * A) + sqrt(discriminant)

        if ls.start_with_small_momentum
            a_small = ls.lr + sqrt(ls.lr^2 + A * ls.lr)
            a = min(a, a_small)
        end

        y = (A / (A + a)) .* x .+ (a / (A + a)) .* v
        grad = gradient(ls.loss, y)
        x_new = y .- ls.lr .* grad
        nest_condition_met = condition(ls, y, x_new)
        it_extra += 2

        if ls.lr * ls.backtracking == 0
            break
        end
    end

    ls.it += it_per_call(ls) + it_extra
    return x_new, a
end

function reset!(ls::NesterovArmijoLineSearch, optimizer)
    reset!(ls, optimizer)  # Call base reset
    ls.global_calls = 0
end