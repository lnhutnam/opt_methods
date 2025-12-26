include("../optimizer.jl")
using Random

"""
    StochasticGradientDescent(loss; lr0=nothing, lr_max=Inf, lr_decay_coef=0.0,
                              lr_decay_power=1.0, it_start_decay=nothing,
                              batch_size=1, avoid_cache_miss=false,
                              importance_sampling=false, kwargs...)

Stochastic gradient descent with decreasing or constant learning rate.

# Arguments
- `loss::Oracle`: Optimization oracle
- `lr0::Union{Float64,Nothing}=nothing`: Initial learning rate (inverse smoothness estimate)
- `lr_max::Float64=Inf`: Maximum learning rate never to be exceeded
- `lr_decay_coef::Float64=0.0`: Coefficient for learning rate decay
  For strongly convex problems, use μ/2 where μ is the strong convexity constant
- `lr_decay_power::Float64=1.0`: Power for iteration exponentiation in decay
  For strongly convex problems, use 1.0
- `it_start_decay::Union{Int,Nothing}=nothing`: Iterations before decay starts
  Default: ~2.5% of it_max iterations with constant lr0
- `batch_size::Int=1`: Number of samples per iteration
- `avoid_cache_miss::Bool=false`: Sample adjacent indices for cache efficiency
  May slow convergence but speeds up iteration time
- `importance_sampling::Bool=false`: Use importance sampling
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
sgd = StochasticGradientDescent(loss, lr0=0.01, batch_size=32)
trace = run!(sgd, x0, it_max=10000)
```
"""
mutable struct StochasticGradientDescent
    optimizer::Optimizer
    lr0::Union{Float64, Nothing}
    lr::Float64
    lr_max::Float64
    lr_decay_coef::Float64
    lr_decay_power::Float64
    it_start_decay::Union{Int, Nothing}
    batch_size::Int
    avoid_cache_miss::Bool
    importance_sampling::Bool
    grad::Vector{Float64}

    function StochasticGradientDescent(loss; lr0=nothing, lr_max=Inf, lr_decay_coef=0.0,
                                      lr_decay_power=1.0, it_start_decay=nothing,
                                      batch_size=1, avoid_cache_miss=false,
                                      importance_sampling=false, kwargs...)
        optimizer = Optimizer(loss; kwargs...)
        new(optimizer, lr0, 0.0, lr_max, lr_decay_coef, lr_decay_power,
            it_start_decay, batch_size, avoid_cache_miss, importance_sampling, Float64[])
    end
end

function step!(sgd::StochasticGradientDescent)
    if sgd.avoid_cache_miss
        i = rand(sgd.optimizer.rng, 1:sgd.optimizer.loss.n)
        idx = collect((i-1) .+ (1:sgd.batch_size))
        idx .= mod1.(idx, sgd.optimizer.loss.n)
        sgd.grad = stochastic_gradient(sgd.optimizer.loss, sgd.optimizer.x, idx=idx)
    else
        # Only pass importance_sampling if the loss supports it
        if hasmethod(stochastic_gradient, (typeof(sgd.optimizer.loss), Vector{Float64}),
                    (:idx, :batch_size, :replace, :normalization, :importance_sampling, :rng, :return_idx))
            sgd.grad = stochastic_gradient(sgd.optimizer.loss, sgd.optimizer.x,
                                          batch_size=sgd.batch_size,
                                          importance_sampling=sgd.importance_sampling)
        else
            sgd.grad = stochastic_gradient(sgd.optimizer.loss, sgd.optimizer.x,
                                          batch_size=sgd.batch_size)
        end
    end

    denom_const = 1.0 / sgd.lr0
    it_decrease = max(0, sgd.optimizer.it - sgd.it_start_decay)
    lr_decayed = 1.0 / (denom_const + sgd.lr_decay_coef * it_decrease^sgd.lr_decay_power)

    if lr_decayed < 0
        lr_decayed = Inf
    end

    sgd.lr = min(lr_decayed, sgd.lr_max)
    sgd.optimizer.x .-= sgd.lr .* sgd.grad

    if sgd.optimizer.use_prox
        sgd.optimizer.x = prox(sgd.optimizer.loss.regularizer, sgd.optimizer.x, sgd.lr)
    end
end

function init_run!(sgd::StochasticGradientDescent, x0; kwargs...)
    init_run!(sgd.optimizer, x0; kwargs...)

    if sgd.lr0 === nothing
        sgd.lr0 = 1.0 / batch_smoothness(sgd.optimizer.loss, sgd.batch_size)
    end

    if sgd.it_start_decay === nothing && isfinite(sgd.optimizer.it_max)
        sgd.it_start_decay = sgd.optimizer.it_max ÷ 40
    elseif sgd.it_start_decay === nothing
        sgd.it_start_decay = 0
    end
end

function run!(sgd::StochasticGradientDescent, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("StochasticGradientDescent: The number of iterations is set to $it_max.")
    end

    sgd.optimizer.t_max = t_max
    sgd.optimizer.it_max = it_max

    # Use first seed for single-seed run
    seed = sgd.optimizer.seeds[1]
    if seed in sgd.optimizer.finished_seeds
        return sgd.optimizer.trace
    end

    sgd.optimizer.rng = MersenneTwister(seed)
    sgd.optimizer.seed = seed
    loss_seed = rand(sgd.optimizer.rng, 1:100000)
    set_seed!(sgd.optimizer.loss, loss_seed)
    init_seed!(sgd.optimizer.trace)

    if ls_it_max === nothing
        sgd.optimizer.ls_it_max = it_max
    else
        sgd.optimizer.ls_it_max = ls_it_max
    end

    if !sgd.optimizer.initialized[seed]
        init_run!(sgd, x0)
        sgd.optimizer.initialized[seed] = true
        if sgd.optimizer.line_search !== nothing
            reset!(sgd.optimizer.line_search, sgd.optimizer)
        end
    end

    while !check_convergence(sgd.optimizer)
        if sgd.optimizer.tolerance > 0
            sgd.optimizer.x_old_tol = copy(sgd.optimizer.x)
        end
        step!(sgd)
        save_checkpoint!(sgd.optimizer)
    end

    append_seed_results!(sgd.optimizer.trace, seed)
    push!(sgd.optimizer.finished_seeds, seed)
    sgd.optimizer.seed = nothing

    return sgd.optimizer.trace
end
