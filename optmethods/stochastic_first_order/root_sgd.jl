include("../optimizer.jl")
using Random

"""
    RootSGD(loss; lr0=nothing, lr_max=Inf, lr_decay_coef=0.0, lr_decay_power=1.0,
            it_start_decay=nothing, first_batch=nothing, batch_size=1,
            avoid_cache_miss=true, kwargs...)

Recursive One-Over-T SGD with decreasing or constant learning rate.

Based on: Cutkosky & Orabona (2019) "Momentum-Based Variance Reduction
in Non-Convex SGD"
https://arxiv.org/pdf/2008.12690.pdf

# Arguments
- `loss::Oracle`: Optimization oracle
- `lr0::Union{Float64,Nothing}=nothing`: Initial learning rate
- `lr_max::Float64=Inf`: Maximum learning rate
- `lr_decay_coef::Float64=0.0`: Coefficient for learning rate decay
- `lr_decay_power::Float64=1.0`: Power for iteration exponentiation in decay
- `it_start_decay::Union{Int,Nothing}=nothing`: Iterations before decay starts
  Default: it_max/40
- `first_batch::Union{Int,Nothing}=nothing`: Initial batch size for first gradient
  Default: 10 * batch_size
- `batch_size::Int=1`: Number of samples per iteration
- `avoid_cache_miss::Bool=true`: Sample adjacent indices for cache efficiency
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
rootsgd = RootSGD(loss, lr0=0.01, batch_size=32)
trace = run!(rootsgd, x0, it_max=10000)
```
"""
mutable struct RootSGD
    optimizer::Optimizer
    lr0::Union{Float64, Nothing}
    lr::Float64
    lr_max::Float64
    lr_decay_coef::Float64
    lr_decay_power::Float64
    it_start_decay::Union{Int, Nothing}
    first_batch::Int
    batch_size::Int
    avoid_cache_miss::Bool

    # Internal state
    x_old::Vector{Float64}
    grad::Vector{Float64}
    grad_old::Vector{Float64}
    grad_estim::Vector{Float64}

    function RootSGD(loss; lr0=nothing, lr_max=Inf, lr_decay_coef=0.0,
                    lr_decay_power=1.0, it_start_decay=nothing,
                    first_batch=nothing, batch_size=1, avoid_cache_miss=true, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if it_start_decay === nothing && isfinite(optimizer.it_max)
            it_start_decay = optimizer.it_max ÷ 40
        elseif it_start_decay === nothing
            it_start_decay = 0
        end

        if first_batch === nothing
            first_batch = 10 * batch_size
        end

        new(optimizer, lr0, 0.0, lr_max, lr_decay_coef, lr_decay_power,
            it_start_decay, first_batch, batch_size, avoid_cache_miss,
            Float64[], Float64[], Float64[], Float64[])
    end
end

function step!(rootsgd::RootSGD)
    denom_const = 1.0 / rootsgd.lr0
    it_decrease = max(0, rootsgd.optimizer.it - rootsgd.it_start_decay)
    lr_decayed = 1.0 / (denom_const + rootsgd.lr_decay_coef * it_decrease^rootsgd.lr_decay_power)

    if lr_decayed < 0
        lr_decayed = Inf
    end

    rootsgd.lr = min(lr_decayed, rootsgd.lr_max)

    if rootsgd.optimizer.it > 0
        if rootsgd.avoid_cache_miss
            i = rand(rootsgd.optimizer.rng, 1:rootsgd.optimizer.loss.n)
            idx = collect((i-1) .+ (1:rootsgd.batch_size))
            idx .= mod1.(idx, rootsgd.optimizer.loss.n)
        else
            idx = rand(rootsgd.optimizer.rng, 1:rootsgd.optimizer.loss.n,
                      rootsgd.batch_size, replace=false)
        end

        rootsgd.grad_old = stochastic_gradient(rootsgd.optimizer.loss, rootsgd.x_old, idx=idx)
        rootsgd.grad_estim .-= rootsgd.grad_old
        rootsgd.grad_estim .*= 1.0 - 1.0 / (rootsgd.optimizer.it + rootsgd.first_batch / rootsgd.batch_size)
        rootsgd.grad = stochastic_gradient(rootsgd.optimizer.loss, rootsgd.optimizer.x, idx=idx)
        rootsgd.grad_estim .+= rootsgd.grad
    else
        rootsgd.grad_estim = stochastic_gradient(rootsgd.optimizer.loss, rootsgd.optimizer.x,
                                                batch_size=rootsgd.first_batch)
    end

    rootsgd.x_old = copy(rootsgd.optimizer.x)
    rootsgd.optimizer.x .-= rootsgd.lr .* rootsgd.grad_estim

    if rootsgd.optimizer.use_prox
        rootsgd.optimizer.x = prox(rootsgd.optimizer.loss.regularizer,
                                  rootsgd.optimizer.x, rootsgd.lr)
    end
end

function init_run!(rootsgd::RootSGD, x0; kwargs...)
    init_run!(rootsgd.optimizer, x0; kwargs...)

    if rootsgd.lr0 === nothing
        rootsgd.lr0 = 1.0 / batch_smoothness(rootsgd.optimizer.loss, rootsgd.batch_size)
    end
end

function run!(rootsgd::RootSGD, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("RootSGD: The number of iterations is set to $it_max.")
    end

    rootsgd.optimizer.t_max = t_max
    rootsgd.optimizer.it_max = it_max

    # Use first seed for single-seed run
    seed = rootsgd.optimizer.seeds[1]
    if seed in rootsgd.optimizer.finished_seeds
        return rootsgd.optimizer.trace
    end

    rootsgd.optimizer.rng = MersenneTwister(seed)
    rootsgd.optimizer.seed = seed
    loss_seed = rand(rootsgd.optimizer.rng, 1:100000)
    set_seed!(rootsgd.optimizer.loss, loss_seed)
    init_seed!(rootsgd.optimizer.trace)

    if ls_it_max === nothing
        rootsgd.optimizer.ls_it_max = it_max
    else
        rootsgd.optimizer.ls_it_max = ls_it_max
    end

    if !rootsgd.optimizer.initialized[seed]
        init_run!(rootsgd, x0)
        rootsgd.optimizer.initialized[seed] = true
        if rootsgd.optimizer.line_search !== nothing
            reset!(rootsgd.optimizer.line_search, rootsgd.optimizer)
        end
    end

    while !check_convergence(rootsgd.optimizer)
        if rootsgd.optimizer.tolerance > 0
            rootsgd.optimizer.x_old_tol = copy(rootsgd.optimizer.x)
        end
        step!(rootsgd)
        save_checkpoint!(rootsgd.optimizer)
    end

    append_seed_results!(rootsgd.optimizer.trace, seed)
    push!(rootsgd.optimizer.finished_seeds, seed)
    rootsgd.optimizer.seed = nothing

    return rootsgd.optimizer.trace
end
