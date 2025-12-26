include("../optimizer.jl")
using Random

"""
    SVRG(loss; lr=nothing, batch_size=1, avoid_cache_miss=false,
         loopless=true, loop_len=nothing, restart_prob=nothing, kwargs...)

Stochastic Variance-Reduced Gradient descent with constant stepsize.

Reference: Johnson & Zhang (2013) "Accelerating Stochastic Gradient Descent
using Predictive Variance Reduction"
https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf

# Arguments
- `loss::Oracle`: Optimization oracle
- `lr::Union{Float64,Nothing}=nothing`: Learning rate (defaults to 0.5/L)
- `batch_size::Int=1`: Number of samples per iteration
- `avoid_cache_miss::Bool=false`: Sample adjacent indices for cache efficiency
- `loopless::Bool=true`: Use loopless variant (probabilistic restarts)
- `loop_len::Union{Int,Nothing}=nothing`: Length of inner loop (if not loopless)
  Default: n/batch_size
- `restart_prob::Union{Float64,Nothing}=nothing`: Restart probability (if loopless)
  Default: batch_size/n
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
svrg = SVRG(loss, lr=0.01, batch_size=32)
trace = run!(svrg, x0, it_max=10000)
```
"""
mutable struct SVRG
    optimizer::Optimizer
    lr::Union{Float64, Nothing}
    batch_size::Int
    avoid_cache_miss::Bool
    loopless::Bool
    loop_len::Union{Int, Nothing}
    restart_prob::Union{Float64, Nothing}

    # Internal state
    x_old::Vector{Float64}
    full_grad_old::Vector{Float64}
    vr_grad::Vector{Float64}
    loop_it::Int
    loops::Int

    function SVRG(loss; lr=nothing, batch_size=1, avoid_cache_miss=false,
                  loopless=true, loop_len=nothing, restart_prob=nothing, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        # Set default restart probability or loop length
        if loopless && restart_prob === nothing
            restart_prob = batch_size / loss.n
        elseif !loopless && loop_len === nothing
            loop_len = loss.n ÷ batch_size
        end

        new(optimizer, lr, batch_size, avoid_cache_miss, loopless, loop_len, restart_prob,
            Float64[], Float64[], Float64[], 0, 0)
    end
end

function step!(svrg::SVRG)
    new_loop = svrg.loopless && rand(svrg.optimizer.rng) < svrg.restart_prob

    if !svrg.loopless && svrg.loop_it == svrg.loop_len
        new_loop = true
    end

    if new_loop || svrg.optimizer.it == 0
        svrg.x_old = copy(svrg.optimizer.x)
        svrg.full_grad_old = gradient(svrg.optimizer.loss, svrg.x_old)
        svrg.vr_grad = copy(svrg.full_grad_old)
        if !svrg.loopless
            svrg.loop_it = 0
        end
        svrg.loops += 1
    else
        if svrg.avoid_cache_miss
            i = rand(svrg.optimizer.rng, 1:svrg.optimizer.loss.n)
            idx = collect((i-1) .+ (1:svrg.batch_size))
            idx .= mod1.(idx, svrg.optimizer.loss.n)
        else
            idx = rand(svrg.optimizer.rng, 1:svrg.optimizer.loss.n, svrg.batch_size)
        end

        stoch_grad = stochastic_gradient(svrg.optimizer.loss, svrg.optimizer.x, idx=idx)
        stoch_grad_old = stochastic_gradient(svrg.optimizer.loss, svrg.x_old, idx=idx)
        svrg.vr_grad = stoch_grad .- stoch_grad_old .+ svrg.full_grad_old
    end

    svrg.optimizer.x .-= svrg.lr .* svrg.vr_grad

    if svrg.optimizer.use_prox
        svrg.optimizer.x = prox(svrg.optimizer.loss.regularizer, svrg.optimizer.x, svrg.lr)
    end

    svrg.loop_it += 1
end

function init_run!(svrg::SVRG, x0; kwargs...)
    init_run!(svrg.optimizer, x0; kwargs...)
    svrg.loop_it = 0
    svrg.loops = 0

    if svrg.lr === nothing
        svrg.lr = 0.5 / batch_smoothness(svrg.optimizer.loss, svrg.batch_size)
    end
end

function run!(svrg::SVRG, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("SVRG: The number of iterations is set to $it_max.")
    end

    svrg.optimizer.t_max = t_max
    svrg.optimizer.it_max = it_max

    # Use first seed for single-seed run
    seed = svrg.optimizer.seeds[1]
    if seed in svrg.optimizer.finished_seeds
        return svrg.optimizer.trace
    end

    svrg.optimizer.rng = MersenneTwister(seed)
    svrg.optimizer.seed = seed
    loss_seed = rand(svrg.optimizer.rng, 1:100000)
    set_seed!(svrg.optimizer.loss, loss_seed)
    init_seed!(svrg.optimizer.trace)

    if ls_it_max === nothing
        svrg.optimizer.ls_it_max = it_max
    else
        svrg.optimizer.ls_it_max = ls_it_max
    end

    if !svrg.optimizer.initialized[seed]
        init_run!(svrg, x0)
        svrg.optimizer.initialized[seed] = true
        if svrg.optimizer.line_search !== nothing
            reset!(svrg.optimizer.line_search, svrg.optimizer)
        end
    end

    while !check_convergence(svrg.optimizer)
        if svrg.optimizer.tolerance > 0
            svrg.optimizer.x_old_tol = copy(svrg.optimizer.x)
        end
        step!(svrg)
        save_checkpoint!(svrg.optimizer)
    end

    append_seed_results!(svrg.optimizer.trace, seed)
    push!(svrg.optimizer.finished_seeds, seed)
    svrg.optimizer.seed = nothing

    return svrg.optimizer.trace
end
