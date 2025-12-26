include("../optimizer.jl")

"""
    RestNest(loss; lr=nothing, it_before_first_rest=10, func_condition=false,
             doubling=false, kwargs...)

Restarted Nesterov's Accelerated Gradient with constant learning rate.

For details, see https://arxiv.org/abs/1204.3982

# Arguments
- `loss::Oracle`: Optimization oracle
- `lr::Union{Float64,Nothing}=nothing`: Inverse smoothness estimate
- `it_before_first_rest::Int=10`: Iterations before first restart is allowed
- `func_condition::Bool=false`: Use objective decrease as restart condition
- `doubling::Bool=false`: Double iterations between restarts instead of checking conditions
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
rest_nest = RestNest(loss, it_before_first_rest=10)
trace = run!(rest_nest, x0, it_max=1000)
```
"""
mutable struct RestNest
    optimizer::Optimizer
    lr::Union{Float64, Nothing}
    it_before_first_rest::Int
    func_condition::Bool
    doubling::Bool

    # Internal state
    x_nest::Vector{Float64}
    x_old::Vector{Float64}
    grad::Vector{Float64}
    alpha::Float64
    momentum::Float64
    it_without_rest::Int
    potential_old::Float64
    n_restarts::Int
    it_until_rest::Int

    function RestNest(loss; lr=nothing, it_before_first_rest=10, func_condition=false,
                     doubling=false, kwargs...)
        optimizer = Optimizer(loss; kwargs...)
        new(optimizer, lr, it_before_first_rest, func_condition, doubling,
            Float64[], Float64[], Float64[], 1.0, 0.0, 0, Inf, 0,
            it_before_first_rest)
    end
end

function restart_condition(rn::RestNest)
    if rn.it_without_rest < rn.it_before_first_rest
        return false
    end

    if rn.doubling
        if rn.it_without_rest >= rn.it_until_rest
            rn.it_until_rest *= 2
            return true
        end
        return false
    end

    if rn.func_condition
        potential = value(rn.optimizer.loss, rn.optimizer.x)
        restart = potential > rn.potential_old
        rn.potential_old = potential
        return restart
    end

    # Gradient restart condition
    direction_is_bad = dot(rn.optimizer.x .- rn.x_old, rn.grad) > 0
    return direction_is_bad
end

function step!(rn::RestNest)
    rn.x_old = copy(rn.optimizer.x)
    rn.grad = gradient(rn.optimizer.loss, rn.x_nest)
    rn.optimizer.x = rn.x_nest .- rn.lr .* rn.grad

    if rn.optimizer.use_prox
        rn.optimizer.x = prox(rn.optimizer.loss.regularizer, rn.optimizer.x, rn.lr)
    end

    if restart_condition(rn)
        rn.n_restarts += 1
        rn.alpha = 1.0
        rn.it_without_rest = 0
        rn.potential_old = Inf
    else
        rn.it_without_rest += 1
    end

    alpha_new = 0.5 * (1 + sqrt(1 + 4 * rn.alpha^2))
    rn.momentum = (rn.alpha - 1) / alpha_new
    rn.alpha = alpha_new
    rn.x_nest = rn.optimizer.x .+ rn.momentum .* (rn.optimizer.x .- rn.x_old)
end

function init_run!(rn::RestNest, x0; kwargs...)
    init_run!(rn.optimizer, x0; kwargs...)

    if rn.lr === nothing
        rn.lr = 1.0 / smoothness(rn.optimizer.loss)
    end

    rn.x_nest = copy(rn.optimizer.x)
    rn.alpha = 1.0
    rn.it_without_rest = 0
    rn.potential_old = Inf
    rn.n_restarts = 0

    if rn.doubling
        rn.it_until_rest = rn.it_before_first_rest
    end
end

function run!(rn::RestNest, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(rn.optimizer.label): The number of iterations is set to $it_max.")
    end

    rn.optimizer.t_max = t_max
    rn.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(rn.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(rn.optimizer.seeds)) seeds...")
    end

    for seed in rn.optimizer.seeds
        if seed in rn.optimizer.finished_seeds
            continue
        end

        rn.optimizer.rng = MersenneTwister(seed)
        rn.optimizer.seed = seed
        loss_seed = rand(rn.optimizer.rng, 1:MAX_SEED)
        set_seed!(rn.optimizer.loss, loss_seed)
        init_seed!(rn.optimizer.trace)

        if ls_it_max === nothing
            rn.optimizer.ls_it_max = it_max
        else
            rn.optimizer.ls_it_max = ls_it_max
        end

        if !rn.optimizer.initialized[seed]
            init_run!(rn, x0)
            rn.optimizer.initialized[seed] = true
            if rn.optimizer.line_search !== nothing
                reset!(rn.optimizer.line_search)
            end
        end

        it_criterion = rn.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(rn.optimizer.ls_it_max))")
        end

        while !check_convergence(rn.optimizer)
            if rn.optimizer.tolerance > 0
                rn.optimizer.x_old_tol = copy(rn.optimizer.x)
            end
            step!(rn)
            save_checkpoint!(rn.optimizer)

            if tqdm_iterations && rn.optimizer.it % 100 == 0
                println("Iteration: $(rn.optimizer.it)")
            end
        end

        append_seed_results!(rn.optimizer.trace, seed)
        push!(rn.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(rn.optimizer.finished_seeds))/$(length(rn.optimizer.seeds))")
        end
    end

    rn.optimizer.seed = nothing
    return rn.optimizer.trace
end
