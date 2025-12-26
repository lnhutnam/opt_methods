include("../optimizer.jl")

"""
    NesterovAcceleratedGradient(loss; lr=nothing, strongly_convex=false,
                                max_momentum=nothing, mu=0.0,
                                start_with_small_momentum=true, kwargs...)

Nesterov's accelerated gradient descent with momentum.

Reference: Nesterov (1983) "A method for solving the convex programming problem
with convergence rate O(1/k²)"

# Arguments
- `loss::Oracle`: Optimization oracle
- `lr::Union{Float64,Nothing}=nothing`: Learning rate (defaults to 1/L)
- `strongly_convex::Bool=false`: Use variant for strongly convex functions
- `max_momentum::Union{Float64,Nothing}=nothing`: Target momentum value
- `mu::Float64=0.0`: Strong convexity constant (required if strongly_convex=true)
- `start_with_small_momentum::Bool=true`: Gradually increase momentum from 0
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
nag = NesterovAcceleratedGradient(loss, lr=0.01)
trace = run!(nag, x0, it_max=1000)
```
"""
mutable struct NesterovAcceleratedGradient
    optimizer::Optimizer
    lr::Union{Float64, Nothing}
    max_momentum::Union{Float64, Nothing}
    mu::Float64
    strongly_convex::Bool
    start_with_small_momentum::Bool

    # Internal state
    x_nest::Vector{Float64}
    x_old::Vector{Float64}
    alpha::Float64
    momentum::Float64
    grad::Vector{Float64}

    function NesterovAcceleratedGradient(loss; lr=nothing, strongly_convex=false,
                                       max_momentum=nothing, mu=0.0,
                                       start_with_small_momentum=true, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if strongly_convex && mu <= 0 && max_momentum === nothing
            error("Mu must be larger than 0 for strongly_convex=true, invalid value: $mu")
        end

        new(optimizer, lr, max_momentum, mu, strongly_convex, start_with_small_momentum,
            Float64[], Float64[], 1.0, 0.0, Float64[])
    end
end

function step!(nag::NesterovAcceleratedGradient)
    if !nag.strongly_convex || nag.start_with_small_momentum
        alpha_new = 0.5 * (1 + sqrt(1 + 4 * nag.alpha^2))
        nag.momentum = (nag.alpha - 1) / alpha_new
        nag.alpha = alpha_new
        if nag.max_momentum !== nothing
            nag.momentum = min(nag.momentum, nag.max_momentum)
        end
    else
        nag.momentum = nag.max_momentum
    end

    nag.x_old = copy(nag.optimizer.x)
    nag.grad = gradient(nag.optimizer.loss, nag.x_nest)
    nag.optimizer.x = nag.x_nest .- nag.lr .* nag.grad

    if nag.optimizer.use_prox
        nag.optimizer.x = prox(nag.optimizer.loss.regularizer, nag.optimizer.x, nag.lr)
    end

    nag.x_nest = nag.optimizer.x .+ nag.momentum .* (nag.optimizer.x .- nag.x_old)
end

function init_run!(nag::NesterovAcceleratedGradient, x0; kwargs...)
    init_run!(nag.optimizer, x0; kwargs...)

    if nag.lr === nothing
        nag.lr = 1.0 / smoothness(nag.optimizer.loss)
    end

    nag.x_nest = copy(nag.optimizer.x)
    nag.alpha = 1.0

    if nag.strongly_convex && nag.max_momentum === nothing
        kappa = (1/nag.lr) / nag.mu
        nag.max_momentum = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
    elseif nag.max_momentum === nothing
        nag.max_momentum = 1.0 - 1e-8
    end
end

# Implement run! for NesterovAcceleratedGradient
function run!(nag::NesterovAcceleratedGradient, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(nag.optimizer.label): The number of iterations is set to $it_max.")
    end

    nag.optimizer.t_max = t_max
    nag.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(nag.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(nag.optimizer.seeds)) seeds...")
    end

    for seed in nag.optimizer.seeds
        if seed in nag.optimizer.finished_seeds
            continue
        end

        nag.optimizer.rng = MersenneTwister(seed)
        nag.optimizer.seed = seed
        loss_seed = rand(nag.optimizer.rng, 1:MAX_SEED)
        set_seed!(nag.optimizer.loss, loss_seed)
        init_seed!(nag.optimizer.trace)

        if ls_it_max === nothing
            nag.optimizer.ls_it_max = it_max
        else
            nag.optimizer.ls_it_max = ls_it_max
        end

        if !nag.optimizer.initialized[seed]
            init_run!(nag, x0)
            nag.optimizer.initialized[seed] = true
            if nag.optimizer.line_search !== nothing
                reset!(nag.optimizer.line_search)
            end
        end

        it_criterion = nag.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(nag.optimizer.ls_it_max))")
        end

        while !check_convergence(nag.optimizer)
            if nag.optimizer.tolerance > 0
                nag.optimizer.x_old_tol = copy(nag.optimizer.x)
            end
            step!(nag)
            save_checkpoint!(nag.optimizer)

            if tqdm_iterations && nag.optimizer.it % 100 == 0
                println("Iteration: $(nag.optimizer.it)")
            end
        end

        append_seed_results!(nag.optimizer.trace, seed)
        push!(nag.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(nag.optimizer.finished_seeds))/$(length(nag.optimizer.seeds))")
        end
    end

    nag.optimizer.seed = nothing
    return nag.optimizer.trace
end
