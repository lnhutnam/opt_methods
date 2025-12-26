include("../optimizer.jl")

"""
    AdgdAccel(loss; lr0=1e-6, max_momentum=1-1e-6, kwargs...)

Accelerated gradient descent with adaptive stepsize and momentum estimation
using local values of smoothness (gradient Lipschitzness) and strong convexity.
Momentum is used as given by Nesterov's acceleration.

Reference: Adaptive accelerated gradient methods

# Arguments
- `loss::Oracle`: Optimization oracle
- `lr0::Float64=1e-6`: Initial learning rate, should be smaller than inverse smoothness
- `max_momentum::Float64=1-1e-6`: Maximum momentum value
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
adgd_accel = AdgdAccel(loss, lr0=1e-6)
trace = run!(adgd_accel, x0, it_max=1000)
```
"""
mutable struct AdgdAccel
    optimizer::Optimizer
    lr0::Float64
    max_momentum::Float64

    # Internal state
    x_nest::Vector{Float64}
    x_nest_old::Vector{Float64}
    x_old::Vector{Float64}
    grad::Vector{Float64}
    grad_old::Union{Vector{Float64}, Nothing}
    lr::Float64
    mu::Float64
    momentum::Float64
    theta::Float64
    theta_mu::Float64
    alpha::Float64
    L::Float64

    function AdgdAccel(loss; lr0=1e-6, max_momentum=1.0-1e-6, kwargs...)
        optimizer = Optimizer(loss; kwargs...)
        new(optimizer, lr0, max_momentum,
            Float64[], Float64[], Float64[], Float64[], nothing,
            lr0, 0.0, 0.0, 1e12, 1.0, 1.0, 0.0)
    end
end

function estimate_new_stepsize!(adgd::AdgdAccel)
    if adgd.grad_old !== nothing
        grad_diff_norm = norm(adgd.grad .- adgd.grad_old)
        x_diff_norm = norm(adgd.x_nest .- adgd.x_nest_old)

        if x_diff_norm > 0
            adgd.L = grad_diff_norm / x_diff_norm
        else
            adgd.L = 0.0
        end

        if adgd.L == 0
            lr_new = sqrt(1 + 0.5 * adgd.theta) * adgd.lr
        else
            lr_new = min(sqrt(1 + 0.5 * adgd.theta) * adgd.lr, 0.5 / adgd.L)
        end
        adgd.theta = lr_new / adgd.lr
        adgd.lr = lr_new
    end
end

function estimate_new_momentum!(adgd::AdgdAccel)
    alpha_new = 0.5 * (1 + sqrt(1 + 4 * adgd.alpha^2))
    adgd.momentum = (adgd.alpha - 1) / alpha_new
    adgd.alpha = alpha_new

    if adgd.grad_old !== nothing
        if adgd.L == 0
            mu_new = adgd.mu / 10
        else
            mu_new = min(sqrt(1 + 0.5 * adgd.theta_mu) * adgd.mu, 0.5 * adgd.L)
        end
        adgd.theta_mu = mu_new / adgd.mu
        adgd.mu = mu_new
        kappa = 1 / (adgd.lr * adgd.mu)
        adgd.momentum = min(adgd.momentum, 1 - 2 / (1 + sqrt(kappa)))
    end

    # Clamp momentum to max_momentum
    adgd.momentum = min(adgd.momentum, adgd.max_momentum)
end

function step!(adgd::AdgdAccel)
    adgd.grad = gradient(adgd.optimizer.loss, adgd.x_nest)
    estimate_new_stepsize!(adgd)
    estimate_new_momentum!(adgd)

    adgd.x_nest_old = copy(adgd.x_nest)
    adgd.x_old = copy(adgd.optimizer.x)
    adgd.grad_old = copy(adgd.grad)

    adgd.optimizer.x = adgd.x_nest .- adgd.lr .* adgd.grad

    if adgd.optimizer.use_prox
        adgd.optimizer.x = prox(adgd.optimizer.loss.regularizer, adgd.optimizer.x, adgd.lr)
    end

    adgd.x_nest = adgd.optimizer.x .+ adgd.momentum .* (adgd.optimizer.x .- adgd.x_old)
end

function init_run!(adgd::AdgdAccel, x0; kwargs...)
    init_run!(adgd.optimizer, x0; kwargs...)

    adgd.x_nest = copy(adgd.optimizer.x)
    adgd.momentum = 0.0
    adgd.lr = adgd.lr0
    adgd.mu = 1.0 / adgd.lr
    adgd.theta = 1e12
    adgd.theta_mu = 1.0
    adgd.grad_old = nothing
    adgd.alpha = 1.0
    adgd.L = 0.0
end

function run!(adgd::AdgdAccel, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(adgd.optimizer.label): The number of iterations is set to $it_max.")
    end

    adgd.optimizer.t_max = t_max
    adgd.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(adgd.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(adgd.optimizer.seeds)) seeds...")
    end

    for seed in adgd.optimizer.seeds
        if seed in adgd.optimizer.finished_seeds
            continue
        end

        adgd.optimizer.rng = MersenneTwister(seed)
        adgd.optimizer.seed = seed
        loss_seed = rand(adgd.optimizer.rng, 1:MAX_SEED)
        set_seed!(adgd.optimizer.loss, loss_seed)
        init_seed!(adgd.optimizer.trace)

        if ls_it_max === nothing
            adgd.optimizer.ls_it_max = it_max
        else
            adgd.optimizer.ls_it_max = ls_it_max
        end

        if !adgd.optimizer.initialized[seed]
            init_run!(adgd, x0)
            adgd.optimizer.initialized[seed] = true
            if adgd.optimizer.line_search !== nothing
                reset!(adgd.optimizer.line_search)
            end
        end

        it_criterion = adgd.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(adgd.optimizer.ls_it_max))")
        end

        while !check_convergence(adgd.optimizer)
            if adgd.optimizer.tolerance > 0
                adgd.optimizer.x_old_tol = copy(adgd.optimizer.x)
            end
            step!(adgd)
            save_checkpoint!(adgd.optimizer)

            if tqdm_iterations && adgd.optimizer.it % 100 == 0
                println("Iteration: $(adgd.optimizer.it)")
            end
        end

        append_seed_results!(adgd.optimizer.trace, seed)
        push!(adgd.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(adgd.optimizer.finished_seeds))/$(length(adgd.optimizer.seeds))")
        end
    end

    adgd.optimizer.seed = nothing
    return adgd.optimizer.trace
end
