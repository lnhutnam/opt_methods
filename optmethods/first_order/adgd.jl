include("../optimizer.jl")

"""
Gradient descent with adaptive stepsize estimation
using local values of smoothness (gradient Lipschitzness).

Arguments:
    lr0 (float, optional): a small value that idealy should be smaller than the
        inverse (local) smoothness constant. Does not affect performance too much.
"""
mutable struct AdaptiveGradientDescent
    optimizer::Optimizer
    lr0::Float64

    # Internal state
    lr::Float64
    theta::Float64
    x_old::Union{Vector{Float64}, Nothing}
    grad_old::Union{Vector{Float64}, Nothing}
    grad::Vector{Float64}

    function AdaptiveGradientDescent(loss; lr0=1e-6, kwargs...)
        optimizer = Optimizer(loss; kwargs...)
        new(optimizer, lr0, lr0, 1e12, nothing, nothing, Float64[])
    end
end

function estimate_new_stepsize!(adgd::AdaptiveGradientDescent)
    if adgd.grad_old !== nothing && adgd.x_old !== nothing
        grad_diff_norm = norm(adgd.grad .- adgd.grad_old)
        x_diff_norm = norm(adgd.optimizer.x .- adgd.x_old)

        if x_diff_norm > 0
            L = grad_diff_norm / x_diff_norm
            if L == 0
                lr_new = sqrt(1 + adgd.theta) * adgd.lr
            else
                lr_new = min(sqrt(1 + adgd.theta) * adgd.lr, 0.5/L)
            end
            adgd.theta = lr_new / adgd.lr
            adgd.lr = lr_new
        end
    end
end

function step!(adgd::AdaptiveGradientDescent)
    adgd.grad = gradient(adgd.optimizer.loss, adgd.optimizer.x)
    estimate_new_stepsize!(adgd)

    adgd.x_old = copy(adgd.optimizer.x)
    adgd.grad_old = copy(adgd.grad)

    adgd.optimizer.x .-= adgd.lr .* adgd.grad

    if adgd.optimizer.use_prox
        adgd.optimizer.x = prox(adgd.optimizer.loss.regularizer, adgd.optimizer.x, adgd.lr)
    end
end

function init_run!(adgd::AdaptiveGradientDescent, x0; kwargs...)
    init_run!(adgd.optimizer, x0; kwargs...)
    adgd.lr = adgd.lr0
    adgd.theta = 1e12
    adgd.grad_old = nothing
    adgd.x_old = nothing
end

function run!(adgd::AdaptiveGradientDescent, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
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