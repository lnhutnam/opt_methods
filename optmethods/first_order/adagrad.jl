include("../optimizer.jl")

"""
Implement Adagrad from Duchi et. al, 2011
    "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
This implementation only supports deterministic gradients and use dense vectors.

Arguments:
    primal_dual (boolean, optional): if true, uses the dual averaging method of Nesterov,
        otherwise uses gradient descent update (default: false)
    lr (float, optional): learning rate coefficient, which needs to be tuned to
        get better performance (default: 1.)
    delta (float, optional): another learning rate parameter, slows down performance if
        chosen too large, so a small value is recommended, otherwise requires tuning (default: 0.)
"""
mutable struct AdaGrad
    optimizer::Optimizer
    primal_dual::Bool
    lr::Float64
    delta::Float64

    # Internal state
    x0::Vector{Float64}
    s::Vector{Float64}
    sum_grad::Vector{Float64}
    inv_lr::Vector{Float64}
    grad::Vector{Float64}

    function AdaGrad(loss; primal_dual=false, lr=1.0, delta=0.0, kwargs...)
        optimizer = Optimizer(loss; kwargs...)
        new(optimizer, primal_dual, lr, delta,
            Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

function estimate_stepsize!(adagrad::AdaGrad)
    adagrad.s = sqrt.(adagrad.s.^2 .+ adagrad.grad.^2)
    adagrad.inv_lr = adagrad.delta .+ adagrad.s
end

function step!(adagrad::AdaGrad)
    adagrad.grad = gradient(adagrad.optimizer.loss, adagrad.optimizer.x)
    estimate_stepsize!(adagrad)

    if adagrad.primal_dual
        adagrad.sum_grad .+= adagrad.grad
        # Safe division avoiding division by zero
        safe_div = adagrad.sum_grad ./ max.(adagrad.inv_lr, eps(Float64))
        adagrad.optimizer.x = adagrad.x0 .- adagrad.lr .* safe_div
    else
        # Safe division avoiding division by zero
        safe_div = adagrad.grad ./ max.(adagrad.inv_lr, eps(Float64))
        adagrad.optimizer.x .-= adagrad.lr .* safe_div
    end
end

function init_run!(adagrad::AdaGrad, x0; kwargs...)
    init_run!(adagrad.optimizer, x0; kwargs...)

    # Convert to dense array if sparse
    if issparse(adagrad.optimizer.x)
        adagrad.optimizer.x = Array(adagrad.optimizer.x)
    end

    adagrad.x0 = copy(adagrad.optimizer.x)
    adagrad.s = zeros(Float64, length(adagrad.optimizer.x))
    adagrad.sum_grad = zeros(Float64, length(adagrad.optimizer.x))
end

function run!(adagrad::AdaGrad, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(adagrad.optimizer.label): The number of iterations is set to $it_max.")
    end

    adagrad.optimizer.t_max = t_max
    adagrad.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(adagrad.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(adagrad.optimizer.seeds)) seeds...")
    end

    for seed in adagrad.optimizer.seeds
        if seed in adagrad.optimizer.finished_seeds
            continue
        end

        adagrad.optimizer.rng = MersenneTwister(seed)
        adagrad.optimizer.seed = seed
        loss_seed = rand(adagrad.optimizer.rng, 1:MAX_SEED)
        set_seed!(adagrad.optimizer.loss, loss_seed)
        init_seed!(adagrad.optimizer.trace)

        if ls_it_max === nothing
            adagrad.optimizer.ls_it_max = it_max
        else
            adagrad.optimizer.ls_it_max = ls_it_max
        end

        if !adagrad.optimizer.initialized[seed]
            init_run!(adagrad, x0)
            adagrad.optimizer.initialized[seed] = true
            if adagrad.optimizer.line_search !== nothing
                reset!(adagrad.optimizer.line_search)
            end
        end

        it_criterion = adagrad.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(adagrad.optimizer.ls_it_max))")
        end

        while !check_convergence(adagrad.optimizer)
            if adagrad.optimizer.tolerance > 0
                adagrad.optimizer.x_old_tol = copy(adagrad.optimizer.x)
            end
            step!(adagrad)
            save_checkpoint!(adagrad.optimizer)

            if tqdm_iterations && adagrad.optimizer.it % 100 == 0
                println("Iteration: $(adagrad.optimizer.it)")
            end
        end

        append_seed_results!(adagrad.optimizer.trace, seed)
        push!(adagrad.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(adagrad.optimizer.finished_seeds))/$(length(adagrad.optimizer.seeds))")
        end
    end

    adagrad.optimizer.seed = nothing
    return adagrad.optimizer.trace
end