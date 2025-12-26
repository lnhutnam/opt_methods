include("../optimizer.jl")

"""
Polyak adaptive gradient descent, proposed in
    (B. T. Poyal, "Introduction to Optimization")
which can be accessed, e.g., here:
    https://www.researchgate.net/publication/342978480_Introduction_to_Optimization

Arguments:
    f_opt (float): precise value of the objective's minimum. If an underestimate is given,
        the algorirthm can be unstable; if an overestimate is given, will not converge below
        the overestimate.
    lr_min (float, optional): the smallest step-size, useful when
        an overestimate of the optimal value is given (default: 0)
    lr_max (float, optional): the laregest allowed step-size, useful when
        an underestimate of the optimal value is given (defaul: Inf)
"""
mutable struct PolyakStepSize
    optimizer::Optimizer
    f_opt::Float64
    lr_min::Float64
    lr_max::Float64

    # Internal state
    lr::Float64
    grad::Vector{Float64}

    function PolyakStepSize(loss; f_opt::Float64, lr_min=0.0, lr_max=Inf, kwargs...)
        optimizer = Optimizer(loss; kwargs...)
        new(optimizer, f_opt, lr_min, lr_max, 0.0, Float64[])
    end
end

function estimate_new_stepsize!(polyak::PolyakStepSize)
    loss_gap = value(polyak.optimizer.loss, polyak.optimizer.x) - polyak.f_opt
    grad_norm_sq = norm(polyak.grad)^2

    if grad_norm_sq > 0
        polyak.lr = loss_gap / grad_norm_sq
        polyak.lr = min(polyak.lr, polyak.lr_max)
        polyak.lr = max(polyak.lr, polyak.lr_min)
    else
        polyak.lr = polyak.lr_min
    end
end

function step!(polyak::PolyakStepSize)
    polyak.grad = gradient(polyak.optimizer.loss, polyak.optimizer.x)
    estimate_new_stepsize!(polyak)
    polyak.optimizer.x .-= polyak.lr .* polyak.grad
end

function init_run!(polyak::PolyakStepSize, x0; kwargs...)
    init_run!(polyak.optimizer, x0; kwargs...)
end

function run!(polyak::PolyakStepSize, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(polyak.optimizer.label): The number of iterations is set to $it_max.")
    end

    polyak.optimizer.t_max = t_max
    polyak.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(polyak.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(polyak.optimizer.seeds)) seeds...")
    end

    for seed in polyak.optimizer.seeds
        if seed in polyak.optimizer.finished_seeds
            continue
        end

        polyak.optimizer.rng = MersenneTwister(seed)
        polyak.optimizer.seed = seed
        loss_seed = rand(polyak.optimizer.rng, 1:MAX_SEED)
        set_seed!(polyak.optimizer.loss, loss_seed)
        init_seed!(polyak.optimizer.trace)

        if ls_it_max === nothing
            polyak.optimizer.ls_it_max = it_max
        else
            polyak.optimizer.ls_it_max = ls_it_max
        end

        if !polyak.optimizer.initialized[seed]
            init_run!(polyak, x0)
            polyak.optimizer.initialized[seed] = true
            if polyak.optimizer.line_search !== nothing
                reset!(polyak.optimizer.line_search)
            end
        end

        it_criterion = polyak.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(polyak.optimizer.ls_it_max))")
        end

        while !check_convergence(polyak.optimizer)
            if polyak.optimizer.tolerance > 0
                polyak.optimizer.x_old_tol = copy(polyak.optimizer.x)
            end
            step!(polyak)
            save_checkpoint!(polyak.optimizer)

            if tqdm_iterations && polyak.optimizer.it % 100 == 0
                println("Iteration: $(polyak.optimizer.it)")
            end
        end

        append_seed_results!(polyak.optimizer.trace, seed)
        push!(polyak.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(polyak.optimizer.finished_seeds))/$(length(polyak.optimizer.seeds))")
        end
    end

    polyak.optimizer.seed = nothing
    return polyak.optimizer.trace
end