include("../optimizer.jl")

"""
Optimized (accelerated) gradient method with constant learning rate.
For a simple convergence proof, see, e.g.,
    https://arxiv.org/abs/2102.07366

Arguments:
    lr (float, optional): an estimate of the inverse smoothness constant
    strongly_convex (bool, optional): use the variant for strongly convex functions,
        which requires mu to be provided (default: false)
    mu (float, optional): strong-convexity constant or a lower bound on it (default: 0)
    start_with_small_momentum (bool, optional): momentum gradually increases. Only used if
        strongly_convex is set to true (default: true)
"""
mutable struct OptimizedGradientMethod
    optimizer::Optimizer
    lr::Union{Float64, Nothing}
    mu::Float64
    strongly_convex::Bool
    start_with_small_momentum::Bool

    # Internal state
    x_nest::Vector{Float64}
    x_old::Vector{Float64}
    alpha::Float64
    momentum1::Float64
    momentum2::Float64
    max_momentum::Float64
    gamma::Float64
    grad::Vector{Float64}

    function OptimizedGradientMethod(loss; lr=nothing, strongly_convex=false, mu=0.0,
                                    start_with_small_momentum=true, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if strongly_convex && mu <= 0
            error("Mu must be larger than 0 for strongly_convex=true, invalid value: $mu")
        end

        new(optimizer, lr, mu, strongly_convex, start_with_small_momentum,
            Float64[], Float64[], 1.0, 0.0, 0.0, 1.0, 0.0, Float64[])
    end
end

function step!(ogm::OptimizedGradientMethod)
    if !ogm.strongly_convex || ogm.start_with_small_momentum
        alpha_new = 0.5 * (1 + sqrt(1 + 4 * ogm.alpha^2))
        ogm.momentum1 = (ogm.alpha - 1) / alpha_new
        ogm.momentum2 = ogm.alpha / alpha_new
        ogm.alpha = alpha_new
        ogm.momentum1 = min(ogm.momentum1, ogm.max_momentum)
        ogm.momentum2 = min(ogm.momentum2, ogm.max_momentum)
    else
        ogm.momentum1 = ogm.momentum2 = ogm.max_momentum
    end

    ogm.x_old = copy(ogm.optimizer.x)
    ogm.grad = gradient(ogm.optimizer.loss, ogm.x_nest)
    ogm.optimizer.x = ogm.x_nest .- ogm.lr .* ogm.grad

    if ogm.optimizer.use_prox
        ogm.optimizer.x = prox(ogm.optimizer.loss.regularizer, ogm.optimizer.x, ogm.lr)
    end

    ogm.x_nest = ogm.optimizer.x .+ ogm.momentum1 .* (ogm.optimizer.x .- ogm.x_old) .+
                 ogm.momentum2 .* (ogm.optimizer.x .- ogm.x_nest)
end

function init_run!(ogm::OptimizedGradientMethod, x0; kwargs...)
    init_run!(ogm.optimizer, x0; kwargs...)

    if ogm.lr === nothing
        ogm.lr = 1.0 / smoothness(ogm.optimizer.loss)
    end

    ogm.x_nest = copy(ogm.optimizer.x)
    ogm.alpha = 1.0

    if ogm.strongly_convex
        kappa = (1/ogm.lr) / ogm.mu
        ogm.gamma = (sqrt(8*kappa + 1) + 3) / (2*kappa - 2)
        ogm.max_momentum = 1 / (2*ogm.gamma + 1)
    else
        ogm.max_momentum = 1.0
    end
end

function run!(ogm::OptimizedGradientMethod, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(ogm.optimizer.label): The number of iterations is set to $it_max.")
    end

    ogm.optimizer.t_max = t_max
    ogm.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(ogm.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(ogm.optimizer.seeds)) seeds...")
    end

    for seed in ogm.optimizer.seeds
        if seed in ogm.optimizer.finished_seeds
            continue
        end

        ogm.optimizer.rng = MersenneTwister(seed)
        ogm.optimizer.seed = seed
        loss_seed = rand(ogm.optimizer.rng, 1:MAX_SEED)
        set_seed!(ogm.optimizer.loss, loss_seed)
        init_seed!(ogm.optimizer.trace)

        if ls_it_max === nothing
            ogm.optimizer.ls_it_max = it_max
        else
            ogm.optimizer.ls_it_max = ls_it_max
        end

        if !ogm.optimizer.initialized[seed]
            init_run!(ogm, x0)
            ogm.optimizer.initialized[seed] = true
            if ogm.optimizer.line_search !== nothing
                reset!(ogm.optimizer.line_search)
            end
        end

        it_criterion = ogm.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(ogm.optimizer.ls_it_max))")
        end

        while !check_convergence(ogm.optimizer)
            if ogm.optimizer.tolerance > 0
                ogm.optimizer.x_old_tol = copy(ogm.optimizer.x)
            end
            step!(ogm)
            save_checkpoint!(ogm.optimizer)

            if tqdm_iterations && ogm.optimizer.it % 100 == 0
                println("Iteration: $(ogm.optimizer.it)")
            end
        end

        append_seed_results!(ogm.optimizer.trace, seed)
        push!(ogm.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(ogm.optimizer.finished_seeds))/$(length(ogm.optimizer.seeds))")
        end
    end

    ogm.optimizer.seed = nothing
    return ogm.optimizer.trace
end