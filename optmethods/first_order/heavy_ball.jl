include("../optimizer.jl")

"""
Gradient descent with Polyak's heavy-ball momentum
For details, see, e.g., https://vsokolov.org/courses/750/files/polyak64.pdf

Arguments:
    lr (float, optional): an estimate of the inverse smoothness constant
    momentum (float, optional): momentum value. For quadratics,
        it should be close to 1-sqrt(λ_min/λ_max), where λ_min and
        λ_max are the smallest/largest eigenvalues of the quadratic matrix
"""
mutable struct HeavyBall
    optimizer::Optimizer
    lr::Union{Float64, Nothing}
    momentum::Union{Float64, Nothing}
    strongly_convex::Bool

    # Internal state
    x_old::Vector{Float64}
    grad::Vector{Float64}

    function HeavyBall(loss; lr=nothing, strongly_convex=false, momentum=nothing, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if momentum !== nothing && momentum < 0
            error("Invalid momentum: $momentum")
        end

        new(optimizer, lr, momentum, strongly_convex, Float64[], Float64[])
    end
end

function step!(hb::HeavyBall)
    if !hb.strongly_convex
        hb.momentum = hb.optimizer.it / (hb.optimizer.it + 1)
    end

    x_copy = copy(hb.optimizer.x)
    hb.grad = gradient(hb.optimizer.loss, hb.optimizer.x)

    if hb.optimizer.use_prox
        hb.optimizer.x = prox(hb.optimizer.loss.regularizer,
                             hb.optimizer.x .- hb.lr .* hb.grad, hb.lr) .+
                        hb.momentum .* (hb.optimizer.x .- hb.x_old)
    else
        hb.optimizer.x = hb.optimizer.x .- hb.lr .* hb.grad .+
                        hb.momentum .* (hb.optimizer.x .- hb.x_old)
    end

    hb.x_old = x_copy
end

function init_run!(hb::HeavyBall, x0; kwargs...)
    init_run!(hb.optimizer, x0; kwargs...)

    if hb.lr === nothing
        hb.lr = 1.0 / smoothness(hb.optimizer.loss)
    end

    hb.x_old = copy(hb.optimizer.x)
end

function run!(hb::HeavyBall, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(hb.optimizer.label): The number of iterations is set to $it_max.")
    end

    hb.optimizer.t_max = t_max
    hb.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(hb.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(hb.optimizer.seeds)) seeds...")
    end

    for seed in hb.optimizer.seeds
        if seed in hb.optimizer.finished_seeds
            continue
        end

        hb.optimizer.rng = MersenneTwister(seed)
        hb.optimizer.seed = seed
        loss_seed = rand(hb.optimizer.rng, 1:MAX_SEED)
        set_seed!(hb.optimizer.loss, loss_seed)
        init_seed!(hb.optimizer.trace)

        if ls_it_max === nothing
            hb.optimizer.ls_it_max = it_max
        else
            hb.optimizer.ls_it_max = ls_it_max
        end

        if !hb.optimizer.initialized[seed]
            init_run!(hb, x0)
            hb.optimizer.initialized[seed] = true
            if hb.optimizer.line_search !== nothing
                reset!(hb.optimizer.line_search)
            end
        end

        it_criterion = hb.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(hb.optimizer.ls_it_max))")
        end

        while !check_convergence(hb.optimizer)
            if hb.optimizer.tolerance > 0
                hb.optimizer.x_old_tol = copy(hb.optimizer.x)
            end
            step!(hb)
            save_checkpoint!(hb.optimizer)

            if tqdm_iterations && hb.optimizer.it % 100 == 0
                println("Iteration: $(hb.optimizer.it)")
            end
        end

        append_seed_results!(hb.optimizer.trace, seed)
        push!(hb.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(hb.optimizer.finished_seeds))/$(length(hb.optimizer.seeds))")
        end
    end

    hb.optimizer.seed = nothing
    return hb.optimizer.trace
end