include("../optimizer.jl")

"""
    GradientDescent(loss; lr=nothing, kwargs...)

Standard gradient descent optimizer with constant learning rate or line search.

# Arguments
- `loss::Oracle`: Optimization oracle
- `lr::Union{Float64,Nothing}=nothing`: Learning rate (defaults to 1/L where L is smoothness)
- `kwargs...`: Additional arguments passed to `Optimizer` (line_search, tolerance, etc.)

# Example
```julia
gd = GradientDescent(loss, lr=0.01)
trace = run!(gd, x0, it_max=1000)
```
"""
mutable struct GradientDescent
    # Inherit from Optimizer functionality
    optimizer::Optimizer
    lr::Union{Float64, Nothing}
    grad::Vector{Float64}

    function GradientDescent(loss; lr=nothing, kwargs...)
        optimizer = Optimizer(loss; kwargs...)
        new(optimizer, lr, Float64[])
    end
end

function step!(gd::GradientDescent)
    gd.grad = gradient(gd.optimizer.loss, gd.optimizer.x)

    if gd.optimizer.line_search === nothing
        gd.optimizer.x .-= gd.lr .* gd.grad
        if gd.optimizer.use_prox
            gd.optimizer.x = prox(gd.optimizer.loss.regularizer, gd.optimizer.x, gd.lr)
        end
    else
        gd.optimizer.x = gd.optimizer.line_search(x=gd.optimizer.x, direction=-gd.grad)
    end
end

function init_run!(gd::GradientDescent, x0; kwargs...)
    init_run!(gd.optimizer, x0; kwargs...)
    if gd.lr === nothing
        gd.lr = 1.0 / smoothness(gd.optimizer.loss)
    end
end

# Implement run! for GradientDescent
function run!(gd::GradientDescent, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(gd.optimizer.label): The number of iterations is set to $it_max.")
    end

    gd.optimizer.t_max = t_max
    gd.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(gd.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(gd.optimizer.seeds)) seeds...")
    end

    for seed in gd.optimizer.seeds
        if seed in gd.optimizer.finished_seeds
            continue
        end

        gd.optimizer.rng = MersenneTwister(seed)
        gd.optimizer.seed = seed
        loss_seed = rand(gd.optimizer.rng, 1:MAX_SEED)
        set_seed!(gd.optimizer.loss, loss_seed)
        init_seed!(gd.optimizer.trace)

        if ls_it_max === nothing
            gd.optimizer.ls_it_max = it_max
        else
            gd.optimizer.ls_it_max = ls_it_max
        end

        if !gd.optimizer.initialized[seed]
            init_run!(gd, x0)
            gd.optimizer.initialized[seed] = true
            if gd.optimizer.line_search !== nothing
                reset!(gd.optimizer.line_search)
            end
        end

        it_criterion = gd.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(gd.optimizer.ls_it_max))")
        end

        while !check_convergence(gd.optimizer)
            if gd.optimizer.tolerance > 0
                gd.optimizer.x_old_tol = copy(gd.optimizer.x)
            end
            step!(gd)  # Call GradientDescent's step!
            save_checkpoint!(gd.optimizer)

            if tqdm_iterations && gd.optimizer.it % 100 == 0
                println("Iteration: $(gd.optimizer.it)")
            end
        end

        append_seed_results!(gd.optimizer.trace, seed)
        push!(gd.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(gd.optimizer.finished_seeds))/$(length(gd.optimizer.seeds))")
        end
    end

    gd.optimizer.seed = nothing
    return gd.optimizer.trace
end
