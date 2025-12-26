include("../optimizer.jl")

"""
    Ig(loss; prox_every_it=false, lr0=nothing, lr_max=Inf, lr_decay_coef=0,
       lr_decay_power=1, epoch_start_decay=nothing, batch_size=1,
       update_trace_at_epoch_end=true, kwargs...)

Incremental Gradient (IG) descent with decreasing or constant learning rate.

For a formal description and convergence guarantees, see Section 10 in
https://arxiv.org/abs/2006.05988

The method is sensitive to finishing the final epoch, so it will terminate earlier
than it_max if it_max is not divisible by the number of steps per epoch.

# Arguments
- `loss::Oracle`: Optimization oracle
- `prox_every_it::Bool=false`: Use proximal operation every iteration or only at epoch end
- `lr0::Union{Float64,Nothing}=nothing`: Inverse smoothness estimate for first epoch_start_decay epochs
- `lr_max::Float64=Inf`: Maximum step-size
- `lr_decay_coef::Float64=0`: Coefficient for step-size decay (μ/3 for strongly convex)
- `lr_decay_power::Float64=1`: Power for step-size decay
- `epoch_start_decay::Union{Int,Nothing}=nothing`: Epochs before decay starts
- `batch_size::Int=1`: Number of samples per iteration
- `update_trace_at_epoch_end::Bool=true`: Only save progress at epoch end
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
ig = Ig(loss, lr_decay_coef=0.01, batch_size=10)
trace = run!(ig, x0, it_max=1000)
```
"""
mutable struct Ig
    optimizer::Optimizer
    prox_every_it::Bool
    lr0::Union{Float64, Nothing}
    lr_max::Float64
    lr_decay_coef::Float64
    lr_decay_power::Float64
    epoch_start_decay::Int
    batch_size::Int
    update_trace_at_epoch_end::Bool

    # Internal state
    steps_per_epoch::Int
    i::Int
    finished_epochs::Int
    lr::Float64
    grad::Vector{Float64}

    function Ig(loss; prox_every_it=false, lr0=nothing, lr_max=Inf, lr_decay_coef=0.0,
                lr_decay_power=1.0, epoch_start_decay=nothing, batch_size=1,
                update_trace_at_epoch_end=true, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        # Calculate epoch_start_decay if not provided
        # Default to 1 if not specified (no decay initially)
        epoch_start_decay_val = epoch_start_decay === nothing ? 1 : epoch_start_decay

        steps_per_epoch = Int(ceil(loss.n / batch_size))

        new(optimizer, prox_every_it, lr0, lr_max, lr_decay_coef, lr_decay_power,
            epoch_start_decay_val, batch_size, update_trace_at_epoch_end,
            steps_per_epoch, 0, 0, 0.0, Float64[])
    end
end

function step!(ig::Ig)
    i_max = min(ig.optimizer.loss.n, ig.i + ig.batch_size)
    idx = collect((ig.i + 1):i_max)  # Julia uses 1-based indexing
    ig.i += ig.batch_size

    if ig.i >= ig.optimizer.loss.n
        ig.i = 0
    end

    normalization = ig.optimizer.loss.n / ig.steps_per_epoch
    ig.grad = stochastic_gradient(ig.optimizer.loss, ig.optimizer.x;
                                  idx=idx, normalization=normalization)

    denom_const = 1.0 / ig.lr0
    it_decrease = ig.steps_per_epoch * max(0, ig.finished_epochs - ig.epoch_start_decay)
    lr_decayed = 1.0 / (denom_const + ig.lr_decay_coef * it_decrease^ig.lr_decay_power)
    ig.lr = min(lr_decayed, ig.lr_max)

    ig.optimizer.x .-= ig.lr .* ig.grad

    end_of_epoch = (ig.i == 0)
    if end_of_epoch
        ig.finished_epochs += 1
    end

    if ig.prox_every_it && ig.optimizer.use_prox
        ig.optimizer.x = prox(ig.optimizer.loss.regularizer, ig.optimizer.x, ig.lr)
    elseif end_of_epoch && ig.optimizer.use_prox
        ig.optimizer.x = prox(ig.optimizer.loss.regularizer, ig.optimizer.x,
                             ig.lr * ig.steps_per_epoch)
    end
end

function should_update_trace_ig(ig::Ig)
    if !ig.update_trace_at_epoch_end
        return should_update_trace(ig.optimizer)
    end
    # Only update trace at end of epoch
    return ig.i == 0
end

function init_run!(ig::Ig, x0; kwargs...)
    init_run!(ig.optimizer, x0; kwargs...)

    ig.finished_epochs = 0
    if ig.lr0 === nothing
        ig.lr0 = 1.0 / batch_smoothness(ig.optimizer.loss, ig.batch_size)
    end
    ig.i = 0
    ig.lr = ig.lr0
end

function run!(ig::Ig, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(ig.optimizer.label): The number of iterations is set to $it_max.")
    end

    ig.optimizer.t_max = t_max
    ig.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(ig.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(ig.optimizer.seeds)) seeds...")
    end

    for seed in ig.optimizer.seeds
        if seed in ig.optimizer.finished_seeds
            continue
        end

        ig.optimizer.rng = MersenneTwister(seed)
        ig.optimizer.seed = seed
        loss_seed = rand(ig.optimizer.rng, 1:MAX_SEED)
        set_seed!(ig.optimizer.loss, loss_seed)
        init_seed!(ig.optimizer.trace)

        if ls_it_max === nothing
            ig.optimizer.ls_it_max = it_max
        else
            ig.optimizer.ls_it_max = ls_it_max
        end

        if !ig.optimizer.initialized[seed]
            init_run!(ig, x0)
            ig.optimizer.initialized[seed] = true
            if ig.optimizer.line_search !== nothing
                reset!(ig.optimizer.line_search)
            end
        end

        it_criterion = ig.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(ig.optimizer.ls_it_max))")
        end

        while !check_convergence(ig.optimizer)
            if ig.optimizer.tolerance > 0
                ig.optimizer.x_old_tol = copy(ig.optimizer.x)
            end
            step!(ig)

            # Use custom trace update logic for IG
            if should_update_trace_ig(ig)
                update_trace!(ig.optimizer)
            end

            # Increment iteration counter
            ig.optimizer.it += 1
            ig.optimizer.t = time() - ig.optimizer.t_start

            if tqdm_iterations && ig.optimizer.it % 100 == 0
                println("Iteration: $(ig.optimizer.it)")
            end
        end

        append_seed_results!(ig.optimizer.trace, seed)
        push!(ig.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(ig.optimizer.finished_seeds))/$(length(ig.optimizer.seeds))")
        end
    end

    ig.optimizer.seed = nothing
    return ig.optimizer.trace
end
