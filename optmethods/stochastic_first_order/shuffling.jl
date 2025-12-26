include("../optimizer.jl")

"""
    Shuffling(loss; reshuffle=false, prox_every_it=false, lr0=nothing,
              lr_max=Inf, lr_decay_coef=0.0, lr_decay_power=1.0,
              epoch_start_decay=1, batch_size=1, importance_sampling=false,
              update_trace_at_epoch_end=true, kwargs...)

Shuffling-based stochastic gradient descent with decreasing or constant learning rate.

For formal description and convergence guarantees, see:
Mishchenko et al. (2020) "Random Reshuffling: Simple Analysis with Vast Improvements"
https://arxiv.org/abs/2006.05988

The method is sensitive to finishing the final epoch, so it will terminate earlier
than it_max if it_max is not divisible by the number of steps per epoch.

# Arguments
- `loss::Oracle`: Optimization oracle
- `reshuffle::Bool=false`: Get new permutation for every epoch
  For convex problems, single permutation often suffices
- `prox_every_it::Bool=false`: Apply prox every iteration vs. end of epoch
  Theory supports end-of-epoch application
- `lr0::Union{Float64,Nothing}=nothing`: Learning rate for first epoch_start_decay epochs
- `lr_max::Float64=Inf`: Maximum learning rate
- `lr_decay_coef::Float64=0.0`: Coefficient for learning rate decay
  For strongly convex: use μ/3 where μ is strong convexity constant
- `lr_decay_power::Float64=1.0`: Power for epoch exponentiation in decay
- `epoch_start_decay::Int=1`: Epochs before decay starts
- `batch_size::Int=1`: Samples per iteration
- `importance_sampling::Bool=false`: Use importance sampling for acceleration
- `update_trace_at_epoch_end::Bool=true`: Save progress only at epoch end
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
shuf = Shuffling(loss, lr0=0.01, batch_size=32, reshuffle=true)
trace = run!(shuf, x0, it_max=10000)
```
"""
mutable struct Shuffling
    optimizer::Optimizer
    reshuffle::Bool
    prox_every_it::Bool
    lr0::Union{Float64, Nothing}
    lr::Float64
    lr_max::Float64
    lr_decay_coef::Float64
    lr_decay_power::Float64
    epoch_start_decay::Int
    batch_size::Int
    importance_sampling::Bool
    update_trace_at_epoch_end::Bool

    # Internal state
    steps_per_epoch::Int
    epoch_max::Int
    finished_epochs::Int
    permutation::Vector{Int}
    sampled_permutations::Int
    i::Int  # Current position in permutation
    grad::Vector{Float64}

    # For importance sampling
    sample_counts::Union{Vector{Int}, Nothing}
    idx_with_copies::Union{Vector{Int}, Nothing}
    n_copies::Union{Int, Nothing}

    function Shuffling(loss; reshuffle=false, prox_every_it=false, lr0=nothing,
                      lr_max=Inf, lr_decay_coef=0.0, lr_decay_power=1.0,
                      epoch_start_decay=1, batch_size=1, importance_sampling=false,
                      update_trace_at_epoch_end=true, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        steps_per_epoch = cld(loss.n, batch_size)  # ceiling division
        epoch_max = optimizer.it_max ÷ steps_per_epoch

        if epoch_start_decay == 0 && isfinite(epoch_max)
            epoch_start_decay = 1 + epoch_max ÷ 40
        end

        sample_counts = nothing
        idx_with_copies = nothing
        n_copies = nothing

        if importance_sampling
            individ_smooth = individ_smoothness(loss)
            sample_counts = individ_smooth ./ mean(individ_smooth)
            sample_counts = Int64.(ceil.(sample_counts))
            idx_with_copies = repeat(collect(1:loss.n), inner=sample_counts)
            n_copies = sum(sample_counts)
            steps_per_epoch = cld(n_copies, batch_size)
        end

        new(optimizer, reshuffle, prox_every_it, lr0, 0.0, lr_max, lr_decay_coef,
            lr_decay_power, epoch_start_decay, batch_size, importance_sampling,
            update_trace_at_epoch_end, steps_per_epoch, epoch_max, 0,
            Int[], 0, 0, Float64[], sample_counts, idx_with_copies, n_copies)
    end
end

function step!(shuf::Shuffling)
    if shuf.optimizer.it % shuf.steps_per_epoch == 0
        # Start new epoch
        if shuf.reshuffle
            if !shuf.importance_sampling
                shuf.permutation = randperm(shuf.optimizer.rng, shuf.optimizer.loss.n)
            else
                shuf.permutation = randperm(shuf.optimizer.rng, shuf.n_copies)
                shuf.permutation = shuf.idx_with_copies[shuf.permutation]
            end
            shuf.sampled_permutations += 1
        end
        shuf.i = 1  # Julia uses 1-based indexing
    end

    i_max = min(length(shuf.permutation), shuf.i + shuf.batch_size - 1)
    idx = shuf.permutation[shuf.i:i_max]
    shuf.i += shuf.batch_size

    # Normalization for incomplete minibatches
    if !shuf.importance_sampling
        normalization = shuf.optimizer.loss.n / shuf.steps_per_epoch
    else
        normalization = sum(shuf.sample_counts[idx]) * shuf.n_copies / shuf.steps_per_epoch
    end

    shuf.grad = stochastic_gradient(shuf.optimizer.loss, shuf.optimizer.x,
                                   idx=idx, normalization=normalization)

    denom_const = 1.0 / shuf.lr0
    it_decrease = shuf.steps_per_epoch * max(0, shuf.finished_epochs - shuf.epoch_start_decay)
    lr_decayed = 1.0 / (denom_const + shuf.lr_decay_coef * it_decrease^shuf.lr_decay_power)
    shuf.lr = min(lr_decayed, shuf.lr_max)

    shuf.optimizer.x .-= shuf.lr .* shuf.grad

    end_of_epoch = (shuf.optimizer.it + 1) % shuf.steps_per_epoch == 0

    if end_of_epoch && shuf.optimizer.use_prox
        shuf.optimizer.x = prox(shuf.optimizer.loss.regularizer, shuf.optimizer.x,
                               shuf.lr * shuf.steps_per_epoch)
        shuf.finished_epochs += 1
    end
end

function should_update_trace(shuf::Shuffling)
    if !shuf.update_trace_at_epoch_end
        # Use default behavior
        if shuf.optimizer.it <= shuf.optimizer.save_first_iterations
            return true
        end

        shuf.optimizer.time_progress = Int((shuf.optimizer.trace_len - shuf.optimizer.save_first_iterations) *
                                          shuf.optimizer.t / shuf.optimizer.t_max)
        shuf.optimizer.iterations_progress = Int((shuf.optimizer.trace_len - shuf.optimizer.save_first_iterations) *
                                                (shuf.optimizer.it / shuf.optimizer.it_max))

        enough_progress = max(shuf.optimizer.time_progress, shuf.optimizer.iterations_progress) > shuf.optimizer.max_progress
        return enough_progress
    end

    if shuf.optimizer.it <= shuf.optimizer.save_first_iterations
        return true
    end

    if shuf.optimizer.it % shuf.steps_per_epoch != 0
        return false
    end

    shuf.optimizer.time_progress = Int((shuf.optimizer.trace_len - shuf.optimizer.save_first_iterations) *
                                      shuf.optimizer.t / shuf.optimizer.t_max)
    shuf.optimizer.iterations_progress = Int((shuf.optimizer.trace_len - shuf.optimizer.save_first_iterations) *
                                            (shuf.optimizer.it / shuf.optimizer.it_max))

    enough_progress = max(shuf.optimizer.time_progress, shuf.optimizer.iterations_progress) > shuf.optimizer.max_progress
    return enough_progress
end

function init_run!(shuf::Shuffling, x0; kwargs...)
    init_run!(shuf.optimizer, x0; kwargs...)

    if shuf.lr0 === nothing
        shuf.lr0 = 1.0 / batch_smoothness(shuf.optimizer.loss, shuf.batch_size)
    end

    shuf.finished_epochs = 0
    if !shuf.importance_sampling
        shuf.permutation = randperm(shuf.optimizer.rng, shuf.optimizer.loss.n)
    else
        perm = randperm(shuf.optimizer.rng, shuf.n_copies)
        shuf.permutation = shuf.idx_with_copies[perm]
    end
    shuf.sampled_permutations = 1
    shuf.i = 1
end

function run!(shuf::Shuffling, x0; kwargs...)
    return run!(shuf.optimizer, x0; kwargs...)
end
