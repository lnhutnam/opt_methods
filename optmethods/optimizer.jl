using Random, LinearAlgebra, Statistics
include("opt_trace.jl")

# Constants for random seed generation
const SEED = 42
const MAX_SEED = 10_000_000_000
const MAX_TOTAL_SEEDS = 1000

"""
    Optimizer(loss; kwargs...)

Base type for optimization algorithms. Implements the optimization loop,
saves the trace of each run, and supports multiple random seeds for
stochastic/randomized algorithms.

# Arguments
- `loss::Oracle`: Oracle instance providing gradients, loss values, etc.
- `trace_len::Int=200`: Number of checkpoints stored in trace
- `use_prox::Bool=true`: Whether to use proximal operator for regularizer
- `tolerance::Float64=0.0`: Stationarity tolerance (uses iterate difference)
- `line_search::Union{LineSearch,Nothing}=nothing`: Line search instance for stepsize tuning
- `save_first_iterations::Int=5`: Number of initial iterations to always save
- `label::Union{String,Nothing}=nothing`: Label for trace and plotting
- `n_seeds::Int=1`: Number of random seeds to run (ignored if seeds provided)
- `seeds::Union{Vector{Int},Nothing}=nothing`: Explicit random seeds to use
- `tqdm::Bool=true`: Whether to show progress during optimization
"""
mutable struct Optimizer
    loss  # Oracle type, but can't annotate due to load order
    trace_len::Int
    use_prox::Bool
    tolerance::Float64
    line_search  # LineSearch type, but can't annotate due to load order
    save_first_iterations::Int
    label::Union{String, Nothing}
    tqdm::Bool

    # Internal state
    initialized::Dict{Int, Bool}
    x_old_tol::Union{Vector{Float64}, Nothing}
    trace::Trace
    seeds::Vector{Int}
    n_seeds::Int
    finished_seeds::Vector{Int}
    seed::Union{Int, Nothing}
    rng::AbstractRNG

    # Runtime state
    dim::Int
    x::Vector{Float64}
    it::Int
    t::Float64
    t_start::Float64
    time_progress::Int
    iterations_progress::Int
    max_progress::Int
    t_max::Float64
    it_max::Float64  # Can be Inf
    ls_it_max::Float64  # Can be Inf
    ls_it::Int

    function Optimizer(loss; trace_len=200, use_prox=true, tolerance=0.0, line_search=nothing,
                      save_first_iterations=5, label=nothing, n_seeds=1, seeds=nothing, tqdm=true)

        use_prox = use_prox && (loss.regularizer !== nothing)

        if n_seeds > MAX_TOTAL_SEEDS
            error("At most $MAX_TOTAL_SEEDS random seeds are supported.")
        end

        if seeds === nothing
            rng = MersenneTwister(SEED)
            # to make sure we get the same random seeds, we generate a lot of them
            # and take only the first n_seeds
            all_seeds = rand(rng, 1:MAX_SEED, MAX_TOTAL_SEEDS)
            seeds = unique(all_seeds)[1:n_seeds]
        end

        n_seeds = length(seeds)
        initialized = Dict(seed => false for seed in seeds)
        trace = Trace(loss, label)

        new(loss, trace_len, use_prox, tolerance, line_search, save_first_iterations,
            label, tqdm, initialized, nothing, trace, seeds, n_seeds, Int[], nothing,
            MersenneTwister(), 0, Float64[], 0, 0.0, 0.0, 0, 0, 0, Inf, Inf, Inf, 0)
    end
end

function run!(optimizer::Optimizer, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(optimizer.label): The number of iterations is set to $it_max.")
    end

    optimizer.t_max = t_max
    optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(optimizer.seeds)) seeds...")
    end

    for seed in optimizer.seeds
        if seed in optimizer.finished_seeds
            continue
        end

        optimizer.rng = MersenneTwister(seed)
        optimizer.seed = seed
        loss_seed = rand(optimizer.rng, 1:MAX_SEED)
        set_seed!(optimizer.loss, loss_seed)
        init_seed!(optimizer.trace)

        if ls_it_max === nothing
            optimizer.ls_it_max = it_max
        else
            optimizer.ls_it_max = ls_it_max
        end

        if !optimizer.initialized[seed]
            init_run!(optimizer, x0)
            optimizer.initialized[seed] = true
            if optimizer.line_search !== nothing
                reset!(optimizer.line_search)
            end
        end

        it_criterion = optimizer.ls_it_max != Inf
        if tqdm_iterations
            tqdm_total = it_criterion ? optimizer.ls_it_max : optimizer.t_max
            tqdm_val = 0
            if it_criterion
                println("Starting optimization with max iterations: $(Int(tqdm_total))")
            end
        end

        while !check_convergence(optimizer)
            if optimizer.tolerance > 0
                optimizer.x_old_tol = copy(optimizer.x)
            end
            step!(optimizer)
            save_checkpoint!(optimizer)

            if tqdm_iterations && optimizer.it % 100 == 0
                println("Iteration: $(optimizer.it)")
            end
        end

        append_seed_results!(optimizer.trace, seed)
        push!(optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(optimizer.finished_seeds))/$(length(optimizer.seeds))")
        end
    end

    optimizer.seed = nothing
    return optimizer.trace
end

function add_seeds!(optimizer::Optimizer, n_extra_seeds=1, extra_seeds=nothing)
    rng = MersenneTwister(SEED)
    if extra_seeds === nothing
        n_seeds = length(optimizer.seeds) + n_extra_seeds
        # we generate a lot of random seeds so that the behaviour is the same
        # when we add one seed twice as when we add two seeds
        all_seeds = rand(rng, 1:MAX_SEED, MAX_TOTAL_SEEDS)
        extra_seeds = unique(all_seeds)[n_seeds-n_extra_seeds+1:n_seeds]
    end
    append!(optimizer.seeds, extra_seeds)
    optimizer.trace.loss_is_computed = false
    if optimizer.trace.its_converted_to_epochs
        # TODO: create a bool variable for each seed
    end
end

function check_convergence(optimizer::Optimizer)
    no_it_left = optimizer.it >= optimizer.it_max
    if optimizer.line_search !== nothing
        no_it_left = no_it_left || (optimizer.line_search.it >= optimizer.ls_it_max)
    end
    no_time_left = time() - optimizer.t_start >= optimizer.t_max

    if optimizer.tolerance > 0
        tolerance_met = optimizer.x_old_tol !== nothing &&
                       norm(optimizer.x - optimizer.x_old_tol) < optimizer.tolerance
    else
        tolerance_met = false
    end
    return no_it_left || no_time_left || tolerance_met
end

function step!(optimizer::Optimizer)
    error("step! method must be implemented for the specific optimizer algorithm. " *
          "This is a base Optimizer that should be wrapped by a specific algorithm like GradientDescent.")
end

function init_run!(optimizer::Optimizer, x0)
    optimizer.dim = length(x0)
    optimizer.x = copy(x0)
    optimizer.trace.xs = [copy(x0)]
    optimizer.trace.its = [0]
    optimizer.trace.ts = [0.0]

    if optimizer.line_search !== nothing
        optimizer.trace.ls_its = [0]
        optimizer.trace.lrs = [optimizer.line_search.lr]
    end

    optimizer.it = 0
    optimizer.t = 0.0
    optimizer.t_start = time()
    optimizer.time_progress = 0
    optimizer.iterations_progress = 0
    optimizer.max_progress = 0

    if optimizer.line_search !== nothing
        reset!(optimizer.line_search, optimizer)
    end
end

function should_update_trace(optimizer::Optimizer)
    if optimizer.it <= optimizer.save_first_iterations
        return true
    end

    # Handle Inf values for t_max and it_max
    if isfinite(optimizer.t_max)
        optimizer.time_progress = Int(floor((optimizer.trace_len - optimizer.save_first_iterations) *
                                           optimizer.t / optimizer.t_max))
    else
        optimizer.time_progress = 0
    end

    if isfinite(optimizer.it_max)
        optimizer.iterations_progress = Int(floor((optimizer.trace_len - optimizer.save_first_iterations) *
                                                  (optimizer.it / optimizer.it_max)))
    else
        optimizer.iterations_progress = 0
    end

    if optimizer.line_search !== nothing && isfinite(optimizer.it_max)
        ls_it = optimizer.line_search.it
        optimizer.iterations_progress = max(optimizer.iterations_progress,
                                           Int(floor((optimizer.trace_len - optimizer.save_first_iterations) *
                                                    (ls_it / optimizer.it_max))))
    end

    enough_progress = max(optimizer.time_progress, optimizer.iterations_progress) > optimizer.max_progress
    return enough_progress
end

function save_checkpoint!(optimizer::Optimizer)
    optimizer.it += 1
    if optimizer.line_search !== nothing
        optimizer.ls_it = optimizer.line_search.it
    end
    optimizer.t = time() - optimizer.t_start

    if should_update_trace(optimizer)
        update_trace!(optimizer)
    end
    optimizer.max_progress = max(optimizer.time_progress, optimizer.iterations_progress)
end

function update_trace!(optimizer::Optimizer)
    push!(optimizer.trace.xs, copy(optimizer.x))
    push!(optimizer.trace.ts, optimizer.t)
    push!(optimizer.trace.its, optimizer.it)

    if optimizer.line_search !== nothing
        push!(optimizer.trace.ls_its, optimizer.line_search.it)
        push!(optimizer.trace.lrs, optimizer.line_search.lr)
    end
end

function compute_loss_of_iterates!(optimizer::Optimizer)
    compute_loss_of_iterates!(optimizer.trace)
end

function reset!(optimizer::Optimizer)
    optimizer.initialized = Dict(key => false for key in optimizer.seeds)
    optimizer.x_old_tol = nothing
    optimizer.trace = Trace(optimizer.loss, optimizer.label)
end