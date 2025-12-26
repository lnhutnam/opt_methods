include("../optimizer.jl")
include("../line_search/best_grid.jl")

"""
    Shorr(loss; gamma=0.5, L=nothing, kwargs...)

Shor's r-algorithm for quasi-Newton optimization.

For a convergence analysis, see:
https://www.researchgate.net/publication/243084304_The_Speed_of_Shor's_R-algorithm

The method generally requires a line search to work properly. A grid search
starting at lr=1 is recommended.

# Arguments
- `loss::Oracle`: Optimization oracle
- `gamma::Float64=0.5`: Rate of Hessian estimate change, must be in (0, 1)
- `L::Union{Float64,Nothing}=nothing`: Upper bound on smoothness constant
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
shorr = Shorr(loss, gamma=0.5)
trace = run!(shorr, x0, it_max=1000)
```
"""
mutable struct Shorr
    optimizer::Optimizer
    gamma::Float64
    L::Float64

    # Internal state
    B::Matrix{Float64}
    grad::Vector{Float64}
    grad_old::Vector{Float64}

    function Shorr(loss; gamma=0.5, L=nothing, kwargs...)
        if !(0.0 < gamma < 1.0)
            error("Invalid gamma: $gamma. Must be in (0, 1)")
        end

        # Determine L if not provided
        if L === nothing
            L_val = smoothness(loss)
            if L_val === nothing
                L_val = 1.0
            end
        else
            L_val = L
        end

        # Create line search if not provided
        if !haskey(kwargs, :line_search) || kwargs[:line_search] === nothing
            line_search = BestGridLineSearch(lr0=1.0, start_with_prev_lr=false,
                                            increase_many_times=true)
            optimizer = Optimizer(loss; line_search=line_search, kwargs...)
        else
            optimizer = Optimizer(loss; kwargs...)
        end

        # Initialize B matrix
        B = (1.0 / sqrt(L_val)) * Matrix{Float64}(I, loss.dim, loss.dim)

        new(optimizer, gamma, L_val, B, Float64[], Float64[])
    end
end

function step!(shorr::Shorr)
    shorr.grad = gradient(shorr.optimizer.loss, shorr.optimizer.x)

    # Avoid machine precision issues
    if shorr.optimizer.line_search !== nothing && shorr.optimizer.line_search.lr != 1.0
        shorr.B *= sqrt(shorr.optimizer.line_search.lr)
        shorr.optimizer.line_search.lr = 1.0
    end

    r = shorr.B' * (shorr.grad .- shorr.grad_old)
    r_norm = norm(r)
    if r_norm > 0
        r ./= r_norm
    end

    # Update B matrix
    shorr.B .-= shorr.gamma .* shorr.B * (r * r')

    # Compute new point
    x_new = shorr.optimizer.x .- shorr.B * (shorr.B' * shorr.grad)

    if shorr.optimizer.line_search !== nothing
        shorr.optimizer.x = shorr.optimizer.line_search(x=shorr.optimizer.x, x_new=x_new)
    else
        shorr.optimizer.x = x_new
    end

    shorr.grad_old = copy(shorr.grad)
end

function init_run!(shorr::Shorr, x0; kwargs...)
    init_run!(shorr.optimizer, x0; kwargs...)

    # Initialize B matrix with correct dimensions
    dim = length(shorr.optimizer.x)
    shorr.B = (1.0 / sqrt(shorr.L)) * Matrix{Float64}(I, dim, dim)

    # Take initial step
    shorr.grad_old = gradient(shorr.optimizer.loss, shorr.optimizer.x)
    shorr.optimizer.x .-= (1.0 / shorr.L) .* shorr.grad_old
    save_checkpoint!(shorr.optimizer)
end

function run!(shorr::Shorr, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(shorr.optimizer.label): The number of iterations is set to $it_max.")
    end

    shorr.optimizer.t_max = t_max
    shorr.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(shorr.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(shorr.optimizer.seeds)) seeds...")
    end

    for seed in shorr.optimizer.seeds
        if seed in shorr.optimizer.finished_seeds
            continue
        end

        shorr.optimizer.rng = MersenneTwister(seed)
        shorr.optimizer.seed = seed
        loss_seed = rand(shorr.optimizer.rng, 1:MAX_SEED)
        set_seed!(shorr.optimizer.loss, loss_seed)
        init_seed!(shorr.optimizer.trace)

        if ls_it_max === nothing
            shorr.optimizer.ls_it_max = it_max
        else
            shorr.optimizer.ls_it_max = ls_it_max
        end

        if !shorr.optimizer.initialized[seed]
            init_run!(shorr, x0)
            shorr.optimizer.initialized[seed] = true
            if shorr.optimizer.line_search !== nothing
                reset!(shorr.optimizer.line_search, shorr.optimizer)
            end
        end

        it_criterion = shorr.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(shorr.optimizer.ls_it_max))")
        end

        while !check_convergence(shorr.optimizer)
            if shorr.optimizer.tolerance > 0
                shorr.optimizer.x_old_tol = copy(shorr.optimizer.x)
            end
            step!(shorr)
            save_checkpoint!(shorr.optimizer)

            if tqdm_iterations && shorr.optimizer.it % 100 == 0
                println("Iteration: $(shorr.optimizer.it)")
            end
        end

        append_seed_results!(shorr.optimizer.trace, seed)
        push!(shorr.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(shorr.optimizer.finished_seeds))/$(length(shorr.optimizer.seeds))")
        end
    end

    shorr.optimizer.seed = nothing
    return shorr.optimizer.trace
end
