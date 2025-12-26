include("../optimizer.jl")
include("../line_search/nest_armijo.jl")

"""
    NestLine(loss; line_search=nothing, lr=nothing, mu=0, start_with_small_momentum=true, kwargs...)

Accelerated gradient descent with line search proposed by Nesterov.

For details, see equation (4.9) in
http://www.optimization-online.org/DB_FILE/2007/09/1784.pdf

The method does not support increasing momentum, which may limit
its efficiency on ill-conditioned problems.

# Arguments
- `loss::Oracle`: Optimization oracle
- `line_search::Union{LineSearch,Nothing}=nothing`: Line search instance (NestArmijo)
- `lr::Union{Float64,Nothing}=nothing`: Inverse smoothness estimate
- `mu::Float64=0`: Strong convexity constant
- `start_with_small_momentum::Bool=true`: Gradually increase momentum
- `kwargs...`: Additional arguments passed to `Optimizer`

# Example
```julia
nest_line = NestLine(loss, mu=0.01)
trace = run!(nest_line, x0, it_max=1000)
```
"""
mutable struct NestLine
    optimizer::Optimizer
    lr::Union{Float64, Nothing}
    mu::Float64
    start_with_small_momentum::Bool

    # Internal state
    v::Vector{Float64}
    A::Float64
    grad::Vector{Float64}

    function NestLine(loss; line_search=nothing, lr=nothing, mu=0.0,
                     start_with_small_momentum=true, kwargs...)
        if mu < 0
            error("Invalid mu: $mu")
        end

        if line_search === nothing
            line_search = NesterovArmijoLineSearch(loss; mu=mu,
                                                  start_with_small_momentum=start_with_small_momentum)
        end

        optimizer = Optimizer(loss; line_search=line_search, kwargs...)
        new(optimizer, lr, mu, start_with_small_momentum,
            Float64[], 0.0, Float64[])
    end
end

function step!(nl::NestLine)
    # Call line search with x, v, and A
    # The line search returns new x and step size a
    nl.optimizer.x, a = nl.optimizer.line_search(x=nl.optimizer.x, v=nl.v, A=nl.A)

    nl.A += a
    nl.grad = gradient(nl.optimizer.loss, nl.optimizer.x)
    nl.v .-= a .* nl.grad
end

function init_run!(nl::NestLine, x0; kwargs...)
    init_run!(nl.optimizer, x0; kwargs...)

    nl.v = copy(nl.optimizer.x)
    nl.A = 0.0
end

function run!(nl::NestLine, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(nl.optimizer.label): The number of iterations is set to $it_max.")
    end

    nl.optimizer.t_max = t_max
    nl.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(nl.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(nl.optimizer.seeds)) seeds...")
    end

    for seed in nl.optimizer.seeds
        if seed in nl.optimizer.finished_seeds
            continue
        end

        nl.optimizer.rng = MersenneTwister(seed)
        nl.optimizer.seed = seed
        loss_seed = rand(nl.optimizer.rng, 1:MAX_SEED)
        set_seed!(nl.optimizer.loss, loss_seed)
        init_seed!(nl.optimizer.trace)

        if ls_it_max === nothing
            nl.optimizer.ls_it_max = it_max
        else
            nl.optimizer.ls_it_max = ls_it_max
        end

        if !nl.optimizer.initialized[seed]
            init_run!(nl, x0)
            nl.optimizer.initialized[seed] = true
            if nl.optimizer.line_search !== nothing
                reset!(nl.optimizer.line_search)
            end
        end

        it_criterion = nl.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(nl.optimizer.ls_it_max))")
        end

        while !check_convergence(nl.optimizer)
            if nl.optimizer.tolerance > 0
                nl.optimizer.x_old_tol = copy(nl.optimizer.x)
            end
            step!(nl)
            save_checkpoint!(nl.optimizer)

            if tqdm_iterations && nl.optimizer.it % 100 == 0
                println("Iteration: $(nl.optimizer.it)")
            end
        end

        append_seed_results!(nl.optimizer.trace, seed)
        push!(nl.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(nl.optimizer.finished_seeds))/$(length(nl.optimizer.seeds))")
        end
    end

    nl.optimizer.seed = nothing
    return nl.optimizer.trace
end
