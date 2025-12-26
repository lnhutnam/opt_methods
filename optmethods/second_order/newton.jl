include("../optimizer.jl")
using Random

"""
Newton algorithm for convex minimization.

Arguments:
    lr (float, optional): dampening constant (default: 1)
"""
mutable struct Newton
    optimizer::Optimizer
    lr::Float64

    # Internal state
    grad::Vector{Float64}
    hess::Matrix{Float64}

    function Newton(loss; lr=1.0, kwargs...)
        optimizer = Optimizer(loss; kwargs...)
        new(optimizer, lr, Float64[], Matrix{Float64}(undef, 0, 0))
    end
end

function step!(newton::Newton)
    newton.grad = gradient(newton.optimizer.loss, newton.optimizer.x)
    newton.hess = hessian(newton.optimizer.loss, newton.optimizer.x)

    # Solve Hess * direction = grad using least squares
    inv_hess_grad_prod = newton.hess \ newton.grad

    if newton.optimizer.line_search === nothing
        newton.optimizer.x .-= newton.lr .* inv_hess_grad_prod
    else
        newton.optimizer.x = newton.optimizer.line_search(gradient=newton.grad, direction=-inv_hess_grad_prod)
    end
end

function init_run!(newton::Newton, x0; kwargs...)
    init_run!(newton.optimizer, x0; kwargs...)

    dim = length(newton.optimizer.x)
    newton.grad = zeros(dim)
    newton.hess = zeros(dim, dim)
end

function run!(newton::Newton, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("Newton: The number of iterations is set to $it_max.")
    end

    newton.optimizer.t_max = t_max
    newton.optimizer.it_max = it_max

    # Use first seed for single-seed run
    seed = newton.optimizer.seeds[1]
    if seed in newton.optimizer.finished_seeds
        return newton.optimizer.trace
    end

    newton.optimizer.rng = MersenneTwister(seed)
    newton.optimizer.seed = seed
    loss_seed = rand(newton.optimizer.rng, 1:100000)
    set_seed!(newton.optimizer.loss, loss_seed)
    init_seed!(newton.optimizer.trace)

    if ls_it_max === nothing
        newton.optimizer.ls_it_max = it_max
    else
        newton.optimizer.ls_it_max = ls_it_max
    end

    if !newton.optimizer.initialized[seed]
        init_run!(newton, x0)
        newton.optimizer.initialized[seed] = true
        if newton.optimizer.line_search !== nothing
            reset!(newton.optimizer.line_search, newton.optimizer)
        end
    end

    while !check_convergence(newton.optimizer)
        if newton.optimizer.tolerance > 0
            newton.optimizer.x_old_tol = copy(newton.optimizer.x)
        end
        step!(newton)
        save_checkpoint!(newton.optimizer)
    end

    append_seed_results!(newton.optimizer.trace, seed)
    push!(newton.optimizer.finished_seeds, seed)
    newton.optimizer.seed = nothing

    return newton.optimizer.trace
end