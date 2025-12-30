include("../optimizer.jl")

"""
    StochasticNewtonCG

Stochastic Newton method with Conjugate Gradient solver.
Uses CG to approximately solve the Newton system Hx = g, avoiding
explicit Hessian inversion. More efficient for large-scale problems.

Arguments:
    lr0 (float, optional): initial learning rate (auto-computed if not provided)
    lr_max (float, optional): maximum learning rate (default: Inf)
    batch_size (int, optional): batch size for gradient computation (default: 1)
    hessian_batch_size (int, optional): batch size for Hessian computation
    cg_maxiter (int, optional): maximum CG iterations (default: 50)
    cg_tol (float, optional): CG convergence tolerance (default: 1e-5)
    regularization (float, optional): Hessian regularization (default: 1e-4)
"""
mutable struct StochasticNewtonCG
    optimizer::Optimizer
    lr0::Union{Float64, Nothing}
    lr::Float64
    lr_max::Float64
    batch_size::Int
    hessian_batch_size::Int
    cg_maxiter::Int
    cg_tol::Float64
    regularization::Float64

    # Internal state
    grad::Vector{Float64}
    direction::Vector{Float64}
    cg_iterations::Int  # Track CG iterations

    function StochasticNewtonCG(loss; lr0=nothing, lr_max=Inf, batch_size=1,
                               hessian_batch_size=nothing, cg_maxiter=50,
                               cg_tol=1e-5, regularization=1e-4, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if hessian_batch_size === nothing
            hessian_batch_size = batch_size
        end

        new(optimizer, lr0, 0.0, lr_max, batch_size, hessian_batch_size,
            cg_maxiter, cg_tol, regularization,
            Float64[], Float64[], 0)
    end
end

function hessian_vector_product(oracle, x::Vector{Float64}, v::Vector{Float64};
                                idx=nothing, batch_size=1, rng=nothing,
                                regularization=1e-4)
    """Compute Hessian-vector product on a mini-batch using finite differences."""
    n = oracle.n

    if idx === nothing
        if rng === nothing
            rng = oracle.rng
        end
        idx = rand(rng, 1:n, batch_size)
    end

    # Use finite difference approximation: H*v ≈ (∇f(x + εv) - ∇f(x)) / ε
    eps = 1e-5
    grad_plus = stochastic_gradient(oracle, x .+ eps .* v, idx=idx, batch_size=batch_size)
    grad = stochastic_gradient(oracle, x, idx=idx, batch_size=batch_size)

    Hv = (grad_plus .- grad) ./ eps
    # Add regularization
    Hv .+= regularization .* v

    return Hv
end

function conjugate_gradient(oracle, x::Vector{Float64}, b::Vector{Float64};
                           idx=nothing, batch_size=1, rng=nothing,
                           maxiter=50, tol=1e-5, regularization=1e-4)
    """
    Solve Hx = b using Conjugate Gradient method with Hessian-vector products.

    Returns:
        solution: approximate solution to Hx = b
        iterations: number of CG iterations performed
    """
    dim = length(x)
    solution = zeros(dim)
    r = copy(b)  # residual
    p = copy(r)  # search direction
    rs_old = dot(r, r)

    for it in 1:maxiter
        # Compute Hessian-vector product
        Ap = hessian_vector_product(oracle, x, p, idx=idx, batch_size=batch_size,
                                   rng=rng, regularization=regularization)

        pAp = dot(p, Ap)
        if abs(pAp) < 1e-12
            # Avoid division by zero
            return solution, it
        end

        alpha = rs_old / pAp
        solution .+= alpha .* p
        r .-= alpha .* Ap

        rs_new = dot(r, r)

        # Check convergence
        if sqrt(rs_new) < tol
            return solution, it
        end

        beta = rs_new / rs_old
        p = r .+ beta .* p
        rs_old = rs_new
    end

    return solution, maxiter
end

function step!(sncg::StochasticNewtonCG)
    # Compute stochastic gradient
    sncg.grad = stochastic_gradient(sncg.optimizer.loss, sncg.optimizer.x,
                                   batch_size=sncg.batch_size,
                                   rng=sncg.optimizer.rng)

    # Sample indices for Hessian computation (reuse for all Hv products)
    idx = rand(sncg.optimizer.rng, 1:sncg.optimizer.loss.n, sncg.hessian_batch_size)

    # Solve Hx = g using Conjugate Gradient
    sncg.direction, cg_it = conjugate_gradient(
        sncg.optimizer.loss, sncg.optimizer.x, sncg.grad,
        idx=idx, batch_size=sncg.hessian_batch_size,
        rng=sncg.optimizer.rng,
        maxiter=sncg.cg_maxiter, tol=sncg.cg_tol,
        regularization=sncg.regularization
    )
    sncg.cg_iterations = cg_it

    # Update parameters
    sncg.optimizer.x .-= sncg.lr .* sncg.direction

    # Apply proximal operator if needed
    if sncg.optimizer.use_prox
        sncg.optimizer.x = prox(sncg.optimizer.loss.regularizer, sncg.optimizer.x, sncg.lr)
    end
end

function init_run!(sncg::StochasticNewtonCG, x0; kwargs...)
    init_run!(sncg.optimizer, x0; kwargs...)

    # Initialize learning rate if not provided
    if sncg.lr0 === nothing
        sncg.lr0 = 1.0 / batch_smoothness(sncg.optimizer.loss, sncg.batch_size)
    end
    sncg.lr = sncg.lr0

    # Initialize internal state
    dim = length(sncg.optimizer.x)
    sncg.grad = zeros(dim)
    sncg.direction = zeros(dim)
    sncg.cg_iterations = 0
end

function run!(sncg::StochasticNewtonCG, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
             tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(sncg.optimizer.label): The number of iterations is set to $it_max.")
    end

    sncg.optimizer.t_max = t_max
    sncg.optimizer.it_max = it_max

    # Use only the first seed for stochastic methods
    seed = sncg.optimizer.seeds[1]
    if seed in sncg.optimizer.finished_seeds
        return sncg.optimizer.trace
    end

    sncg.optimizer.rng = MersenneTwister(seed)
    sncg.optimizer.seed = seed
    loss_seed = rand(sncg.optimizer.rng, 1:MAX_SEED)
    set_seed!(sncg.optimizer.loss, loss_seed)
    init_seed!(sncg.optimizer.trace)

    if ls_it_max === nothing
        sncg.optimizer.ls_it_max = it_max
    else
        sncg.optimizer.ls_it_max = ls_it_max
    end

    if !sncg.optimizer.initialized[seed]
        init_run!(sncg, x0)
        sncg.optimizer.initialized[seed] = true
        if sncg.optimizer.line_search !== nothing
            reset!(sncg.optimizer.line_search, sncg.optimizer)
        end
    end

    if tqdm_iterations
        println("Starting optimization with max iterations: $(Int(sncg.optimizer.ls_it_max))")
    end

    total_cg_iterations = 0
    while !check_convergence(sncg.optimizer)
        if sncg.optimizer.tolerance > 0
            sncg.optimizer.x_old_tol = copy(sncg.optimizer.x)
        end
        step!(sncg)
        total_cg_iterations += sncg.cg_iterations
        save_checkpoint!(sncg.optimizer)

        if tqdm_iterations && sncg.optimizer.it % 100 == 0
            avg_cg = total_cg_iterations / sncg.optimizer.it
            println("Iteration: $(sncg.optimizer.it), Avg CG iters: $(round(avg_cg, digits=2))")
        end
    end

    append_seed_results!(sncg.optimizer.trace, seed)
    push!(sncg.optimizer.finished_seeds, seed)
    sncg.optimizer.seed = nothing

    println("Total CG iterations: $total_cg_iterations")
    return sncg.optimizer.trace
end
