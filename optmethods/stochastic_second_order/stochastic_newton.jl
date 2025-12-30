include("../optimizer.jl")

"""
    StochasticNewton

Stochastic Newton method using subsampled Hessians and gradients.
This method computes the Newton direction using a stochastic approximation
of both the gradient and Hessian on mini-batches.

Arguments:
    lr0 (float, optional): initial learning rate (auto-computed if not provided)
    lr_max (float, optional): maximum learning rate (default: Inf)
    lr_decay_coef (float, optional): learning rate decay coefficient (default: 0.0)
    lr_decay_power (float, optional): learning rate decay power (default: 1.0)
    batch_size (int, optional): batch size for gradient computation (default: 1)
    hessian_batch_size (int, optional): batch size for Hessian computation
        (default: same as batch_size)
    regularization (float, optional): Hessian regularization parameter (default: 1e-4)
    adaptive_reg (bool, optional): use adaptive regularization (default: true)
"""
mutable struct StochasticNewton
    optimizer::Optimizer
    lr0::Union{Float64, Nothing}
    lr::Float64
    lr_max::Float64
    lr_decay_coef::Float64
    lr_decay_power::Float64
    batch_size::Int
    hessian_batch_size::Int
    regularization::Float64
    adaptive_reg::Bool

    # Internal state
    grad::Vector{Float64}
    hess::Matrix{Float64}
    direction::Vector{Float64}

    function StochasticNewton(loss; lr0=nothing, lr_max=Inf, lr_decay_coef=0.0,
                             lr_decay_power=1.0, batch_size=1, hessian_batch_size=nothing,
                             regularization=1e-4, adaptive_reg=true, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if hessian_batch_size === nothing
            hessian_batch_size = batch_size
        end

        new(optimizer, lr0, 0.0, lr_max, lr_decay_coef, lr_decay_power,
            batch_size, hessian_batch_size, regularization, adaptive_reg,
            Float64[], zeros(0, 0), Float64[])
    end
end

function stochastic_hessian(oracle, x::Vector{Float64}; idx=nothing,
                           batch_size=1, rng=nothing)
    """Compute stochastic Hessian on a mini-batch."""
    n = oracle.n
    dim = oracle.dim

    if idx === nothing
        if rng === nothing
            rng = oracle.rng
        end
        idx = rand(rng, 1:n, batch_size)
    end

    # For logistic regression and similar models
    if isdefined(oracle, :A) && isdefined(oracle, :b)
        A_batch = oracle.A[idx, :]

        if nameof(typeof(oracle)) == :LogisticRegression
            Ax = A_batch * x
            activation = 1 ./ (1 .+ exp.(-Ax))  # sigmoid
            weights = activation .* (1 .- activation)
            A_weighted = A_batch' .* weights'
            return A_weighted * A_batch ./ batch_size .+ oracle.l2 .* I(dim)
        elseif nameof(typeof(oracle)) == :LinearRegression
            return A_batch' * A_batch ./ batch_size .+ oracle.l2 .* I(dim)
        end
    end

    # Fallback: use finite differences for Hessian approximation
    # This is expensive but works for any loss function
    return hessian(oracle, x)
end

function step!(sn::StochasticNewton)
    # Update learning rate with decay
    it = sn.optimizer.it
    sn.lr = sn.lr0 / (1.0 + sn.lr_decay_coef * it^sn.lr_decay_power)
    sn.lr = min(sn.lr, sn.lr_max)

    # Compute stochastic gradient
    sn.grad = stochastic_gradient(sn.optimizer.loss, sn.optimizer.x,
                                 batch_size=sn.batch_size,
                                 rng=sn.optimizer.rng)

    # Compute stochastic Hessian
    sn.hess = stochastic_hessian(sn.optimizer.loss, sn.optimizer.x,
                                batch_size=sn.hessian_batch_size,
                                rng=sn.optimizer.rng)

    # Add regularization to Hessian for numerical stability
    dim = length(sn.optimizer.x)
    reg = sn.regularization

    if sn.adaptive_reg
        # Adaptive regularization based on gradient norm
        grad_norm = norm(sn.grad)
        reg = sn.regularization * (1.0 + grad_norm)
    end

    hess_reg = sn.hess .+ reg .* I(dim)

    # Solve for Newton direction: H^{-1} * g
    try
        sn.direction = hess_reg \ sn.grad
    catch e
        # If Hessian is singular, fall back to gradient descent
        @warn "Hessian inversion failed, using gradient direction" maxlog=3
        sn.direction = sn.grad
    end

    # Update parameters
    sn.optimizer.x .-= sn.lr .* sn.direction

    # Apply proximal operator if needed
    if sn.optimizer.use_prox
        sn.optimizer.x = prox(sn.optimizer.loss.regularizer, sn.optimizer.x, sn.lr)
    end
end

function init_run!(sn::StochasticNewton, x0; kwargs...)
    init_run!(sn.optimizer, x0; kwargs...)

    # Initialize learning rate if not provided
    if sn.lr0 === nothing
        sn.lr0 = 1.0 / batch_smoothness(sn.optimizer.loss, sn.batch_size)
    end
    sn.lr = sn.lr0

    # Initialize internal state
    dim = length(sn.optimizer.x)
    sn.grad = zeros(dim)
    sn.hess = zeros(dim, dim)
    sn.direction = zeros(dim)
end

function run!(sn::StochasticNewton, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
             tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(sn.optimizer.label): The number of iterations is set to $it_max.")
    end

    sn.optimizer.t_max = t_max
    sn.optimizer.it_max = it_max

    # Use only the first seed for stochastic methods
    seed = sn.optimizer.seeds[1]
    if seed in sn.optimizer.finished_seeds
        return sn.optimizer.trace
    end

    sn.optimizer.rng = MersenneTwister(seed)
    sn.optimizer.seed = seed
    loss_seed = rand(sn.optimizer.rng, 1:MAX_SEED)
    set_seed!(sn.optimizer.loss, loss_seed)
    init_seed!(sn.optimizer.trace)

    if ls_it_max === nothing
        sn.optimizer.ls_it_max = it_max
    else
        sn.optimizer.ls_it_max = ls_it_max
    end

    if !sn.optimizer.initialized[seed]
        init_run!(sn, x0)
        sn.optimizer.initialized[seed] = true
        if sn.optimizer.line_search !== nothing
            reset!(sn.optimizer.line_search, sn.optimizer)
        end
    end

    if tqdm_iterations
        println("Starting optimization with max iterations: $(Int(sn.optimizer.ls_it_max))")
    end

    while !check_convergence(sn.optimizer)
        if sn.optimizer.tolerance > 0
            sn.optimizer.x_old_tol = copy(sn.optimizer.x)
        end
        step!(sn)
        save_checkpoint!(sn.optimizer)

        if tqdm_iterations && sn.optimizer.it % 100 == 0
            println("Iteration: $(sn.optimizer.it)")
        end
    end

    append_seed_results!(sn.optimizer.trace, seed)
    push!(sn.optimizer.finished_seeds, seed)
    sn.optimizer.seed = nothing

    return sn.optimizer.trace
end
