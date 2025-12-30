include("../optimizer.jl")

"""
    NaturalGradient

Natural Gradient Descent using the Fisher Information Matrix.
The natural gradient provides a geometrically invariant update direction
by using the Fisher information matrix (FIM) instead of the Euclidean metric.

For probabilistic models, the FIM captures the geometry of the parameter space
more accurately than the Hessian. For logistic regression and similar models,
the FIM is equivalent to the expected Hessian.

Arguments:
    lr0 (float, optional): initial learning rate (auto-computed if not provided)
    lr_max (float, optional): maximum learning rate (default: Inf)
    lr_decay_coef (float, optional): learning rate decay coefficient (default: 0.0)
    lr_decay_power (float, optional): learning rate decay power (default: 1.0)
    batch_size (int, optional): batch size for gradient computation (default: 1)
    fisher_batch_size (int, optional): batch size for Fisher matrix computation
    regularization (float, optional): FIM regularization parameter (default: 1e-4)
    use_empirical_fisher (bool, optional): use empirical Fisher instead of exact (default: false)
    moving_average (float, optional): moving average coefficient for FIM (default: 0.0, disabled)
"""
mutable struct NaturalGradient
    optimizer::Optimizer
    lr0::Union{Float64, Nothing}
    lr::Float64
    lr_max::Float64
    lr_decay_coef::Float64
    lr_decay_power::Float64
    batch_size::Int
    fisher_batch_size::Int
    regularization::Float64
    use_empirical_fisher::Bool
    moving_average::Float64

    # Internal state
    grad::Vector{Float64}
    fisher::Union{Matrix{Float64}, Nothing}
    fisher_accum::Union{Matrix{Float64}, Nothing}  # For moving average
    direction::Vector{Float64}

    function NaturalGradient(loss; lr0=nothing, lr_max=Inf, lr_decay_coef=0.0,
                            lr_decay_power=1.0, batch_size=1, fisher_batch_size=nothing,
                            regularization=1e-4, use_empirical_fisher=false,
                            moving_average=0.0, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if fisher_batch_size === nothing
            fisher_batch_size = batch_size
        end

        new(optimizer, lr0, 0.0, lr_max, lr_decay_coef, lr_decay_power,
            batch_size, fisher_batch_size, regularization, use_empirical_fisher,
            moving_average,
            Float64[], nothing, nothing, Float64[])
    end
end

function compute_fisher_information(oracle, x::Vector{Float64};
                                   idx=nothing, batch_size=1, rng=nothing,
                                   use_empirical=false)
    """
    Compute Fisher Information Matrix.

    For classification models (logistic regression):
    - Exact Fisher: F = E[∇log p(y|x) * ∇log p(y|x)ᵀ] = A'diag(p(1-p))A
    - Empirical Fisher: F ≈ ∇ℓ * ∇ℓᵀ (outer product of gradients)
    """
    n = oracle.n
    dim = oracle.dim

    if idx === nothing
        if rng === nothing
            rng = oracle.rng
        end
        idx = rand(rng, 1:n, batch_size)
    end

    if use_empirical
        # Empirical Fisher: outer product of gradient
        grad = stochastic_gradient(oracle, x, idx=idx, batch_size=batch_size)
        fisher = grad * grad'
        return fisher
    end

    # Exact Fisher for specific models
    if isdefined(oracle, :A) && isdefined(oracle, :b)
        A_batch = oracle.A[idx, :]

        if nameof(typeof(oracle)) == :LogisticRegression
            # For logistic regression: F = A'diag(p(1-p))A
            Ax = A_batch * x
            activation = 1 ./ (1 .+ exp.(-Ax))  # sigmoid
            weights = activation .* (1 .- activation)
            A_weighted = A_batch' .* weights'
            fisher = A_weighted * A_batch ./ batch_size
            return fisher .+ oracle.l2 .* I(dim)

        elseif nameof(typeof(oracle)) == :LinearRegression
            # For linear regression with Gaussian noise: F = A'A / σ²
            # Assuming unit variance, F = A'A
            fisher = A_batch' * A_batch ./ batch_size
            return fisher .+ oracle.l2 .* I(dim)
        end
    end

    # Fallback: use empirical Fisher
    grad = stochastic_gradient(oracle, x, idx=idx, batch_size=batch_size)
    fisher = grad * grad'
    return fisher
end

function step!(ng::NaturalGradient)
    # Update learning rate with decay
    it = ng.optimizer.it
    ng.lr = ng.lr0 / (1.0 + ng.lr_decay_coef * it^ng.lr_decay_power)
    ng.lr = min(ng.lr, ng.lr_max)

    # Compute stochastic gradient
    ng.grad = stochastic_gradient(ng.optimizer.loss, ng.optimizer.x,
                                 batch_size=ng.batch_size,
                                 rng=ng.optimizer.rng)

    # Compute Fisher Information Matrix
    fisher = compute_fisher_information(ng.optimizer.loss, ng.optimizer.x,
                                       batch_size=ng.fisher_batch_size,
                                       rng=ng.optimizer.rng,
                                       use_empirical=ng.use_empirical_fisher)

    # Apply moving average if enabled
    if ng.moving_average > 0.0
        if ng.fisher_accum === nothing
            ng.fisher_accum = copy(fisher)
        else
            ng.fisher_accum = ng.moving_average .* ng.fisher_accum .+
                             (1.0 - ng.moving_average) .* fisher
        end
        ng.fisher = copy(ng.fisher_accum)
    else
        ng.fisher = fisher
    end

    # Add regularization for numerical stability
    dim = length(ng.optimizer.x)
    fisher_reg = ng.fisher .+ ng.regularization .* I(dim)

    # Compute natural gradient direction: F^{-1} * g
    try
        ng.direction = fisher_reg \ ng.grad
    catch e
        # If Fisher matrix is singular, fall back to gradient descent
        @warn "Fisher matrix inversion failed, using gradient direction" maxlog=3
        ng.direction = ng.grad
    end

    # Update parameters (natural gradient descent)
    ng.optimizer.x .-= ng.lr .* ng.direction

    # Apply proximal operator if needed
    if ng.optimizer.use_prox
        ng.optimizer.x = prox(ng.optimizer.loss.regularizer, ng.optimizer.x, ng.lr)
    end
end

function init_run!(ng::NaturalGradient, x0; kwargs...)
    init_run!(ng.optimizer, x0; kwargs...)

    # Initialize learning rate if not provided
    if ng.lr0 === nothing
        ng.lr0 = 1.0 / batch_smoothness(ng.optimizer.loss, ng.batch_size)
    end
    ng.lr = ng.lr0

    # Initialize internal state
    dim = length(ng.optimizer.x)
    ng.grad = zeros(dim)
    ng.fisher = nothing
    ng.fisher_accum = nothing
    ng.direction = zeros(dim)
end

function run!(ng::NaturalGradient, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
             tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(ng.optimizer.label): The number of iterations is set to $it_max.")
    end

    ng.optimizer.t_max = t_max
    ng.optimizer.it_max = it_max

    # Use only the first seed for stochastic methods
    seed = ng.optimizer.seeds[1]
    if seed in ng.optimizer.finished_seeds
        return ng.optimizer.trace
    end

    ng.optimizer.rng = MersenneTwister(seed)
    ng.optimizer.seed = seed
    loss_seed = rand(ng.optimizer.rng, 1:MAX_SEED)
    set_seed!(ng.optimizer.loss, loss_seed)
    init_seed!(ng.optimizer.trace)

    if ls_it_max === nothing
        ng.optimizer.ls_it_max = it_max
    else
        ng.optimizer.ls_it_max = ls_it_max
    end

    if !ng.optimizer.initialized[seed]
        init_run!(ng, x0)
        ng.optimizer.initialized[seed] = true
        if ng.optimizer.line_search !== nothing
            reset!(ng.optimizer.line_search, ng.optimizer)
        end
    end

    if tqdm_iterations
        println("Starting optimization with max iterations: $(Int(ng.optimizer.ls_it_max))")
    end

    while !check_convergence(ng.optimizer)
        if ng.optimizer.tolerance > 0
            ng.optimizer.x_old_tol = copy(ng.optimizer.x)
        end
        step!(ng)
        save_checkpoint!(ng.optimizer)

        if tqdm_iterations && ng.optimizer.it % 100 == 0
            println("Iteration: $(ng.optimizer.it)")
        end
    end

    append_seed_results!(ng.optimizer.trace, seed)
    push!(ng.optimizer.finished_seeds, seed)
    ng.optimizer.seed = nothing

    return ng.optimizer.trace
end
