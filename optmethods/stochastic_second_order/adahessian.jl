include("../optimizer.jl")

"""
    AdaHessian

Adaptive Hessian-based optimizer combining second-order curvature information
with adaptive learning rates (similar to Adam). Uses diagonal Hessian approximation
for computational efficiency.

Reference:
    Yao, Z., Gholami, A., Keutzer, K., & Mahoney, M. W. (2020).
    "AdaHessian: An Adaptive Second Order Optimizer for Machine Learning"
    arXiv:2006.00719

Key features:
- Uses diagonal Hessian approximation (efficient to compute)
- Maintains exponential moving averages of gradients and Hessian diagonal
- Adaptive per-parameter learning rates based on curvature
- Spatial averaging for better Hessian estimates

Arguments:
    lr (float, optional): learning rate (default: 0.15)
    beta1 (float, optional): coefficient for gradient moving average (default: 0.9)
    beta2 (float, optional): coefficient for Hessian moving average (default: 0.999)
    epsilon (float, optional): small constant for numerical stability (default: 1e-8)
    batch_size (int, optional): batch size for gradient computation (default: 1)
    hessian_batch_size (int, optional): batch size for Hessian computation
    spatial_average (bool, optional): use spatial averaging of Hessian (default: true)
    block_length (int, optional): block length for spatial averaging (default: 1)
"""
mutable struct AdaHessian
    optimizer::Optimizer
    lr::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    batch_size::Int
    hessian_batch_size::Int
    spatial_average::Bool
    block_length::Int

    # Internal state (momentum terms)
    m::Vector{Float64}  # First moment (gradient momentum)
    v::Vector{Float64}  # Second moment (Hessian diagonal momentum)
    t::Int  # Time step

    # Working buffers
    grad::Vector{Float64}
    hess_diag::Vector{Float64}

    function AdaHessian(loss; lr=0.15, beta1=0.9, beta2=0.999, epsilon=1e-8,
                       batch_size=1, hessian_batch_size=nothing,
                       spatial_average=true, block_length=1, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if hessian_batch_size === nothing
            hessian_batch_size = batch_size
        end

        new(optimizer, lr, beta1, beta2, epsilon, batch_size, hessian_batch_size,
            spatial_average, block_length,
            Float64[], Float64[], 0,
            Float64[], Float64[])
    end
end

function compute_hessian_diagonal(oracle, x::Vector{Float64};
                                 idx=nothing, batch_size=1, rng=nothing)
    """
    Compute diagonal of the Hessian matrix efficiently using Hutchinson's trace estimator.

    For a function f, the diagonal can be estimated as:
    diag(H) ≈ E[z ⊙ (∇²f)z] where z ~ N(0, I) and ⊙ is element-wise product

    In practice, we use: diag(H) ≈ (∇f(x + εz) - ∇f(x - εz)) / (2ε) ⊙ z
    where z is a random vector with ±1 entries (Rademacher)
    """
    n = oracle.n
    dim = oracle.dim

    if idx === nothing
        if rng === nothing
            rng = oracle.rng
        end
        idx = rand(rng, 1:n, batch_size)
    end

    # Generate random direction (Rademacher distribution: ±1)
    z = rand(rng, [-1.0, 1.0], dim)

    # Finite difference approximation
    eps = 1e-4
    grad_plus = stochastic_gradient(oracle, x .+ eps .* z, idx=idx, batch_size=batch_size)
    grad_minus = stochastic_gradient(oracle, x .- eps .* z, idx=idx, batch_size=batch_size)

    # Hessian-vector product approximation
    hess_vec = (grad_plus .- grad_minus) ./ (2.0 * eps)

    # Extract diagonal: element-wise product with z
    hess_diag = hess_vec .* z

    # Take absolute value (we want magnitude of curvature)
    hess_diag = abs.(hess_diag)

    return hess_diag
end

function spatial_average_filter(v::Vector{Float64}, block_length::Int)
    """
    Apply spatial averaging to smooth the Hessian diagonal estimates.
    Each element is averaged with its neighbors within block_length.
    """
    if block_length <= 1
        return v
    end

    dim = length(v)
    v_smoothed = copy(v)

    for i in 1:dim
        start_idx = max(1, i - block_length)
        end_idx = min(dim, i + block_length)
        v_smoothed[i] = mean(v[start_idx:end_idx])
    end

    return v_smoothed
end

function step!(adahess::AdaHessian)
    # Increment time step
    adahess.t += 1

    # Compute stochastic gradient
    adahess.grad = stochastic_gradient(adahess.optimizer.loss, adahess.optimizer.x,
                                      batch_size=adahess.batch_size,
                                      rng=adahess.optimizer.rng)

    # Compute Hessian diagonal
    adahess.hess_diag = compute_hessian_diagonal(adahess.optimizer.loss,
                                                 adahess.optimizer.x,
                                                 batch_size=adahess.hessian_batch_size,
                                                 rng=adahess.optimizer.rng)

    # Apply spatial averaging if enabled
    if adahess.spatial_average
        adahess.hess_diag = spatial_average_filter(adahess.hess_diag, adahess.block_length)
    end

    # Update biased first moment estimate (gradient momentum)
    adahess.m = adahess.beta1 .* adahess.m .+ (1.0 - adahess.beta1) .* adahess.grad

    # Update biased second moment estimate (Hessian diagonal momentum)
    adahess.v = adahess.beta2 .* adahess.v .+ (1.0 - adahess.beta2) .* adahess.hess_diag

    # Compute bias-corrected estimates
    m_hat = adahess.m ./ (1.0 - adahess.beta1^adahess.t)
    v_hat = adahess.v ./ (1.0 - adahess.beta2^adahess.t)

    # Update parameters using adaptive learning rates
    # θ_{t+1} = θ_t - lr * m_hat / (sqrt(v_hat) + ε)
    adahess.optimizer.x .-= adahess.lr .* m_hat ./ (sqrt.(v_hat) .+ adahess.epsilon)

    # Apply proximal operator if needed
    if adahess.optimizer.use_prox
        adahess.optimizer.x = prox(adahess.optimizer.loss.regularizer,
                                  adahess.optimizer.x, adahess.lr)
    end
end

function init_run!(adahess::AdaHessian, x0; kwargs...)
    init_run!(adahess.optimizer, x0; kwargs...)

    # Initialize momentum terms
    dim = length(adahess.optimizer.x)
    adahess.m = zeros(dim)
    adahess.v = zeros(dim)
    adahess.t = 0

    # Initialize working buffers
    adahess.grad = zeros(dim)
    adahess.hess_diag = zeros(dim)
end

function run!(adahess::AdaHessian, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
             tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(adahess.optimizer.label): The number of iterations is set to $it_max.")
    end

    adahess.optimizer.t_max = t_max
    adahess.optimizer.it_max = it_max

    # Use only the first seed for stochastic methods
    seed = adahess.optimizer.seeds[1]
    if seed in adahess.optimizer.finished_seeds
        return adahess.optimizer.trace
    end

    adahess.optimizer.rng = MersenneTwister(seed)
    adahess.optimizer.seed = seed
    loss_seed = rand(adahess.optimizer.rng, 1:MAX_SEED)
    set_seed!(adahess.optimizer.loss, loss_seed)
    init_seed!(adahess.optimizer.trace)

    if ls_it_max === nothing
        adahess.optimizer.ls_it_max = it_max
    else
        adahess.optimizer.ls_it_max = ls_it_max
    end

    if !adahess.optimizer.initialized[seed]
        init_run!(adahess, x0)
        adahess.optimizer.initialized[seed] = true
        if adahess.optimizer.line_search !== nothing
            reset!(adahess.optimizer.line_search, adahess.optimizer)
        end
    end

    if tqdm_iterations
        println("Starting optimization with max iterations: $(Int(adahess.optimizer.ls_it_max))")
    end

    while !check_convergence(adahess.optimizer)
        if adahess.optimizer.tolerance > 0
            adahess.optimizer.x_old_tol = copy(adahess.optimizer.x)
        end
        step!(adahess)
        save_checkpoint!(adahess.optimizer)

        if tqdm_iterations && adahess.optimizer.it % 100 == 0
            # Show average curvature for monitoring
            avg_curv = mean(adahess.v)
            println("Iteration: $(adahess.optimizer.it), Avg Curvature: $(round(avg_curv, digits=6))")
        end
    end

    append_seed_results!(adahess.optimizer.trace, seed)
    push!(adahess.optimizer.finished_seeds, seed)
    adahess.optimizer.seed = nothing

    return adahess.optimizer.trace
end
