include("../optimizer.jl")

"""
    StochasticLBFGS

Stochastic Limited-memory BFGS algorithm.
Maintains a limited-memory approximation of the inverse Hessian using
stochastic gradient information. Well-suited for large-scale stochastic optimization.

Key differences from deterministic L-BFGS:
- Uses stochastic gradients with mini-batches
- More conservative update criteria to handle gradient noise
- Optional variance reduction techniques

Arguments:
    lr0 (float, optional): initial learning rate (auto-computed if not provided)
    lr_max (float, optional): maximum learning rate (default: Inf)
    lr_decay_coef (float, optional): learning rate decay coefficient (default: 0.0)
    lr_decay_power (float, optional): learning rate decay power (default: 1.0)
    batch_size (int, optional): batch size for gradient computation (default: 1)
    mem_size (int, optional): memory size for L-BFGS (default: 10)
    L (float, optional): smoothness constant for initial Hessian approximation
    curvature_threshold (float, optional): threshold for accepting curvature pairs (default: 0.0)
    damping (float, optional): damping parameter for stability (default: 0.0)
"""
mutable struct StochasticLBFGS
    optimizer::Optimizer
    lr0::Union{Float64, Nothing}
    lr::Float64
    lr_max::Float64
    lr_decay_coef::Float64
    lr_decay_power::Float64
    batch_size::Int
    mem_size::Int
    L::Union{Float64, Nothing}
    curvature_threshold::Float64
    damping::Float64

    # L-BFGS specific storage
    x_difs::Vector{Vector{Float64}}
    grad_difs::Vector{Vector{Float64}}
    rhos::Vector{Float64}

    # Internal state
    grad::Vector{Float64}
    grad_old::Vector{Float64}
    x_old::Vector{Float64}
    L_local::Float64
    updates_rejected::Int

    function StochasticLBFGS(loss; lr0=nothing, lr_max=Inf, lr_decay_coef=0.0,
                            lr_decay_power=1.0, batch_size=1, mem_size=10, L=nothing,
                            curvature_threshold=0.0, damping=0.0, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if L === nothing
            L = smoothness(loss)
        end

        new(optimizer, lr0, 0.0, lr_max, lr_decay_coef, lr_decay_power,
            batch_size, mem_size, L, curvature_threshold, damping,
            Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Float64[],
            Float64[], Float64[], Float64[], 0.0, 0)
    end
end

function two_loop_recursion(slbfgs::StochasticLBFGS, q::Vector{Float64})
    """
    Two-loop recursion algorithm to compute H^{-1} * q efficiently.
    """
    alphas = Float64[]

    # First loop: compute alphas and update q
    for i in length(slbfgs.x_difs):-1:1
        s = slbfgs.x_difs[i]
        y = slbfgs.grad_difs[i]
        rho = slbfgs.rhos[i]

        alpha = rho * dot(s, q)
        push!(alphas, alpha)
        q .-= alpha .* y
    end

    # Apply initial Hessian approximation H_0^{-1}
    if length(slbfgs.x_difs) > 0
        # Use Barzilai-Borwein scaling
        s_last = slbfgs.x_difs[end]
        y_last = slbfgs.grad_difs[end]
        gamma = dot(s_last, y_last) / dot(y_last, y_last)
        r = gamma .* q
    else
        # Initial scaling using smoothness constant
        r = q ./ slbfgs.L
    end

    # Second loop: update r
    reverse!(alphas)
    for i in 1:length(slbfgs.x_difs)
        s = slbfgs.x_difs[i]
        y = slbfgs.grad_difs[i]
        rho = slbfgs.rhos[i]
        alpha = alphas[i]

        beta = rho * dot(y, r)
        r .+= s .* (alpha - beta)
    end

    return r
end

function step!(slbfgs::StochasticLBFGS)
    # Update learning rate with decay
    it = slbfgs.optimizer.it
    slbfgs.lr = slbfgs.lr0 / (1.0 + slbfgs.lr_decay_coef * it^slbfgs.lr_decay_power)
    slbfgs.lr = min(slbfgs.lr, slbfgs.lr_max)

    # Store old values for curvature pair
    if it > 0
        slbfgs.x_old = copy(slbfgs.optimizer.x)
        slbfgs.grad_old = copy(slbfgs.grad)
    end

    # Compute stochastic gradient
    slbfgs.grad = stochastic_gradient(slbfgs.optimizer.loss, slbfgs.optimizer.x,
                                     batch_size=slbfgs.batch_size,
                                     rng=slbfgs.optimizer.rng)

    # Compute search direction using two-loop recursion
    q = copy(slbfgs.grad)
    direction = two_loop_recursion(slbfgs, q)

    # Update parameters
    slbfgs.optimizer.x .-= slbfgs.lr .* direction

    # Apply proximal operator if needed
    if slbfgs.optimizer.use_prox
        slbfgs.optimizer.x = prox(slbfgs.optimizer.loss.regularizer,
                                 slbfgs.optimizer.x, slbfgs.lr)
    end

    # Update curvature pairs for next iteration
    if it > 0
        s_new = slbfgs.optimizer.x .- slbfgs.x_old
        y_new = slbfgs.grad .- slbfgs.grad_old

        # Apply damping for stability (Powell's damping)
        if slbfgs.damping > 0.0
            sy = dot(s_new, y_new)
            ss = dot(s_new, s_new)
            if sy < slbfgs.damping * ss
                theta = (1.0 - slbfgs.damping) * ss / (ss - sy)
                y_new = theta .* y_new .+ (1.0 - theta) .* s_new
            end
        end

        # Check curvature condition before adding
        sy = dot(s_new, y_new)
        if sy > slbfgs.curvature_threshold
            push!(slbfgs.x_difs, copy(s_new))
            push!(slbfgs.grad_difs, copy(y_new))
            push!(slbfgs.rhos, 1.0 / sy)

            # Maintain memory size
            if length(slbfgs.x_difs) > slbfgs.mem_size
                popfirst!(slbfgs.x_difs)
                popfirst!(slbfgs.grad_difs)
                popfirst!(slbfgs.rhos)
            end
        else
            slbfgs.updates_rejected += 1
        end
    end
end

function init_run!(slbfgs::StochasticLBFGS, x0; kwargs...)
    init_run!(slbfgs.optimizer, x0; kwargs...)

    # Initialize learning rate if not provided
    if slbfgs.lr0 === nothing
        slbfgs.lr0 = 1.0 / batch_smoothness(slbfgs.optimizer.loss, slbfgs.batch_size)
    end
    slbfgs.lr = slbfgs.lr0

    # Clear history
    empty!(slbfgs.x_difs)
    empty!(slbfgs.grad_difs)
    empty!(slbfgs.rhos)

    # Initialize internal state
    dim = length(slbfgs.optimizer.x)
    slbfgs.grad = zeros(dim)
    slbfgs.grad_old = zeros(dim)
    slbfgs.x_old = zeros(dim)
    slbfgs.updates_rejected = 0
end

function run!(slbfgs::StochasticLBFGS, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
             tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(slbfgs.optimizer.label): The number of iterations is set to $it_max.")
    end

    slbfgs.optimizer.t_max = t_max
    slbfgs.optimizer.it_max = it_max

    # Use only the first seed for stochastic methods
    seed = slbfgs.optimizer.seeds[1]
    if seed in slbfgs.optimizer.finished_seeds
        return slbfgs.optimizer.trace
    end

    slbfgs.optimizer.rng = MersenneTwister(seed)
    slbfgs.optimizer.seed = seed
    loss_seed = rand(slbfgs.optimizer.rng, 1:MAX_SEED)
    set_seed!(slbfgs.optimizer.loss, loss_seed)
    init_seed!(slbfgs.optimizer.trace)

    if ls_it_max === nothing
        slbfgs.optimizer.ls_it_max = it_max
    else
        slbfgs.optimizer.ls_it_max = ls_it_max
    end

    if !slbfgs.optimizer.initialized[seed]
        init_run!(slbfgs, x0)
        slbfgs.optimizer.initialized[seed] = true
        if slbfgs.optimizer.line_search !== nothing
            reset!(slbfgs.optimizer.line_search, slbfgs.optimizer)
        end
    end

    if tqdm_iterations
        println("Starting optimization with max iterations: $(Int(slbfgs.optimizer.ls_it_max))")
    end

    while !check_convergence(slbfgs.optimizer)
        if slbfgs.optimizer.tolerance > 0
            slbfgs.optimizer.x_old_tol = copy(slbfgs.optimizer.x)
        end
        step!(slbfgs)
        save_checkpoint!(slbfgs.optimizer)

        if tqdm_iterations && slbfgs.optimizer.it % 100 == 0
            mem_usage = length(slbfgs.x_difs)
            println("Iteration: $(slbfgs.optimizer.it), Memory: $mem_usage/$(slbfgs.mem_size), Rejected: $(slbfgs.updates_rejected)")
        end
    end

    append_seed_results!(slbfgs.optimizer.trace, seed)
    push!(slbfgs.optimizer.finished_seeds, seed)
    slbfgs.optimizer.seed = nothing

    println("Total updates rejected: $(slbfgs.updates_rejected)")
    return slbfgs.optimizer.trace
end
