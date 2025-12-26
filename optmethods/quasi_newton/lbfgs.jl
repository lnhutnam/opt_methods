include("../optimizer.jl")

"""
Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm. See
    p. 177 in (J. Nocedal and S. J. Wright, "Numerical Optimization", 2nd edition)
or
    https://en.wikipedia.org/wiki/Limited-memory_BFGS
for a general description.

Arguments:
    L (float, optional): an upper bound on the smoothness constant
        to initialize the Hessian estimate
    hess_estim (Matrix, optional): initial Hessian estimate
    inv_hess_estim (Matrix, optional): initial inverse Hessian estimate
    lr (float, optional): stepsize (default: 1)
    mem_size (int, optional): memory size (default: 10)
    adaptive_init (bool, optional): whether to use adaptive initialization (default: false)
"""
mutable struct LBFGS
    optimizer::Optimizer
    L::Union{Float64, Nothing}
    lr::Float64
    mem_size::Int
    adaptive_init::Bool
    B::Union{Matrix{Float64}, Nothing}
    B_inv::Union{Matrix{Float64}, Nothing}

    # L-BFGS specific storage
    x_difs::Vector{Vector{Float64}}
    grad_difs::Vector{Vector{Float64}}
    rhos::Vector{Float64}

    # Internal state
    grad::Vector{Float64}
    L_local::Float64

    function LBFGS(loss; L=nothing, hess_estim=nothing, inv_hess_estim=nothing,
                   lr=1.0, mem_size=10, adaptive_init=false, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if L === nothing && hess_estim === nothing && inv_hess_estim === nothing
            L_val = smoothness(loss)
            if L_val === nothing
                error("Either smoothness constant L or Hessian/inverse-Hessian estimate must be provided")
            end
            L = L_val
        end

        B = hess_estim === nothing ? nothing : Matrix{Float64}(hess_estim)
        B_inv = inv_hess_estim === nothing ? nothing : Matrix{Float64}(inv_hess_estim)

        if inv_hess_estim === nothing && hess_estim !== nothing
            B_inv = pinv(B)
        end

        new(optimizer, L, lr, mem_size, adaptive_init, B, B_inv,
            Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Float64[],
            Float64[], 0.0)
    end
end

function step!(lbfgs::LBFGS)
    lbfgs.grad = gradient(lbfgs.optimizer.loss, lbfgs.optimizer.x)
    q = copy(lbfgs.grad)
    alphas = Float64[]

    # First loop: compute alphas and update q
    for i in length(lbfgs.x_difs):-1:1
        s = lbfgs.x_difs[i]
        y = lbfgs.grad_difs[i]
        rho = lbfgs.rhos[i]

        alpha = rho * inner_prod(s, q)
        push!(alphas, alpha)
        q .-= alpha .* y
    end

    # Apply initial Hessian approximation
    if lbfgs.B_inv !== nothing
        r = lbfgs.B_inv * q
    else
        if lbfgs.adaptive_init && length(lbfgs.x_difs) > 0
            y = lbfgs.grad_difs[end]
            y_norm = norm(y)
            lbfgs.L_local = y_norm^2 * lbfgs.rhos[end]
            r = q ./ lbfgs.L_local
        else
            r = q ./ lbfgs.L
        end
    end

    # Second loop: update r
    reverse!(alphas)  # Reverse to get correct order
    for i in 1:length(lbfgs.x_difs)
        s = lbfgs.x_difs[i]
        y = lbfgs.grad_difs[i]
        rho = lbfgs.rhos[i]
        alpha = alphas[i]

        beta = rho * inner_prod(y, r)
        r .+= s .* (alpha - beta)
    end

    x_new = lbfgs.optimizer.x .- lbfgs.lr .* r

    if lbfgs.optimizer.line_search !== nothing
        x_new = lbfgs.optimizer.line_search(x=lbfgs.optimizer.x, x_new=x_new, gradient=lbfgs.grad)
    end

    grad_new = gradient(lbfgs.optimizer.loss, x_new)

    # Update history
    s_new = x_new .- lbfgs.optimizer.x
    y_new = grad_new .- lbfgs.grad
    rho_inv = inner_prod(s_new, y_new)

    if rho_inv > 0
        push!(lbfgs.x_difs, copy(s_new))
        push!(lbfgs.grad_difs, copy(y_new))
        push!(lbfgs.rhos, 1.0 / rho_inv)

        # Maintain memory size
        if length(lbfgs.x_difs) > lbfgs.mem_size
            popfirst!(lbfgs.x_difs)
            popfirst!(lbfgs.grad_difs)
            popfirst!(lbfgs.rhos)
        end
    end

    lbfgs.optimizer.x = x_new
    lbfgs.grad = grad_new
end

function init_run!(lbfgs::LBFGS, x0; kwargs...)
    init_run!(lbfgs.optimizer, x0; kwargs...)

    # Clear history
    empty!(lbfgs.x_difs)
    empty!(lbfgs.grad_difs)
    empty!(lbfgs.rhos)

    lbfgs.grad = gradient(lbfgs.optimizer.loss, lbfgs.optimizer.x)
end

function run!(lbfgs::LBFGS, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(lbfgs.optimizer.label): The number of iterations is set to $it_max.")
    end

    lbfgs.optimizer.t_max = t_max
    lbfgs.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(lbfgs.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(lbfgs.optimizer.seeds)) seeds...")
    end

    for seed in lbfgs.optimizer.seeds
        if seed in lbfgs.optimizer.finished_seeds
            continue
        end

        lbfgs.optimizer.rng = MersenneTwister(seed)
        lbfgs.optimizer.seed = seed
        loss_seed = rand(lbfgs.optimizer.rng, 1:MAX_SEED)
        set_seed!(lbfgs.optimizer.loss, loss_seed)
        init_seed!(lbfgs.optimizer.trace)

        if ls_it_max === nothing
            lbfgs.optimizer.ls_it_max = it_max
        else
            lbfgs.optimizer.ls_it_max = ls_it_max
        end

        if !lbfgs.optimizer.initialized[seed]
            init_run!(lbfgs, x0)
            lbfgs.optimizer.initialized[seed] = true
            if lbfgs.optimizer.line_search !== nothing
                reset!(lbfgs.optimizer.line_search, lbfgs.optimizer)
            end
        end

        it_criterion = lbfgs.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(lbfgs.optimizer.ls_it_max))")
        end

        while !check_convergence(lbfgs.optimizer)
            if lbfgs.optimizer.tolerance > 0
                lbfgs.optimizer.x_old_tol = copy(lbfgs.optimizer.x)
            end
            step!(lbfgs)
            save_checkpoint!(lbfgs.optimizer)

            if tqdm_iterations && lbfgs.optimizer.it % 100 == 0
                println("Iteration: $(lbfgs.optimizer.it)")
            end
        end

        append_seed_results!(lbfgs.optimizer.trace, seed)
        push!(lbfgs.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(lbfgs.optimizer.finished_seeds))/$(length(lbfgs.optimizer.seeds))")
        end
    end

    lbfgs.optimizer.seed = nothing
    return lbfgs.optimizer.trace
end