include("../optimizer.jl")
using LinearAlgebra
using Random

function empirical_hess_lip(grad, grad_old, hess, x, x_old, loss)
    grad_error = grad .- grad_old .- hess * (x .- x_old)
    r2 = norm(x .- x_old)^2
    if r2 > 0
        return 2 * norm(grad_error) / r2
    end
    return eps(Float64)
end

"""
Regularized Newton algorithm for second-order minimization.
By default returns the Regularized Newton method from paper
    "Regularized Newton Method with Global O(1/k^2) Convergence"
    https://arxiv.org/abs/2112.02089

Arguments:
    loss (optmethods.loss.Oracle): loss oracle
    identity_coef (float, optional): initial regularization coefficient (default: nothing)
    hess_lip (float, optional): estimate for the Hessian Lipschitz constant.
        If not provided, it is estimated or a small value is used (default: nothing)
    adaptive (bool, optional): use decreasing regularization based on either empirical Hessian-Lipschitz constant
        or a line-search procedure
    line_search (optmethods.LineSearch, optional): a callable line search. If nothing, line search is intialized
        automatically as an instance of RegularizedNewtonLineSearch (default: nothing)
    use_line_search (bool, optional): use line search to estimate the Lipschitz constan of the Hessian.
        If adaptive is true, line search will be non-monotonic and regularization may decrease (default: false)
    backtracking (float, optional): backtracking constant for the line search if line_search is nothing and
        use_line_search is true (default: 0.5)
"""
mutable struct RegNewton
    optimizer::Optimizer
    identity_coef::Union{Float64, Nothing}
    hess_lip::Float64
    H::Float64
    adaptive::Bool
    use_line_search::Bool

    # Internal state
    grad::Vector{Float64}
    hess::Matrix{Float64}
    x_old::Union{Vector{Float64}, Nothing}
    grad_old::Union{Vector{Float64}, Nothing}

    function RegNewton(loss; identity_coef=nothing, hess_lip=nothing, adaptive=false,
                       line_search=nothing, use_line_search=false, backtracking=0.5, kwargs...)

        # Get or estimate hess_lip
        if hess_lip === nothing
            hess_lip = hessian_lipschitz(loss)
            if hess_lip === nothing
                hess_lip = 1e-5
                @warn "No estimate of Hessian-Lipschitzness is given, so a small value $hess_lip is used as a heuristic."
            end
        end

        H = hess_lip / 2

        # Line search will need to be passed in externally to avoid circular dependencies

        optimizer = Optimizer(loss; line_search=line_search, kwargs...)

        new(optimizer, identity_coef, hess_lip, H, adaptive, use_line_search,
            Float64[], Matrix{Float64}(undef, 0, 0), nothing, nothing)
    end
end

function step!(reg_newton::RegNewton)
    reg_newton.grad = gradient(reg_newton.optimizer.loss, reg_newton.optimizer.x)

    if reg_newton.adaptive && reg_newton.x_old !== nothing && !reg_newton.use_line_search
        reg_newton.hess_lip /= 2
        empirical_lip = empirical_hess_lip(reg_newton.grad, reg_newton.grad_old,
                                         reg_newton.hess, reg_newton.optimizer.x,
                                         reg_newton.x_old, reg_newton.optimizer.loss)
        reg_newton.hess_lip = max(reg_newton.hess_lip, empirical_lip)
    end

    reg_newton.hess = hessian(reg_newton.optimizer.loss, reg_newton.optimizer.x)

    if reg_newton.use_line_search
        reg_newton.optimizer.x = reg_newton.optimizer.line_search(reg_newton.optimizer.x,
                                                                reg_newton.grad, reg_newton.hess)
    else
        if reg_newton.adaptive
            reg_newton.H = reg_newton.hess_lip / 2
        end
        grad_norm = norm(reg_newton.grad)
        reg_newton.identity_coef = sqrt(reg_newton.H * grad_norm)

        reg_newton.x_old = copy(reg_newton.optimizer.x)
        reg_newton.grad_old = copy(reg_newton.grad)

        # Solve (Hess + identity_coef * I) * delta_x = -grad
        regularized_hess = reg_newton.hess + reg_newton.identity_coef * I(length(reg_newton.optimizer.x))
        delta_x = regularized_hess \ (-reg_newton.grad)
        reg_newton.optimizer.x .+= delta_x
    end
end

function init_run!(reg_newton::RegNewton, x0; kwargs...)
    init_run!(reg_newton.optimizer, x0; kwargs...)

    dim = length(reg_newton.optimizer.x)
    reg_newton.grad = zeros(dim)
    reg_newton.hess = zeros(dim, dim)
    reg_newton.x_old = nothing
    reg_newton.grad_old = nothing

    # Initialize trace for learning rates if needed
    if hasfield(typeof(reg_newton.optimizer.trace), :lrs)
        reg_newton.optimizer.trace.lrs = Float64[]
    end
end

function run!(reg_newton::RegNewton, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("RegNewton: The number of iterations is set to $it_max.")
    end

    reg_newton.optimizer.t_max = t_max
    reg_newton.optimizer.it_max = it_max

    # Use first seed for single-seed run
    seed = reg_newton.optimizer.seeds[1]
    if seed in reg_newton.optimizer.finished_seeds
        return reg_newton.optimizer.trace
    end

    reg_newton.optimizer.rng = MersenneTwister(seed)
    reg_newton.optimizer.seed = seed
    loss_seed = rand(reg_newton.optimizer.rng, 1:100000)
    set_seed!(reg_newton.optimizer.loss, loss_seed)
    init_seed!(reg_newton.optimizer.trace)

    if ls_it_max === nothing
        reg_newton.optimizer.ls_it_max = it_max
    else
        reg_newton.optimizer.ls_it_max = ls_it_max
    end

    if !reg_newton.optimizer.initialized[seed]
        init_run!(reg_newton, x0)
        reg_newton.optimizer.initialized[seed] = true
        if reg_newton.optimizer.line_search !== nothing
            reset!(reg_newton.optimizer.line_search, reg_newton.optimizer)
        end
    end

    while !check_convergence(reg_newton.optimizer)
        if reg_newton.optimizer.tolerance > 0
            reg_newton.optimizer.x_old_tol = copy(reg_newton.optimizer.x)
        end
        step!(reg_newton)
        save_checkpoint!(reg_newton.optimizer)
        update_trace!(reg_newton)
    end

    append_seed_results!(reg_newton.optimizer.trace, seed)
    push!(reg_newton.optimizer.finished_seeds, seed)
    reg_newton.optimizer.seed = nothing

    return reg_newton.optimizer.trace
end

function update_trace!(reg_newton::RegNewton)
    # Call parent update_trace if available
    if hasmethod(update_trace!, (typeof(reg_newton.optimizer),))
        update_trace!(reg_newton.optimizer)
    end

    # Add learning rate to trace if not using line search
    if !reg_newton.use_line_search && hasfield(typeof(reg_newton.optimizer.trace), :lrs)
        push!(reg_newton.optimizer.trace.lrs, 1 / reg_newton.identity_coef)
    end
end