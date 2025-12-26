include("../optimizer.jl")
using LinearAlgebra
using Random

mutable struct MockLineSearch
    lr::Union{Float64, Nothing}
    it::Int

    function MockLineSearch()
        new(nothing, 0)
    end
end

function reset!(mock_ls::MockLineSearch, optimizer)
    mock_ls.it = 0
end

function arc_cubic_solver(x, g, H, M; it_max=100, epsilon=1e-8, loss=nothing)
    """
    Solve min_z <g, z-x> + 1/2<z-x, H(z-x)> + M/3 ||z-x||^3

    For explanation of Cauchy point, see "Gradient Descent
        Efficiently Finds the Cubic-Regularized Non-Convex Newton Step"
        https://arxiv.org/pdf/1612.00547.pdf
    Other potential implementations can be found in paper
        "Adaptive cubic regularisation methods"
        https://people.maths.ox.ac.uk/cartis/papers/ARCpI.pdf
    """
    solver_it = 1
    lam = 0.0

    # Compute Newton step
    newton_step = -H \ g
    if M == 0
        return x + newton_step, solver_it, lam
    end

    function cauchy_point(g, H, M)
        if norm(g) == 0 || M == 0
            return zeros(size(g))
        end
        g_dir = g / norm(g)
        H_g_g = dot(g_dir, H * g_dir)
        R = -H_g_g / (2*M) + sqrt((H_g_g/M)^2/4 + norm(g)/M)
        return -R * g_dir
    end

    function conv_criterion(s, r)
        """
        The convergence criterion is an increasing and concave function in r
        and it is equal to 0 only if r is the solution to the cubic problem
        """
        s_norm = norm(s)
        return 1/s_norm - 1/r
    end

    # Solution s satisfies ||s|| >= Cauchy_radius
    r_min = norm(cauchy_point(g, H, M))

    if loss !== nothing
        x_new = x + newton_step
        if value(loss, x) > value(loss, x_new)
            return x_new, solver_it, lam
        end
    end

    r_max = norm(newton_step)
    if r_max - r_min < epsilon
        return x + newton_step, solver_it, lam
    end

    id_matrix = I(length(g))
    s_lam = zeros(length(g))  # Initialize s_lam
    for _ in 1:it_max
        # Run bisection on the regularization using conv_criterion
        r_try = (r_min + r_max) / 2
        lam = r_try * M
        s_lam = -(H + lam * id_matrix) \ g
        solver_it += 1
        crit = conv_criterion(s_lam, r_try)
        if abs(crit) < epsilon
            return x + s_lam, solver_it, lam
        end
        if crit < 0
            r_min = r_try
        else
            r_max = r_try
        end
        if r_max - r_min < epsilon
            break
        end
    end
    return x + s_lam, solver_it, lam
end

"""
Adaptive Regularisation algorithm using Cubics (ARC) is a second-order optimizer based on Cubic Newton.
This implementation is based on the paper by Cartis et al.,
    "Adaptive cubic regularisation methods for unconstrained optimization.
        Part I: motivation, convergence and numerical results"
We use the same rules for initializing eta1, eta2, sigma and updating sigma as given in the paper.

Arguments:
    eta1 (float, optional): parameter to identify very successful iterations (default: 0.1)
    eta2 (float, optional): parameter to identify unsuccessful iterations (default: 0.9)
    sigma_eps (float, optional): minimal value of the cubic-penalty coefficient (default: 1e-16)
    sigma (float, optional): an estimate of the Hessian's Lipschitz constant
    solver_it_max (int, optional): subsolver hard limit on iteration number (default: 100)
    solver_eps (float, optional): subsolver precision parameter (default: 1e-4)
    cubic_solver (callable, optional): subsolver (default: arc_cubic_solver)
"""
mutable struct Arc
    optimizer::Optimizer
    eta1::Float64
    eta2::Float64
    sigma_eps::Float64
    sigma::Float64
    cubic_solver::Function
    solver_it::Int
    solver_it_max::Int
    solver_eps::Float64
    mock_line_search::MockLineSearch

    # Internal state
    grad::Vector{Float64}
    hess::Matrix{Float64}
    f_prev::Union{Float64, Nothing}

    function Arc(loss; eta1=0.1, eta2=0.9, sigma_eps=1e-16, sigma=nothing,
                solver_it_max=100, solver_eps=1e-4, cubic_solver=arc_cubic_solver, kwargs...)

        if sigma === nothing
            sigma = hessian_lipschitz(loss)
            if sigma === nothing
                sigma = 1.0
            else
                sigma = sigma / 2
            end
        end

        optimizer = Optimizer(loss; kwargs...)
        mock_ls = MockLineSearch()

        new(optimizer, eta1, eta2, sigma_eps, sigma, cubic_solver, 0, solver_it_max,
            solver_eps, mock_ls, Float64[], Matrix{Float64}(undef, 0, 0), nothing)
    end
end

function step!(arc::Arc)
    if arc.f_prev === nothing
        arc.f_prev = value(arc.optimizer.loss, arc.optimizer.x)
    end

    arc.grad = gradient(arc.optimizer.loss, arc.optimizer.x)
    grad_norm = norm(arc.grad)
    arc.hess = hessian(arc.optimizer.loss, arc.optimizer.x)

    solver_eps = min(arc.solver_eps, sqrt(grad_norm)) * grad_norm
    x_cubic, solver_it, lam = arc.cubic_solver(arc.optimizer.x, arc.grad, arc.hess,
                                             arc.sigma; it_max=arc.solver_it_max,
                                             epsilon=solver_eps, loss=arc.optimizer.loss)

    s = x_cubic - arc.optimizer.x
    model_value = arc.f_prev + dot(s, arc.grad) + 0.5 * dot(s, arc.hess * s) + arc.sigma/3 * norm(s)^3
    f_new = value(arc.optimizer.loss, x_cubic)
    rho = (arc.f_prev - f_new) / (arc.f_prev - model_value)

    if rho > arc.eta1
        arc.optimizer.x = x_cubic
        arc.f_prev = f_new
    else
        arc.sigma *= 2
    end

    if rho > arc.eta2
        arc.sigma = max(arc.sigma_eps, min(arc.sigma / 2, grad_norm))
    end

    arc.mock_line_search.it += solver_it
    arc.mock_line_search.lr = lam > 0 ? 1 / lam : Inf
end

function init_run!(arc::Arc, x0; kwargs...)
    init_run!(arc.optimizer, x0; kwargs...)

    dim = length(arc.optimizer.x)
    arc.grad = zeros(dim)
    arc.hess = zeros(dim, dim)
    arc.f_prev = nothing

    # Initialize trace for sigmas if needed
    if hasfield(typeof(arc.optimizer.trace), :sigmas)
        arc.optimizer.trace.sigmas = [arc.sigma]
    end
end

function run!(arc::Arc, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("Arc: The number of iterations is set to $it_max.")
    end

    arc.optimizer.t_max = t_max
    arc.optimizer.it_max = it_max

    # Use first seed for single-seed run
    seed = arc.optimizer.seeds[1]
    if seed in arc.optimizer.finished_seeds
        return arc.optimizer.trace
    end

    arc.optimizer.rng = MersenneTwister(seed)
    arc.optimizer.seed = seed
    loss_seed = rand(arc.optimizer.rng, 1:100000)
    set_seed!(arc.optimizer.loss, loss_seed)
    init_seed!(arc.optimizer.trace)

    if ls_it_max === nothing
        arc.optimizer.ls_it_max = it_max
    else
        arc.optimizer.ls_it_max = ls_it_max
    end

    if !arc.optimizer.initialized[seed]
        init_run!(arc, x0)
        arc.optimizer.initialized[seed] = true
        if arc.optimizer.line_search !== nothing
            reset!(arc.optimizer.line_search, arc.optimizer)
        end
    end

    while !check_convergence(arc.optimizer)
        if arc.optimizer.tolerance > 0
            arc.optimizer.x_old_tol = copy(arc.optimizer.x)
        end
        step!(arc)
        save_checkpoint!(arc.optimizer)
        update_trace!(arc)
    end

    append_seed_results!(arc.optimizer.trace, seed)
    push!(arc.optimizer.finished_seeds, seed)
    arc.optimizer.seed = nothing

    return arc.optimizer.trace
end

function update_trace!(arc::Arc)
    # Call parent update_trace if available
    if hasmethod(update_trace!, (typeof(arc.optimizer),))
        update_trace!(arc.optimizer)
    end

    # Add sigma to trace if available
    if hasfield(typeof(arc.optimizer.trace), :sigmas)
        push!(arc.optimizer.trace.sigmas, arc.sigma)
    end
end