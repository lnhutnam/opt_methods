include("../optimizer.jl")
using LinearAlgebra
using Random

function ls_cubic_solver(x, g, H, M; it_max=100, epsilon=1e-8, loss=nothing)
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

    # Compute Newton step
    newton_step = -H \ g
    if M == 0
        return x + newton_step, solver_it
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
            return x_new, solver_it
        end
    end

    r_max = norm(newton_step)
    if r_max - r_min < epsilon
        return x + newton_step, solver_it
    end

    id_matrix = I(length(g))
    for _ in 1:it_max
        r_try = (r_min + r_max) / 2
        lam = r_try * M
        s_lam = -(H + lam * id_matrix) \ g
        solver_it += 1
        crit = conv_criterion(s_lam, r_try)
        if abs(crit) < epsilon
            return x + s_lam, solver_it
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
    return x + s_lam, solver_it
end

"""
Newton method with cubic regularization for global convergence.
The method was studied by Nesterov and Polyak in the following paper:
    "Cubic regularization of Newton method and its global performance"
    https://link.springer.com/article/10.1007/s10107-006-0706-8

Arguments:
    reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
    solver_it_max (int, optional): maximum iterations for cubic subproblem solver (default: 100)
    solver_eps (float, optional): tolerance for cubic subproblem solver (default: 1e-8)
    cubic_solver (function, optional): custom cubic subproblem solver (default: ls_cubic_solver)
"""
mutable struct Cubic
    optimizer::Optimizer
    reg_coef::Union{Float64, Nothing}
    cubic_solver::Function
    solver_it::Int
    solver_it_max::Int
    solver_eps::Float64

    # Internal state
    grad::Vector{Float64}
    hess::Matrix{Float64}

    function Cubic(loss; reg_coef=nothing, solver_it_max=100, solver_eps=1e-8,
                   cubic_solver=ls_cubic_solver, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if reg_coef === nothing
            reg_coef = hessian_lipschitz(loss)
        end

        new(optimizer, reg_coef, cubic_solver, 0, solver_it_max, solver_eps,
            Float64[], Matrix{Float64}(undef, 0, 0))
    end
end

function step!(cubic::Cubic)
    cubic.grad = gradient(cubic.optimizer.loss, cubic.optimizer.x)
    cubic.hess = hessian(cubic.optimizer.loss, cubic.optimizer.x)

    x_new, solver_it = cubic.cubic_solver(cubic.optimizer.x, cubic.grad, cubic.hess,
                                        cubic.reg_coef/2; it_max=cubic.solver_it_max,
                                        epsilon=cubic.solver_eps, loss=cubic.optimizer.loss)
    cubic.optimizer.x = x_new
    cubic.solver_it += solver_it
end

function init_run!(cubic::Cubic, x0; kwargs...)
    init_run!(cubic.optimizer, x0; kwargs...)

    dim = length(cubic.optimizer.x)
    cubic.grad = zeros(dim)
    cubic.hess = zeros(dim, dim)

    # Initialize trace for solver iterations if needed
    if hasfield(typeof(cubic.optimizer.trace), :solver_its)
        cubic.optimizer.trace.solver_its = [0]
    end
end

function run!(cubic::Cubic, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=false, tqdm_iterations=false)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("Cubic: The number of iterations is set to $it_max.")
    end

    cubic.optimizer.t_max = t_max
    cubic.optimizer.it_max = it_max

    # Use first seed for single-seed run
    seed = cubic.optimizer.seeds[1]
    if seed in cubic.optimizer.finished_seeds
        return cubic.optimizer.trace
    end

    cubic.optimizer.rng = MersenneTwister(seed)
    cubic.optimizer.seed = seed
    loss_seed = rand(cubic.optimizer.rng, 1:100000)
    set_seed!(cubic.optimizer.loss, loss_seed)
    init_seed!(cubic.optimizer.trace)

    if ls_it_max === nothing
        cubic.optimizer.ls_it_max = it_max
    else
        cubic.optimizer.ls_it_max = ls_it_max
    end

    if !cubic.optimizer.initialized[seed]
        init_run!(cubic, x0)
        cubic.optimizer.initialized[seed] = true
        if cubic.optimizer.line_search !== nothing
            reset!(cubic.optimizer.line_search, cubic.optimizer)
        end
    end

    while !check_convergence(cubic.optimizer)
        if cubic.optimizer.tolerance > 0
            cubic.optimizer.x_old_tol = copy(cubic.optimizer.x)
        end
        step!(cubic)
        save_checkpoint!(cubic.optimizer)
        update_trace!(cubic)
    end

    append_seed_results!(cubic.optimizer.trace, seed)
    push!(cubic.optimizer.finished_seeds, seed)
    cubic.optimizer.seed = nothing

    return cubic.optimizer.trace
end

function update_trace!(cubic::Cubic)
    # Call parent update_trace if available
    if hasmethod(update_trace!, (typeof(cubic.optimizer),))
        update_trace!(cubic.optimizer)
    end

    # Add solver iterations to trace if available
    if hasfield(typeof(cubic.optimizer.trace), :solver_its)
        push!(cubic.optimizer.trace.solver_its, cubic.solver_it)
    end
end