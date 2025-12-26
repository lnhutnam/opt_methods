include("../optimizer.jl")

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
        newton.optimizer.x = newton.optimizer.line_search(newton.optimizer.x, -inv_hess_grad_prod)
    end
end

function init_run!(newton::Newton, x0; kwargs...)
    init_run!(newton.optimizer, x0; kwargs...)

    dim = length(newton.optimizer.x)
    newton.grad = zeros(dim)
    newton.hess = zeros(dim, dim)
end

function run!(newton::Newton, x0; kwargs...)
    # Override the step! method for the underlying optimizer
    original_step! = newton.optimizer.step!
    newton.optimizer.step! = () -> step!(newton)

    result = run!(newton.optimizer, x0; kwargs...)

    # Restore original step! method
    newton.optimizer.step! = original_step!
    return result
end