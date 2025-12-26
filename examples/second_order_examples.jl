using LinearAlgebra
using Random
include("../optmethods/loss/linear_regression.jl")
include("../optmethods/loss/logistic_regression.jl")
include("../optmethods/second_order/newton.jl")
include("../optmethods/second_order/reg_newton.jl")
include("../optmethods/second_order/cubic.jl")
include("../optmethods/second_order/arc.jl")
include("../optmethods/line_search/armijo.jl")

# Define quadratic loss: f(x) = 0.5 * x^T A x + b^T x + c
struct QuadraticLoss
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Float64
    dim::Int
    regularizer::Union{Nothing, Any}
end

QuadraticLoss(A, b, c, dim) = QuadraticLoss(A, b, c, dim, nothing)

function value(ql::QuadraticLoss, x)
    return 0.5 * dot(x, ql.A * x) + dot(ql.b, x) + ql.c
end

function gradient(ql::QuadraticLoss, x)
    return ql.A * x + ql.b
end

function hessian(ql::QuadraticLoss, x)
    return ql.A
end

function hessian_lipschitz(ql::QuadraticLoss)
    return maximum(eigvals(ql.A))
end

function set_seed!(ql::QuadraticLoss, seed)
    # No randomness in this loss
end

# Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
struct RosenbrockLoss
    a::Float64
    b::Float64
    dim::Int
    regularizer::Union{Nothing, Any}
end

RosenbrockLoss(a, b, dim) = RosenbrockLoss(a, b, dim, nothing)

function value(rl::RosenbrockLoss, x)
    return (rl.a - x[1])^2 + rl.b * (x[2] - x[1]^2)^2
end

function gradient(rl::RosenbrockLoss, x)
    grad = zeros(2)
    grad[1] = -2*(rl.a - x[1]) - 4*rl.b*x[1]*(x[2] - x[1]^2)
    grad[2] = 2*rl.b*(x[2] - x[1]^2)
    return grad
end

function hessian(rl::RosenbrockLoss, x)
    H = zeros(2, 2)
    H[1,1] = 2 + 12*rl.b*x[1]^2 - 4*rl.b*(x[2] - x[1]^2)
    H[1,2] = H[2,1] = -4*rl.b*x[1]
    H[2,2] = 2*rl.b
    return H
end

function hessian_lipschitz(rl::RosenbrockLoss)
    return 100.0  # Conservative estimate for Rosenbrock
end

function set_seed!(rl::RosenbrockLoss, seed)
    # No randomness in this loss
end

# Simple quadratic for parameter testing
struct SimpleQuadratic
    A::Matrix{Float64}
    b::Vector{Float64}
    dim::Int
    regularizer::Union{Nothing, Any}
end

SimpleQuadratic(A, b, dim) = SimpleQuadratic(A, b, dim, nothing)

function value(sq::SimpleQuadratic, x)
    return 0.5 * dot(x, sq.A * x) + dot(sq.b, x)
end

function gradient(sq::SimpleQuadratic, x)
    return sq.A * x + sq.b
end

function hessian(sq::SimpleQuadratic, x)
    return sq.A
end

function hessian_lipschitz(sq::SimpleQuadratic)
    return maximum(eigvals(sq.A))
end

function set_seed!(sq::SimpleQuadratic, seed)
    # No randomness in this loss
end

function test_basic_second_order_methods()
    println("Testing Basic Second-Order Methods")
    println("=" ^ 40)

    # Generate test data for quadratic function
    Random.seed!(42)
    n, d = 20, 3
    A = randn(d, d)
    A = A' * A + 0.1 * I(d)  # Make positive definite
    b = randn(d)
    c = 1.0

    loss = QuadraticLoss(A, b, c, d)
    x0 = randn(d)
    x_opt = -A \ b  # Analytical solution

    println("\\nOptimal solution: $x_opt")
    println("Optimal value: $(value(loss, x_opt))")

    println("\\nTesting Newton Method:")
    newton = Newton(loss)
    result_newton = run!(newton, copy(x0), it_max=10)
    compute_loss_of_iterates!(result_newton)
    seed = first(keys(result_newton.xs_all))
    println("Newton iterations: $(result_newton.its_all[seed][end])")
    println("Newton final solution: $(result_newton.xs_all[seed][end])")
    println("Newton final loss: $(result_newton.loss_vals_all[seed][end])")
    println("Newton error: $(norm(result_newton.xs_all[seed][end] - x_opt))")

    println("\\nTesting Regularized Newton:")
    reg_newton = RegNewton(loss, adaptive=true)
    result_reg_newton = run!(reg_newton, copy(x0), it_max=20)
    compute_loss_of_iterates!(result_reg_newton)
    seed = first(keys(result_reg_newton.xs_all))
    println("RegNewton iterations: $(result_reg_newton.its_all[seed][end])")
    println("RegNewton final solution: $(result_reg_newton.xs_all[seed][end])")
    println("RegNewton final loss: $(result_reg_newton.loss_vals_all[seed][end])")
    println("RegNewton error: $(norm(result_reg_newton.xs_all[seed][end] - x_opt))")

    println("\\nTesting Cubic Newton:")
    cubic = Cubic(loss)
    result_cubic = run!(cubic, copy(x0), it_max=15)
    compute_loss_of_iterates!(result_cubic)
    seed = first(keys(result_cubic.xs_all))
    println("Cubic iterations: $(result_cubic.its_all[seed][end])")
    println("Cubic final solution: $(result_cubic.xs_all[seed][end])")
    println("Cubic final loss: $(result_cubic.loss_vals_all[seed][end])")
    println("Cubic error: $(norm(result_cubic.xs_all[seed][end] - x_opt))")

    println("\\nTesting ARC:")
    arc = Arc(loss)
    result_arc = run!(arc, copy(x0), it_max=15)
    compute_loss_of_iterates!(result_arc)
    seed = first(keys(result_arc.xs_all))
    println("ARC iterations: $(result_arc.its_all[seed][end])")
    println("ARC final solution: $(result_arc.xs_all[seed][end])")
    println("ARC final loss: $(result_arc.loss_vals_all[seed][end])")
    println("ARC error: $(norm(result_arc.xs_all[seed][end] - x_opt))")
end

function test_nonconvex_optimization()
    println("\\n\\nTesting on Nonconvex Problem (Rosenbrock Function)")
    println("=" ^ 50)

    rosenbrock = RosenbrockLoss(1.0, 100.0, 2)
    x0 = [-1.2, 1.0]  # Standard starting point
    x_opt = [1.0, 1.0]  # Known optimum

    println("\\nOptimal solution: $x_opt")
    println("Optimal value: $(value(rosenbrock, x_opt))")

    println("\\nTesting Cubic Newton on Rosenbrock:")
    cubic = Cubic(rosenbrock, solver_eps=1e-6)
    result_cubic = run!(cubic, copy(x0), it_max=100)
    compute_loss_of_iterates!(result_cubic)
    seed = first(keys(result_cubic.xs_all))
    println("Cubic iterations: $(result_cubic.its_all[seed][end])")
    println("Cubic final solution: $(result_cubic.xs_all[seed][end])")
    println("Cubic final loss: $(result_cubic.loss_vals_all[seed][end])")
    println("Cubic error: $(norm(result_cubic.xs_all[seed][end] - x_opt))")

    println("\\nTesting ARC on Rosenbrock:")
    arc = Arc(rosenbrock, solver_eps=1e-6)
    result_arc = run!(arc, copy(x0), it_max=100)
    compute_loss_of_iterates!(result_arc)
    seed = first(keys(result_arc.xs_all))
    println("ARC iterations: $(result_arc.its_all[seed][end])")
    println("ARC final solution: $(result_arc.xs_all[seed][end])")
    println("ARC final loss: $(result_arc.loss_vals_all[seed][end])")
    println("ARC error: $(norm(result_arc.xs_all[seed][end] - x_opt))")

    println("\\nTesting Regularized Newton on Rosenbrock:")
    reg_newton = RegNewton(rosenbrock, adaptive=true)
    result_reg_newton = run!(reg_newton, copy(x0), it_max=100)
    compute_loss_of_iterates!(result_reg_newton)
    seed = first(keys(result_reg_newton.xs_all))
    println("RegNewton iterations: $(result_reg_newton.its_all[seed][end])")
    println("RegNewton final solution: $(result_reg_newton.xs_all[seed][end])")
    println("RegNewton final loss: $(result_reg_newton.loss_vals_all[seed][end])")
    println("RegNewton error: $(norm(result_reg_newton.xs_all[seed][end] - x_opt))")
end

function test_second_order_with_line_search()
    println("\\n\\nTesting Second-Order with Line Search")
    println("=" ^ 40)

    # Generate test data for logistic regression
    Random.seed!(789)
    n, d = 50, 4
    X = randn(n, d)
    y = rand(0:1, n)

    loss = LogisticRegression(X, y)
    x0 = randn(d)

    println("\\nTesting Newton with Armijo line search:")
    armijo_ls = ArmijoLineSearch()
    newton_ls = Newton(loss, line_search=armijo_ls)
    result_newton_ls = run!(newton_ls, x0, it_max=20)
    compute_loss_of_iterates!(result_newton_ls)
    seed = first(keys(result_newton_ls.xs_all))
    println("Newton+Armijo iterations: $(result_newton_ls.its_all[seed][end])")
    println("Newton+Armijo final loss: $(result_newton_ls.loss_vals_all[seed][end])")
end

function test_custom_parameters()
    println("\\n\\nTesting Custom Parameters")
    println("=" ^ 30)

    # Simple quadratic for parameter testing
    Random.seed!(999)
    d = 3
    A = I(d) + 0.1 * randn(d, d)
    A = A' * A
    b = randn(d)

    loss = SimpleQuadratic(A, b, d)
    x0 = randn(d)

    println("\\nTesting Cubic with custom regularization:")
    cubic_custom = Cubic(loss, reg_coef=1.0, solver_it_max=50, solver_eps=1e-10)
    result_cubic_custom = run!(cubic_custom, copy(x0), it_max=15)
    seed = first(keys(result_cubic_custom.xs_all))
    println("Cubic custom iterations: $(result_cubic_custom.its_all[seed][end])")

    println("\\nTesting ARC with custom parameters:")
    arc_custom = Arc(loss, eta1=0.2, eta2=0.8, sigma=0.5, solver_it_max=50)
    result_arc_custom = run!(arc_custom, copy(x0), it_max=15)
    seed = first(keys(result_arc_custom.xs_all))
    println("ARC custom iterations: $(result_arc_custom.its_all[seed][end])")

    println("\\nTesting RegNewton with custom Hessian Lipschitz:")
    reg_newton_custom = RegNewton(loss, hess_lip=2.0, adaptive=false)
    result_reg_newton_custom = run!(reg_newton_custom, copy(x0), it_max=20)
    seed = first(keys(result_reg_newton_custom.xs_all))
    println("RegNewton custom iterations: $(result_reg_newton_custom.its_all[seed][end])")
end

# Run all tests
test_basic_second_order_methods()
test_nonconvex_optimization()
test_second_order_with_line_search()
test_custom_parameters()