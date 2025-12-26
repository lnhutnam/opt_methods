using LinearAlgebra
using Random
include("../optmethods/loss/linear_regression.jl")
include("../optmethods/loss/logistic_regression.jl")
include("../optmethods/quasi_newton/bfgs.jl")
include("../optmethods/quasi_newton/lbfgs.jl")
include("../optmethods/quasi_newton/dfp.jl")
include("../optmethods/quasi_newton/sr1.jl")
include("../optmethods/line_search/armijo.jl")

# Define QuadraticLoss struct and methods at top level
struct QuadraticLoss <: Oracle
    A::Matrix{Float64}
    b::Vector{Float64}
    dim::Int
    regularizer

    function QuadraticLoss(A, b)
        new(A, b, length(b), nothing)
    end
end

function value(ql::QuadraticLoss, x::Vector{Float64})
    return 0.5 * dot(x, ql.A * x) + dot(ql.b, x)
end

function gradient(ql::QuadraticLoss, x::Vector{Float64})
    return ql.A * x + ql.b
end

function hessian(ql::QuadraticLoss, x::Vector{Float64})
    return ql.A
end

function smoothness(ql::QuadraticLoss)
    return maximum(eigvals(ql.A))
end

function set_seed!(ql::QuadraticLoss, seed::Int64)
    # No randomness in quadratic loss
end

function test_quasi_newton_methods()
    println("Testing Quasi-Newton Methods")
    println("=" ^ 40)

    # Generate test data
    Random.seed!(42)
    n, d = 100, 5
    X = randn(n, d)
    y = X * randn(d) + 0.1 * randn(n)

    # Create linear regression loss
    loss = LinearRegression(X, y)
    x0 = randn(d)

    println("\nTesting BFGS:")
    bfgs = BFGS(loss)
    trace_bfgs = run!(bfgs, x0, it_max=50)
    compute_loss_of_iterates!(trace_bfgs)
    final_loss_bfgs = best_loss_value(trace_bfgs)
    println("BFGS final loss: $final_loss_bfgs")

    println("\nTesting L-BFGS:")
    lbfgs = LBFGS(loss, mem_size=5)
    trace_lbfgs = run!(lbfgs, copy(x0), it_max=50)
    compute_loss_of_iterates!(trace_lbfgs)
    final_loss_lbfgs = best_loss_value(trace_lbfgs)
    println("L-BFGS final loss: $final_loss_lbfgs")

    println("\nTesting DFP:")
    dfp = DFP(loss)
    trace_dfp = run!(dfp, copy(x0), it_max=50)
    compute_loss_of_iterates!(trace_dfp)
    final_loss_dfp = best_loss_value(trace_dfp)
    println("DFP final loss: $final_loss_dfp")

    println("\nTesting SR1:")
    sr1 = SR1(loss)
    trace_sr1 = run!(sr1, copy(x0), it_max=50)
    compute_loss_of_iterates!(trace_sr1)
    final_loss_sr1 = best_loss_value(trace_sr1)
    println("SR1 final loss: $final_loss_sr1")

    println("\n" * "=" ^ 40)
    println("All quasi-Newton methods tested successfully!")
end

function test_quasi_newton_with_line_search()
    println("\n\nTesting Quasi-Newton with Line Search")
    println("=" ^ 40)

    # Generate test data
    Random.seed!(123)
    n, d = 50, 3
    X = randn(n, d)
    y = rand(0:1, n)

    # Create logistic regression loss
    loss = LogisticRegression(X, y)
    x0 = randn(d)

    # Test with Armijo line search
    println("\nTesting BFGS with Armijo line search:")
    armijo_ls = ArmijoLineSearch()
    bfgs_ls = BFGS(loss, line_search=armijo_ls)
    trace = run!(bfgs_ls, x0, it_max=30)
    compute_loss_of_iterates!(trace)
    final_loss = best_loss_value(trace)
    println("BFGS + Armijo final loss: $final_loss")

    println("\nTesting L-BFGS with adaptive initialization:")
    lbfgs_adaptive = LBFGS(loss, adaptive_init=true, mem_size=3)
    trace_adaptive = run!(lbfgs_adaptive, copy(x0), it_max=30)
    compute_loss_of_iterates!(trace_adaptive)
    final_loss_adaptive = best_loss_value(trace_adaptive)
    println("L-BFGS adaptive final loss: $final_loss_adaptive")
end

function test_quasi_newton_hessian_estimates()
    println("\n\nTesting Quasi-Newton with Custom Hessian Estimates")
    println("=" ^ 40)

    # Generate quadratic problem: f(x) = 0.5 * x^T A x + b^T x
    d = 4
    Random.seed!(456)
    A = randn(d, d)
    A = A' * A + 0.1 * I(d)  # Make positive definite
    b = randn(d)

    loss = QuadraticLoss(A, b)
    x0 = randn(d)

    # Test with custom Hessian estimate
    println("\nTesting BFGS with custom Hessian estimate:")
    custom_hess = 2.0 * I(d)  # Start with scaled identity
    bfgs_custom = BFGS(loss, hess_estim=custom_hess)
    trace_custom = run!(bfgs_custom, x0, it_max=20)
    compute_loss_of_iterates!(trace_custom)
    final_loss_custom = best_loss_value(trace_custom)
    println("BFGS custom Hessian final loss: $final_loss_custom")

    # Test storing Hessian estimates
    println("\nTesting BFGS with Hessian storage:")
    bfgs_store = BFGS(loss, store_hess_estimate=true)
    trace_store = run!(bfgs_store, copy(x0), it_max=20)
    compute_loss_of_iterates!(trace_store)
    final_loss_store = best_loss_value(trace_store)
    println("BFGS store Hessian final loss: $final_loss_store")

    # Compare final Hessian estimate to true Hessian
    println("True Hessian eigenvalues: $(sort(eigvals(A)))")
    if bfgs_store.B !== nothing
        println("Final Hessian estimate eigenvalues: $(sort(eigvals(bfgs_store.B)))")
    end
end

# Run all tests
test_quasi_newton_methods()
test_quasi_newton_with_line_search()
test_quasi_newton_hessian_estimates()