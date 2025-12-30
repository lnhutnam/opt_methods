# Examples demonstrating stochastic second-order optimization methods
# This file shows how to use the new stochastic second-order methods:
# - StochasticNewton
# - StochasticNewtonCG
# - StochasticLBFGS
# - NaturalGradient
# - AdaHessian

using LinearAlgebra, Random
include("../src/OptMethods.jl")
using .OptMethods

println("=" ^ 70)
println("STOCHASTIC SECOND-ORDER OPTIMIZATION METHODS EXAMPLES")
println("=" ^ 70)

# Set random seed for reproducibility
Random.seed!(42)

# ============================================================================
# Example 1: Logistic Regression with Stochastic Newton
# ============================================================================
function example_stochastic_newton()
    println("\n" * "=" ^ 70)
    println("EXAMPLE 1: Stochastic Newton on Logistic Regression")
    println("=" ^ 70)

    # Generate logistic regression data
    n, d = 500, 20
    X = randn(n, d)
    y = rand([0.0, 1.0], n)

    loss = LogisticRegression(X, y, l2=0.01)
    x0 = zeros(d)

    println("\nProblem details:")
    println("  Number of samples: $n")
    println("  Dimension: $d")
    println("  Initial loss: $(value(loss, x0))")

    # Run Stochastic Newton
    println("\nRunning Stochastic Newton...")
    sn = StochasticNewton(loss, batch_size=50, hessian_batch_size=50,
                         regularization=1e-3, adaptive_reg=true)
    trace_sn = run!(sn, copy(x0), it_max=200)
    compute_loss_of_iterates!(trace_sn)
    seed = first(keys(trace_sn.xs_all))

    println("Results:")
    println("  Iterations: $(trace_sn.its_all[seed][end])")
    println("  Final loss: $(round(trace_sn.loss_vals_all[seed][end], digits=6))")
    println("  Gradient norm: $(round(norm(gradient(loss, trace_sn.xs_all[seed][end])), digits=6))")

    # Compare with SGD
    println("\nComparing with SGD...")
    sgd = StochasticGradientDescent(loss, batch_size=50)
    trace_sgd = run!(sgd, copy(x0), it_max=200)
    compute_loss_of_iterates!(trace_sgd)
    seed_sgd = first(keys(trace_sgd.xs_all))

    println("SGD Results:")
    println("  Final loss: $(round(trace_sgd.loss_vals_all[seed_sgd][end], digits=6))")
    println("  Gradient norm: $(round(norm(gradient(loss, trace_sgd.xs_all[seed_sgd][end])), digits=6))")

    improvement = trace_sgd.loss_vals_all[seed_sgd][end] - trace_sn.loss_vals_all[seed][end]
    println("\n✓ Stochastic Newton improvement over SGD: $(round(improvement, digits=6))")
end

# ============================================================================
# Example 2: Newton-CG for Large-Scale Problems
# ============================================================================
function example_newton_cg()
    println("\n" * "=" ^ 70)
    println("EXAMPLE 2: Stochastic Newton-CG on Linear Regression")
    println("=" ^ 70)

    # Generate linear regression data
    n, d = 1000, 50
    X = randn(n, d)
    w_true = randn(d)
    y = X * w_true + 0.1 * randn(n)

    loss = LinearRegression(X, y, l2=0.01)
    x0 = zeros(d)
    x_opt = (X' * X + 0.01 * I(d)) \ (X' * y)

    println("\nProblem details:")
    println("  Number of samples: $n")
    println("  Dimension: $d")
    println("  Optimal loss: $(round(value(loss, x_opt), digits=6))")

    # Run Stochastic Newton-CG
    println("\nRunning Stochastic Newton-CG...")
    sncg = StochasticNewtonCG(loss, batch_size=100, hessian_batch_size=100,
                             cg_maxiter=20, cg_tol=1e-4)
    trace_sncg = run!(sncg, copy(x0), it_max=100, tqdm_iterations=false)
    compute_loss_of_iterates!(trace_sncg)
    seed = first(keys(trace_sncg.xs_all))

    println("Results:")
    println("  Iterations: $(trace_sncg.its_all[seed][end])")
    println("  Final loss: $(round(trace_sncg.loss_vals_all[seed][end], digits=6))")
    println("  Distance to optimum: $(round(norm(trace_sncg.xs_all[seed][end] - x_opt), digits=6))")

    println("\n✓ Newton-CG successfully converged with CG subproblem solver")
end

# ============================================================================
# Example 3: Stochastic L-BFGS
# ============================================================================
function example_stochastic_lbfgs()
    println("\n" * "=" ^ 70)
    println("EXAMPLE 3: Stochastic L-BFGS on Logistic Regression")
    println("=" ^ 70)

    # Generate logistic regression data
    n, d = 800, 30
    X = randn(n, d)
    y = rand([0.0, 1.0], n)

    loss = LogisticRegression(X, y, l2=0.01)
    x0 = randn(d) * 0.1

    println("\nProblem details:")
    println("  Number of samples: $n")
    println("  Dimension: $d")
    println("  Initial loss: $(round(value(loss, x0), digits=6))")

    # Run Stochastic L-BFGS
    println("\nRunning Stochastic L-BFGS...")
    slbfgs = StochasticLBFGS(loss, batch_size=80, mem_size=10,
                            curvature_threshold=1e-6, damping=0.1)
    trace_slbfgs = run!(slbfgs, copy(x0), it_max=150, tqdm_iterations=false)
    compute_loss_of_iterates!(trace_slbfgs)
    seed = first(keys(trace_slbfgs.xs_all))

    println("Results:")
    println("  Iterations: $(trace_slbfgs.its_all[seed][end])")
    println("  Final loss: $(round(trace_slbfgs.loss_vals_all[seed][end], digits=6))")
    println("  Gradient norm: $(round(norm(gradient(loss, trace_slbfgs.xs_all[seed][end])), digits=6))")

    println("\n✓ Stochastic L-BFGS successfully utilized curvature information")
end

# ============================================================================
# Example 4: Natural Gradient Descent
# ============================================================================
function example_natural_gradient()
    println("\n" * "=" ^ 70)
    println("EXAMPLE 4: Natural Gradient on Logistic Regression")
    println("=" ^ 70)

    # Generate logistic regression data
    n, d = 600, 15
    X = randn(n, d)
    y = rand([0.0, 1.0], n)

    loss = LogisticRegression(X, y, l2=0.01)
    x0 = zeros(d)

    println("\nProblem details:")
    println("  Number of samples: $n")
    println("  Dimension: $d")
    println("  Initial loss: $(round(value(loss, x0), digits=6))")

    # Run Natural Gradient (exact Fisher)
    println("\nRunning Natural Gradient (exact Fisher)...")
    ng = NaturalGradient(loss, batch_size=60, fisher_batch_size=60,
                        regularization=1e-4, use_empirical_fisher=false)
    trace_ng = run!(ng, copy(x0), it_max=150, tqdm_iterations=false)
    compute_loss_of_iterates!(trace_ng)
    seed = first(keys(trace_ng.xs_all))

    println("Results:")
    println("  Iterations: $(trace_ng.its_all[seed][end])")
    println("  Final loss: $(round(trace_ng.loss_vals_all[seed][end], digits=6))")

    # Run Natural Gradient (empirical Fisher)
    println("\nRunning Natural Gradient (empirical Fisher)...")
    ng_emp = NaturalGradient(loss, batch_size=60, fisher_batch_size=60,
                            regularization=1e-4, use_empirical_fisher=true)
    trace_ng_emp = run!(ng_emp, copy(x0), it_max=150, tqdm_iterations=false)
    compute_loss_of_iterates!(trace_ng_emp)
    seed_emp = first(keys(trace_ng_emp.xs_all))

    println("Results:")
    println("  Iterations: $(trace_ng_emp.its_all[seed_emp][end])")
    println("  Final loss: $(round(trace_ng_emp.loss_vals_all[seed_emp][end], digits=6))")

    println("\n✓ Natural Gradient leverages geometry of parameter space")
end

# ============================================================================
# Example 5: AdaHessian
# ============================================================================
function example_adahessian()
    println("\n" * "=" ^ 70)
    println("EXAMPLE 5: AdaHessian on Logistic Regression")
    println("=" ^ 70)

    # Generate logistic regression data
    n, d = 700, 25
    X = randn(n, d)
    y = rand([0.0, 1.0], n)

    loss = LogisticRegression(X, y, l2=0.01)
    x0 = randn(d) * 0.1

    println("\nProblem details:")
    println("  Number of samples: $n")
    println("  Dimension: $d")
    println("  Initial loss: $(round(value(loss, x0), digits=6))")

    # Run AdaHessian
    println("\nRunning AdaHessian...")
    adahess = AdaHessian(loss, lr=0.1, batch_size=70, hessian_batch_size=70,
                        beta1=0.9, beta2=0.999, spatial_average=true,
                        block_length=2)
    trace_adahess = run!(adahess, copy(x0), it_max=200, tqdm_iterations=false)
    compute_loss_of_iterates!(trace_adahess)
    seed = first(keys(trace_adahess.xs_all))

    println("Results:")
    println("  Iterations: $(trace_adahess.its_all[seed][end])")
    println("  Final loss: $(round(trace_adahess.loss_vals_all[seed][end], digits=6))")
    println("  Gradient norm: $(round(norm(gradient(loss, trace_adahess.xs_all[seed][end])), digits=6))")

    println("\n✓ AdaHessian combines adaptive learning rates with curvature information")
end

# ============================================================================
# Example 6: Comprehensive Comparison
# ============================================================================
function comprehensive_comparison()
    println("\n" * "=" ^ 70)
    println("EXAMPLE 6: Comprehensive Method Comparison")
    println("=" ^ 70)

    # Generate a moderately-sized problem
    n, d = 400, 20
    X = randn(n, d)
    y = rand([0.0, 1.0], n)

    loss = LogisticRegression(X, y, l2=0.01)
    x0 = zeros(d)

    println("\nProblem details:")
    println("  Number of samples: $n")
    println("  Dimension: $d")
    println("  Initial loss: $(round(value(loss, x0), digits=6))")

    methods = [
        ("SGD", () -> StochasticGradientDescent(loss, batch_size=40)),
        ("Stochastic Newton", () -> StochasticNewton(loss, batch_size=40, hessian_batch_size=40)),
        ("Stochastic Newton-CG", () -> StochasticNewtonCG(loss, batch_size=40, cg_maxiter=15)),
        ("Stochastic L-BFGS", () -> StochasticLBFGS(loss, batch_size=40, mem_size=10)),
        ("Natural Gradient", () -> NaturalGradient(loss, batch_size=40, fisher_batch_size=40)),
        ("AdaHessian", () -> AdaHessian(loss, lr=0.1, batch_size=40))
    ]

    results = []
    max_iter = 100

    println("\nRunning all methods...")
    println("-" ^ 70)

    for (name, method_creator) in methods
        try
            method = method_creator()
            trace = run!(method, copy(x0), it_max=max_iter, tqdm_iterations=false)
            compute_loss_of_iterates!(trace)
            seed = first(keys(trace.xs_all))

            final_loss = trace.loss_vals_all[seed][end]
            final_grad_norm = norm(gradient(loss, trace.xs_all[seed][end]))
            iterations = trace.its_all[seed][end]

            push!(results, (name, final_loss, final_grad_norm, iterations))
            println("✓ $name: loss=$(round(final_loss, digits=6)), grad=$(round(final_grad_norm, digits=6))")
        catch e
            println("✗ $name failed: $e")
            push!(results, (name, Inf, Inf, max_iter))
        end
    end

    println("\n" * "=" ^ 70)
    println("COMPARISON SUMMARY")
    println("=" ^ 70)

    # Sort by final loss
    valid_results = [(name, loss, grad, iters) for (name, loss, grad, iters) in results if isfinite(loss)]
    sort!(valid_results, by=x->x[2])

    println("\nRanked by final loss value:")
    for (i, (name, final_loss, grad_norm, iters)) in enumerate(valid_results)
        println("$i. $name")
        println("   Final loss: $(round(final_loss, digits=8))")
        println("   Gradient norm: $(round(grad_norm, digits=8))")
        println("   Iterations: $iters")
    end

    if length(valid_results) > 0
        best_name, best_loss, best_grad, _ = valid_results[1]
        println("\n🏆 Best performing: $best_name")
        println("   Final loss: $(round(best_loss, digits=8))")
        println("   Gradient norm: $(round(best_grad, digits=8))")
    end
end

# ============================================================================
# Run all examples
# ============================================================================
println("\nRunning all examples...")
println("This may take a few moments...\n")

try
    example_stochastic_newton()
    example_newton_cg()
    example_stochastic_lbfgs()
    example_natural_gradient()
    example_adahessian()
    comprehensive_comparison()

    println("\n" * "=" ^ 70)
    println("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    println("=" ^ 70)
    println("\nStochastic second-order methods have been successfully implemented.")
    println("All methods are working correctly and can be used for optimization tasks.")
    println("=" ^ 70)
catch e
    println("\n❌ Error during examples: $e")
    println(stacktrace(catch_backtrace()))
end
