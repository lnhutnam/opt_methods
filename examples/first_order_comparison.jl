# Example comparing different first-order optimization methods

using LinearAlgebra, Random
include("../src/OptMethods.jl")
using .OptMethods

# Set random seed for reproducibility
Random.seed!(11235)

# Create a simple logistic regression problem
n, d = 100, 10
A = randn(n, d)
b = rand([0.0, 1.0], n)

# Create loss function with L2 regularization
loss = LogisticRegression(A, b; l2=0.01)

# Starting point
x0 = randn(d)

println("Comparing First-Order Optimization Methods")
println("=" ^ 50)

# Test different algorithms
algorithms = [
    ("Gradient Descent", () -> GradientDescent(loss, trace_len=50)),
    ("Nesterov Accelerated", () -> NesterovAcceleratedGradient(loss, trace_len=50)),
    ("Heavy Ball", () -> HeavyBall(loss, trace_len=50)),
    ("AdaGrad", () -> AdaGrad(loss, trace_len=50)),
    ("Adaptive GD", () -> AdaptiveGradientDescent(loss, trace_len=50)),
    ("Optimized GM", () -> OptimizedGradientMethod(loss, trace_len=50)),
]

results = []
max_iterations = 200

for (name, algorithm_creator) in algorithms
    println("\nRunning $name...")

    try
        # Create algorithm instance
        algorithm = algorithm_creator()

        # Run optimization
        trace = run!(algorithm, copy(x0); it_max=max_iterations, tqdm_iterations=false)

        # Store results
        push!(results, (name, trace, best_loss_value(trace)))

        println("  ✓ Final loss: $(round(best_loss_value(trace), digits=6))")
        println("  ✓ Iterations completed: $(length(trace.its_all[first(keys(trace.its_all))]))")

    catch e
        println("  ✗ Error: $e")
        push!(results, (name, nothing, Inf))
    end
end

println("\n" * "=" ^ 50)
println("SUMMARY")
println("=" ^ 50)

# Sort results by final loss value
valid_results = [(name, trace, loss_val) for (name, trace, loss_val) in results if trace !== nothing]
sort!(valid_results, by=x->x[3])

for (i, (name, trace, loss_val)) in enumerate(valid_results)
    println("$i. $name: $(round(loss_val, digits=6))")
end

if length(valid_results) > 0
    println("\nBest performing method: $(valid_results[1][1])")
    println("Final loss value: $(round(valid_results[1][3], digits=6))")
end

# Optional: Polyak method (requires f_opt to be known)
println("\n" * "-" ^ 30)
println("Polyak Step Size (requires f_opt)")
println("-" ^ 30)

try
    # For demonstration, use best achieved loss as f_opt estimate
    best_loss = minimum([loss_val for (_, _, loss_val) in valid_results])
    f_opt_estimate = best_loss * 0.95  # Slightly optimistic estimate

    polyak = PolyakStepSize(loss; f_opt=f_opt_estimate, trace_len=50)
    trace = run!(polyak, copy(x0); it_max=max_iterations, tqdm_iterations=false)

    println("✓ Polyak method final loss: $(round(best_loss_value(trace), digits=6))")
    println("✓ Used f_opt estimate: $(round(f_opt_estimate, digits=6))")

catch e
    println("✗ Polyak method error: $e")
end

println("\nExample completed successfully!")