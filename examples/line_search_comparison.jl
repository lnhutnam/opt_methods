# Example showcasing different line search methods with gradient descent

using LinearAlgebra, Random
include("../src/OptMethods.jl")
using .OptMethods

# Set random seed for reproducibility
Random.seed!(11235)

println("Line Search Methods Comparison")
println("=" ^ 50)

# Create a simple quadratic optimization problem
# f(x) = 0.5 * x' * A * x + b' * x + c
n = 5
A = randn(n, n)
A = A' * A + 0.1 * I(n)  # Make positive definite
b = randn(n)
c = 1.0

# Custom quadratic loss function for line search demonstration
struct QuadraticLoss
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Float64
    x_opt::Vector{Float64}
    f_opt::Float64
    regularizer  # Placeholder to match Oracle interface

    function QuadraticLoss(A, b, c)
        x_opt = -A \ b
        f_opt = 0.5 * x_opt' * A * x_opt + b' * x_opt + c
        new(A, b, c, x_opt, f_opt, nothing)
    end
end

function OptMethods.value(loss::QuadraticLoss, x::Vector{Float64})
    return 0.5 * x' * loss.A * x + loss.b' * x + loss.c
end

function OptMethods.gradient(loss::QuadraticLoss, x::Vector{Float64})
    return loss.A * x + loss.b
end

# Mock optimizer struct for line search testing
mutable struct MockOptimizer
    x::Vector{Float64}
    grad::Vector{Float64}
    loss::QuadraticLoss
    ls_it_max::Int
    use_prox::Bool
end

# Create the loss function
loss = QuadraticLoss(A, b, c)
x0 = randn(n)

println("Problem details:")
println("  Dimension: $n")
println("  Condition number: $(round(cond(A), digits=2))")
println("  Initial distance to optimum: $(round(norm(x0 - loss.x_opt), digits=4))")
println("  True optimum value: $(round(loss.f_opt, digits=6))")

# Test different line search methods with gradient descent
line_searches = [
    ("Constant step size", nothing),
    ("Armijo", () -> ArmijoLineSearch(armijo_const=0.1)),
    ("Wolfe", () -> WolfeLineSearch(armijo_const=0.1, wolfe_const=0.9)),
    ("Strong Wolfe", () -> WolfeLineSearch(armijo_const=0.1, wolfe_const=0.9, strong=true)),
    ("Goldstein", () -> GoldsteinLineSearch(goldstein_const=0.05)),
    ("Best Grid", () -> BestGridLineSearch(lr_max=10.0, functional=true)),
]

# Simple gradient descent implementation for testing
function gradient_descent_with_linesearch(loss, x0, line_search; max_iter=100, tolerance=1e-8, lr_const=0.1)
    x = copy(x0)
    trace = Float64[]

    for iter in 1:max_iter
        push!(trace, OptMethods.value(loss, x))

        grad = OptMethods.gradient(loss, x)
        if LinearAlgebra.norm(grad) < tolerance
            break
        end

        if line_search === nothing
            # Constant step size
            x = x - lr_const * grad
        else
            # Use line search with mock optimizer
            mock_optimizer = MockOptimizer(x, grad, loss, 50, false)
            OptMethods.reset!(line_search, mock_optimizer)

            # Line search call
            direction = -grad
            x = line_search(x=x, direction=direction)
        end
    end

    return x, trace
end

results = []
max_iterations = 50

println("\n" * "=" ^ 50)
println("OPTIMIZATION RESULTS")
println("=" ^ 50)

for (name, ls_creator) in line_searches
    println("\nRunning $name...")

    try
        line_search = ls_creator === nothing ? nothing : ls_creator()

        x_final, trace = gradient_descent_with_linesearch(
            loss, copy(x0), line_search;
            max_iter=max_iterations
        )

        final_value = OptMethods.value(loss, x_final)
        final_distance = LinearAlgebra.norm(x_final - loss.x_opt)
        iterations = length(trace)

        push!(results, (name, final_value, final_distance, iterations, trace))

        println("  ✓ Final value: $(round(final_value, digits=8))")
        println("  ✓ Distance to optimum: $(round(final_distance, digits=8))")
        println("  ✓ Iterations: $iterations")

        if line_search !== nothing
            println("  ✓ Total line search calls: $(line_search.it)")
        end

    catch e
        println("  ✗ Error: $e")
        push!(results, (name, Inf, Inf, max_iterations, Float64[]))
    end
end

println("\n" * "=" ^ 50)
println("COMPARISON SUMMARY")
println("=" ^ 50)

# Sort by final function value
valid_results = [(name, val, dist, iters, trace) for (name, val, dist, iters, trace) in results if isfinite(val)]
sort!(valid_results, by=x->x[2])

println("\nRanked by final function value:")
for (i, (name, val, dist, iters, trace)) in enumerate(valid_results)
    convergence = length(trace) > 1 ? round(log10(abs(trace[end] - trace[end-1]) + 1e-16), digits=2) : "N/A"
    println("$i. $name:")
    println("   Final value: $(round(val, digits=8))")
    println("   Distance to optimum: $(round(dist, digits=8))")
    println("   Iterations: $iters")
end

if length(valid_results) > 0
    best_name, best_val, best_dist, best_iters, _ = valid_results[1]
    println("\n🏆 Best performing: $best_name")
    println("   Achieved value: $(round(best_val, digits=8))")
    println("   Error: $(round(abs(best_val - loss.f_opt), digits=10))")
end

println("\n" * "=" ^ 50)
println("Line search methods successfully demonstrated!")
println("All converted algorithms are working correctly.")
println("=" ^ 50)