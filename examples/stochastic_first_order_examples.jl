using LinearAlgebra
using Random
include("../optmethods/loss/linear_regression.jl")
include("../optmethods/loss/logistic_regression.jl")
include("../optmethods/stochastic_first_order/sgd.jl")
include("../optmethods/stochastic_first_order/svrg.jl")
include("../optmethods/stochastic_first_order/shuffling.jl")
include("../optmethods/stochastic_first_order/root_sgd.jl")

function test_sgd_basic()
    println("Testing SGD on Linear Regression")
    println("=" ^ 40)

    # Generate test data
    Random.seed!(42)
    n, d = 100, 5
    X = randn(n, d)
    y = randn(n)

    loss = LinearRegression(X, y)
    x0 = zeros(d)

    # Analytical solution
    x_opt = X \ y

    println("\nOptimal solution: $x_opt")
    println("Optimal value: $(value(loss, x_opt))")

    println("\nTesting SGD with constant learning rate:")
    sgd = StochasticGradientDescent(loss, lr0=0.01, batch_size=10)
    result_sgd = run!(sgd, copy(x0), it_max=1000)
    compute_loss_of_iterates!(result_sgd)
    seed = first(keys(result_sgd.xs_all))
    println("SGD iterations: $(result_sgd.its_all[seed][end])")
    println("SGD final solution: $(result_sgd.xs_all[seed][end])")
    println("SGD final loss: $(result_sgd.loss_vals_all[seed][end])")
    println("SGD error: $(norm(result_sgd.xs_all[seed][end] - x_opt))")

    println("\nTesting SGD with learning rate decay:")
    sgd_decay = StochasticGradientDescent(loss, lr0=0.1, lr_decay_coef=0.01,
                                         batch_size=10, it_start_decay=100)
    result_sgd_decay = run!(sgd_decay, copy(x0), it_max=1000)
    compute_loss_of_iterates!(result_sgd_decay)
    seed = first(keys(result_sgd_decay.xs_all))
    println("SGD+decay iterations: $(result_sgd_decay.its_all[seed][end])")
    println("SGD+decay final solution: $(result_sgd_decay.xs_all[seed][end])")
    println("SGD+decay final loss: $(result_sgd_decay.loss_vals_all[seed][end])")
    println("SGD+decay error: $(norm(result_sgd_decay.xs_all[seed][end] - x_opt))")
end

function test_svrg()
    println("\n\nTesting SVRG on Logistic Regression")
    println("=" ^ 45)

    # Generate test data
    Random.seed!(123)
    n, d = 100, 5
    X = randn(n, d)
    y = rand(0:1, n)

    loss = LogisticRegression(X, y)
    x0 = zeros(d)

    println("\nTesting SVRG (loopless):")
    svrg = SVRG(loss, lr=0.01, batch_size=10, loopless=true)
    result_svrg = run!(svrg, copy(x0), it_max=1000)
    compute_loss_of_iterates!(result_svrg)
    seed = first(keys(result_svrg.xs_all))
    println("SVRG iterations: $(result_svrg.its_all[seed][end])")
    println("SVRG final loss: $(result_svrg.loss_vals_all[seed][end])")

    println("\nTesting SVRG (loop-based):")
    svrg_loop = SVRG(loss, lr=0.01, batch_size=10, loopless=false, loop_len=10)
    result_svrg_loop = run!(svrg_loop, copy(x0), it_max=1000)
    compute_loss_of_iterates!(result_svrg_loop)
    seed = first(keys(result_svrg_loop.xs_all))
    println("SVRG (loop) iterations: $(result_svrg_loop.its_all[seed][end])")
    println("SVRG (loop) final loss: $(result_svrg_loop.loss_vals_all[seed][end])")
end

function test_shuffling()
    println("\n\nTesting Shuffling on Linear Regression")
    println("=" ^ 42)

    # Generate test data
    Random.seed!(456)
    n, d = 100, 5
    X = randn(n, d)
    y = randn(n)

    loss = LinearRegression(X, y)
    x0 = zeros(d)
    x_opt = X \ y

    println("\nOptimal value: $(value(loss, x_opt))")

    println("\nTesting Shuffling without reshuffling:")
    shuf = Shuffling(loss, lr0=0.01, batch_size=10, reshuffle=false)
    result_shuf = run!(shuf, copy(x0), it_max=1000)
    compute_loss_of_iterates!(result_shuf)
    seed = first(keys(result_shuf.xs_all))
    println("Shuffling iterations: $(result_shuf.its_all[seed][end])")
    println("Shuffling final loss: $(result_shuf.loss_vals_all[seed][end])")
    println("Shuffling error: $(norm(result_shuf.xs_all[seed][end] - x_opt))")

    println("\nTesting Shuffling with reshuffling:")
    shuf_reshuf = Shuffling(loss, lr0=0.01, batch_size=10, reshuffle=true)
    result_shuf_reshuf = run!(shuf_reshuf, copy(x0), it_max=1000)
    compute_loss_of_iterates!(result_shuf_reshuf)
    seed = first(keys(result_shuf_reshuf.xs_all))
    println("Shuffling+reshuffle iterations: $(result_shuf_reshuf.its_all[seed][end])")
    println("Shuffling+reshuffle final loss: $(result_shuf_reshuf.loss_vals_all[seed][end])")
    println("Shuffling+reshuffle error: $(norm(result_shuf_reshuf.xs_all[seed][end] - x_opt))")

    println("\nTesting Shuffling with learning rate decay:")
    shuf_decay = Shuffling(loss, lr0=0.1, lr_decay_coef=0.01, epoch_start_decay=5,
                          batch_size=10, reshuffle=true)
    result_shuf_decay = run!(shuf_decay, copy(x0), it_max=1000)
    compute_loss_of_iterates!(result_shuf_decay)
    seed = first(keys(result_shuf_decay.xs_all))
    println("Shuffling+decay iterations: $(result_shuf_decay.its_all[seed][end])")
    println("Shuffling+decay final loss: $(result_shuf_decay.loss_vals_all[seed][end])")
    println("Shuffling+decay error: $(norm(result_shuf_decay.xs_all[seed][end] - x_opt))")
end

function test_root_sgd()
    println("\n\nTesting ROOT-SGD on Logistic Regression")
    println("=" ^ 43)

    # Generate test data
    Random.seed!(789)
    n, d = 100, 5
    X = randn(n, d)
    y = rand(0:1, n)

    loss = LogisticRegression(X, y)
    x0 = zeros(d)

    println("\nTesting ROOT-SGD:")
    rootsgd = RootSGD(loss, lr0=0.01, batch_size=10, first_batch=100)
    result_rootsgd = run!(rootsgd, copy(x0), it_max=1000)
    compute_loss_of_iterates!(result_rootsgd)
    seed = first(keys(result_rootsgd.xs_all))
    println("ROOT-SGD iterations: $(result_rootsgd.its_all[seed][end])")
    println("ROOT-SGD final loss: $(result_rootsgd.loss_vals_all[seed][end])")

    println("\nTesting ROOT-SGD with decay:")
    rootsgd_decay = RootSGD(loss, lr0=0.1, lr_decay_coef=0.01,
                           it_start_decay=100, batch_size=10)
    result_rootsgd_decay = run!(rootsgd_decay, copy(x0), it_max=1000)
    compute_loss_of_iterates!(result_rootsgd_decay)
    seed = first(keys(result_rootsgd_decay.xs_all))
    println("ROOT-SGD+decay iterations: $(result_rootsgd_decay.its_all[seed][end])")
    println("ROOT-SGD+decay final loss: $(result_rootsgd_decay.loss_vals_all[seed][end])")
end

function test_batch_sizes()
    println("\n\nTesting Different Batch Sizes")
    println("=" ^ 35)

    Random.seed!(999)
    n, d = 200, 10
    X = randn(n, d)
    y = randn(n)

    loss = LinearRegression(X, y)
    x0 = zeros(d)

    for batch_size in [1, 10, 50]
        println("\nBatch size: $batch_size")
        sgd = StochasticGradientDescent(loss, lr0=0.01, batch_size=batch_size)
        result = run!(sgd, copy(x0), it_max=500)
        compute_loss_of_iterates!(result)
        seed = first(keys(result.xs_all))
        println("  Final loss: $(result.loss_vals_all[seed][end])")
    end
end

# Run all tests
test_sgd_basic()
test_svrg()
test_shuffling()
test_root_sgd()
test_batch_sizes()
