module OptMethods

# Core exports
export Optimizer, run!, step!, init_run!
export Oracle, BaseOracle, value, gradient, hessian, smoothness
export Trace, init_seed!, append_seed_results!, compute_loss_of_iterates!, best_loss_value
export plot_losses!, plot_distances!, save, from_pickle
export Regularizer, BoundedL2Regularizer, prox, prox_l1, prox_l2
export LinearRegression, LogisticRegression, LogSumExp

# Algorithm exports
export GradientDescent, AdaptiveGradientDescent, AdgdAccel
export HeavyBall, NesterovAcceleratedGradient, OptimizedGradientMethod
export RestNest, NestLine
export AdaGrad, PolyakStepSize
export Ig

# Stochastic algorithm exports
export StochasticGradientDescent, SVRG, Shuffling, RootSGD

# Line search exports
export LineSearch, BaseLineSearch
export ArmijoLineSearch, WolfeLineSearch, GoldsteinLineSearch
export BestGridLineSearch, NesterovArmijoLineSearch, RegularizedNewtonLineSearch

# Utility exports
export relative_round, get_trace
export safe_sparse_add, safe_sparse_inner_prod, safe_sparse_multiply, safe_sparse_norm

# Include core files
include("../optmethods/utils.jl")
include("../optmethods/opt_trace.jl")
include("../optmethods/optimizer.jl")

# Loss functions
include("../optmethods/loss/loss.jl")

# Line search methods
include("../optmethods/line_search/line_search.jl")

# First order methods
include("../optmethods/first_order/gd.jl")
include("../optmethods/first_order/nesterov.jl")
include("../optmethods/first_order/heavy_ball.jl")
include("../optmethods/first_order/adagrad.jl")
include("../optmethods/first_order/polyak.jl")
include("../optmethods/first_order/ogm.jl")
include("../optmethods/first_order/adgd.jl")
include("../optmethods/first_order/adgd_accel.jl")
include("../optmethods/first_order/rest_nest.jl")
include("../optmethods/first_order/nest_line.jl")
include("../optmethods/first_order/ig.jl")

# Stochastic first order methods
include("../optmethods/stochastic_first_order/stochastic_first_order.jl")

end # module