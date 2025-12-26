# Loss functions and oracles module

# Base oracle interface
include("loss_oracle.jl")

# Utility functions
include("utils.jl")

# Regularizers
include("regularizer.jl")
include("bounded_l2.jl")

# Loss functions
include("linear_regression.jl")
include("logistic_regression.jl")
include("log_sum_exp.jl")

# Export all loss types and functions
export Oracle, BaseOracle
export LinearRegression, LogisticRegression, LogSumExp
export Regularizer, BoundedL2Regularizer

# Export utility functions
export safe_sparse_add, safe_sparse_inner_prod, safe_sparse_multiply, safe_sparse_norm
export value, gradient, hessian, stochastic_gradient
export smoothness, max_smoothness, average_smoothness, individ_smoothness
export prox, prox_l1, prox_l2
export norm, inner_prod, outer_prod, is_equal
export set_seed!