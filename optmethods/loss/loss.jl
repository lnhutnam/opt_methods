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