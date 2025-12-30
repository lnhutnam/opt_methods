# Stochastic Second-Order Optimization Methods
# This module contains advanced second-order optimization algorithms
# that work with stochastic gradients and Hessian approximations.

include("stochastic_newton.jl")
include("stochastic_newton_cg.jl")
include("stochastic_lbfgs.jl")
include("natural_gradient.jl")
include("adahessian.jl")
