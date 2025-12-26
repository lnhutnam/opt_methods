# First-order optimization methods module

# Basic gradient methods
include("gd.jl")
include("adgd.jl")
include("adgd_accel.jl")

# Momentum-based methods
include("heavy_ball.jl")
include("nesterov.jl")
include("ogm.jl")
include("rest_nest.jl")
include("nest_line.jl")

# Adaptive methods
include("adagrad.jl")
include("polyak.jl")

# Incremental/Stochastic methods
include("ig.jl")

# Export all algorithm types
export GradientDescent, AdaptiveGradientDescent, AdgdAccel
export HeavyBall, NesterovAcceleratedGradient, OptimizedGradientMethod
export RestNest, NestLine
export AdaGrad, PolyakStepSize
export Ig