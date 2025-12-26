# Line Search methods module

# Base line search interface
include("base_line_search.jl")

# Specific line search implementations
include("armijo.jl")
include("wolfe.jl")
include("goldstein.jl")
include("best_grid.jl")
include("nest_armijo.jl")
include("reg_newton_ls.jl")

# Export all line search types
export LineSearch, BaseLineSearch
export ArmijoLineSearch, WolfeLineSearch, GoldsteinLineSearch
export BestGridLineSearch, NesterovArmijoLineSearch, RegularizedNewtonLineSearch
export reset!, it_per_call