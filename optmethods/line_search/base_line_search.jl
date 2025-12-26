"""
    LineSearch

Abstract type for line search methods that find the best step size α
such that x + α*direction satisfies certain conditions (e.g., sufficient decrease,
curvature conditions, etc.).
"""
abstract type LineSearch end

"""
    BaseLineSearch(; lr0=1.0, count_first_it=false, count_last_it=true, it_max=50, tolerance=0.0)

Base implementation for line search methods.

# Arguments
- `lr0::Float64=1.0`: Initial learning rate estimate
- `count_first_it::Bool=false`: Count first iteration (false for methods reusing info)
- `count_last_it::Bool=true`: Count last iteration
- `it_max::Int=50`: Maximum inner iterations per call
- `tolerance::Float64=0.0`: Allowed condition violation
"""
mutable struct BaseLineSearch <: LineSearch
    lr0::Float64
    lr::Float64
    count_first_it::Bool
    count_last_it::Bool
    it::Int
    it_max::Int
    tolerance::Float64

    # Set during optimization
    optimizer  # Optimizer type, but can't annotate due to load order
    loss  # Oracle type, but can't annotate due to load order
    use_prox::Bool

    function BaseLineSearch(; lr0=1.0, count_first_it=false, count_last_it=true,
                           it_max=50, tolerance=0.0)
        new(lr0, lr0, count_first_it, count_last_it, 0, it_max, tolerance,
            nothing, nothing, false)
    end
end

function it_per_call(ls::LineSearch)
    return Int(ls.count_first_it) + Int(ls.count_last_it)
end

function reset!(ls::LineSearch, optimizer)
    ls.lr = ls.lr0
    ls.it = 0
    ls.optimizer = optimizer
    ls.loss = optimizer.loss
    ls.use_prox = optimizer.use_prox
end

# Abstract method to be implemented by concrete line search types
function (ls::LineSearch)(; x=nothing, direction=nothing, x_new=nothing)
    error("Line search call method must be implemented by concrete type")
end