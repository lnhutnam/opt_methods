"""
    relative_round(x)

Round a number to its most significant digits while preserving magnitude.
Useful for creating consistent file names from floating point values.

# Example
```julia
relative_round(0.00123456) # Returns a value rounded to 3 significant digits
```
"""
function relative_round(x)
    mantissa, exponent = frexp(x)
    return round(mantissa; digits=3) * 2^exponent
end

function get_trace(path, loss)
    if !isfile(path)
        return nothing
    end
    # Note: Pickle functionality would require Pickle.jl package
    # For now, return nothing and implement native Julia serialization
    @warn "get_trace requires Pickle.jl package for Python compatibility. Use native Julia serialization instead."
    return nothing
end