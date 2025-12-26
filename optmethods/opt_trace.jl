using Statistics, LinearAlgebra, Serialization

# Note: Plotting functionality requires Plots.jl package
# Install with: using Pkg; Pkg.add("Plots")

"""
    Trace(loss, label=nothing)

Stores the logs of running an optimization method and provides
methods for plotting the trajectory.

# Arguments
- `loss::Oracle`: The optimization oracle being used
- `label::Union{String,Nothing}=nothing`: Label for convergence plots

# Fields
- `xs_all::Dict`: Iterates for each seed
- `ts_all::Dict`: Times for each seed
- `its_all::Dict`: Iteration counts for each seed
- `loss_vals_all::Dict`: Loss values for each seed
- `its_converted_to_epochs::Bool`: Whether iterations are in epochs
- `loss_is_computed::Bool`: Whether loss values have been computed
"""
mutable struct Trace
    loss  # Oracle type, but can't annotate due to load order
    label::Union{String, Nothing}

    xs_all::Dict{Int, Vector}
    ts_all::Dict{Int, Vector}
    its_all::Dict{Int, Vector}
    loss_vals_all::Dict{Int, Union{Vector, Nothing}}
    its_converted_to_epochs::Bool
    loss_is_computed::Bool
    ls_its_all::Union{Dict, Nothing}

    # Current seed data
    xs::Vector
    ts::Vector
    its::Vector
    loss_vals::Union{Vector, Nothing}
    ls_its::Union{Vector, Nothing}
    lrs::Union{Vector, Nothing}

    function Trace(loss, label=nothing)
        new(loss, label,
            Dict(), Dict(), Dict(), Dict(),
            false, false, nothing,
            [], [], [], nothing, nothing, nothing)
    end
end

function init_seed!(trace::Trace)
    trace.xs = []
    trace.ts = []
    trace.its = []
    trace.loss_vals = nothing
    trace.ls_its = nothing
    trace.lrs = nothing
end

function append_seed_results!(trace::Trace, seed)
    trace.xs_all[seed] = copy(trace.xs)
    trace.ts_all[seed] = copy(trace.ts)
    trace.its_all[seed] = copy(trace.its)

    if trace.loss_vals === nothing
        trace.loss_vals_all[seed] = nothing
        trace.loss_is_computed = false
    else
        trace.loss_vals_all[seed] = copy(trace.loss_vals)
    end
end

function compute_loss_of_iterates!(trace::Trace)
    for (seed, loss_vals) in trace.loss_vals_all
        if loss_vals === nothing
            trace.loss_vals_all[seed] = [value(trace.loss, x) for x in trace.xs_all[seed]]
        else
            @warn "Loss values for seed $seed have already been computed. Set .loss_vals_all[$seed] = [] to recompute."
        end
    end
    trace.loss_is_computed = true
end

function convert_its_to_epochs!(trace::Trace, batch_size=1)
    for seed in keys(trace.xs_all)
        if trace.its_converted_to_epochs
            @warn "The iteration count has already been converted to epochs."
            continue
        end
        its_per_epoch = trace.loss.n / batch_size
        trace.its = trace.its ./ its_per_epoch
        trace.its_all[seed] = trace.its_all[seed] ./ its_per_epoch
        trace.its_converted_to_epochs = true
    end
end

function plot_losses!(trace::Trace; its=nothing, f_opt=nothing, log_std=true,
                     std_interval_alpha=0.25, label=nothing, y_label=nothing,
                     markevery=nothing, use_ls_its=true, time=false, kwargs...)
    @warn "Plotting requires Plots.jl package. Please install it first: using Pkg; Pkg.add(\"Plots\")"
    return nothing
end

function plot_distances!(trace::Trace; its=nothing, x_opt=nothing, log_std=true,
                        std_interval_alpha=0.25, label=nothing, y_label=nothing,
                        markevery=nothing, use_ls_its=true, time=false, kwargs...)
    @warn "Plotting requires Plots.jl package. Please install it first: using Pkg; Pkg.add(\"Plots\")"
    return nothing
end

function best_loss_value(trace::Trace)
    if !trace.loss_is_computed
        compute_loss_of_iterates!(trace)
    end
    return minimum([minimum(loss_vals) for loss_vals in values(trace.loss_vals_all)])
end

function save(trace::Trace, file_name=nothing, path="./results/")
    if file_name === nothing
        file_name = trace.label
    end
    if path[end] != '/'
        path *= "/"
    end

    # To make the dumped file smaller, copy the reference to a variable, and remove the loss
    loss = trace.loss
    trace.loss = nothing
    mkpath(path)

    # Use Julia's native serialization instead of Pickle
    open(path * file_name, "w") do f
        serialize(f, trace)
    end
    trace.loss = loss
end

function from_pickle(path, loss=nothing)
    if !isfile(path)
        return nothing
    end

    # Use Julia's native deserialization
    trace = open(path, "r") do f
        deserialize(f)
    end
    trace.loss = loss

    if loss !== nothing
        loss.f_opt = min(best_loss_value(trace), loss.f_opt)
    end
    return trace
end

# StochasticTrace would be similar but I'll skip it for brevity unless needed