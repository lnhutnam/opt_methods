include("../optimizer.jl")

"""
Davidon–Fletcher–Powell algorithm. See
    https://arxiv.org/pdf/2004.14866.pdf
for a convergence proof and see
    https://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula
for a general description.

Arguments:
    L (float, optional): an upper bound on the smoothness constant
        to initialize the Hessian estimate
    hess_estim (Matrix, optional): initial Hessian estimate
    lr (float, optional): stepsize (default: 1)
"""
mutable struct DFP
    optimizer::Optimizer
    L::Union{Float64, Nothing}
    B::Union{Matrix{Float64}, Nothing}
    lr::Float64

    # Internal state
    B_inv::Matrix{Float64}
    grad::Vector{Float64}

    function DFP(loss; L=nothing, hess_estim=nothing, lr=1.0, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if L === nothing && hess_estim === nothing
            L_val = smoothness(loss)
            if L_val === nothing
                error("Either smoothness constant L or Hessian estimate must be provided")
            end
            L = L_val
        end

        B = hess_estim === nothing ? nothing : Matrix{Float64}(hess_estim)

        new(optimizer, L, B, lr,
            Matrix{Float64}(undef, 0, 0), Float64[])
    end
end

function step!(dfp::DFP)
    dfp.grad = gradient(dfp.optimizer.loss, dfp.optimizer.x)
    x_new = dfp.optimizer.x .- dfp.lr .* (dfp.B_inv * dfp.grad)

    if dfp.optimizer.line_search !== nothing
        x_new = dfp.optimizer.line_search(x=dfp.optimizer.x, x_new=x_new, gradient=dfp.grad)
    end

    s = x_new .- dfp.optimizer.x
    grad_new = gradient(dfp.optimizer.loss, x_new)
    y = grad_new .- dfp.grad
    dfp.grad = grad_new

    # DFP updates
    Bs = dfp.B * s
    sBs = dot(s, Bs)
    y_s = dot(y, s)

    # Update Hessian estimate (DFP formula)
    if y_s > 0
        dfp.B .+= (1 + sBs/y_s) / y_s .* (y * y') .- ((y * Bs') .+ (Bs * y')) ./ y_s
    end

    # Update inverse Hessian estimate (DFP formula)
    if y_s > 0
        B_inv_y = dfp.B_inv * y
        y_B_inv_y = dot(y, B_inv_y)

        if y_B_inv_y > 0
            dfp.B_inv .+= (s * s') ./ y_s
            dfp.B_inv .-= (B_inv_y * B_inv_y') ./ y_B_inv_y
        end
    end

    dfp.optimizer.x = x_new
end

function init_run!(dfp::DFP, x0; kwargs...)
    init_run!(dfp.optimizer, x0; kwargs...)

    dim = length(dfp.optimizer.x)

    if dfp.B === nothing
        dfp.B = dfp.L * I(dim)
        dfp.B_inv = (1.0 / dfp.L) * I(dim)
    else
        dfp.B_inv = pinv(dfp.B)
    end

    dfp.grad = gradient(dfp.optimizer.loss, dfp.optimizer.x)
end

function run!(dfp::DFP, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(dfp.optimizer.label): The number of iterations is set to $it_max.")
    end

    dfp.optimizer.t_max = t_max
    dfp.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(dfp.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(dfp.optimizer.seeds)) seeds...")
    end

    for seed in dfp.optimizer.seeds
        if seed in dfp.optimizer.finished_seeds
            continue
        end

        dfp.optimizer.rng = MersenneTwister(seed)
        dfp.optimizer.seed = seed
        loss_seed = rand(dfp.optimizer.rng, 1:MAX_SEED)
        set_seed!(dfp.optimizer.loss, loss_seed)
        init_seed!(dfp.optimizer.trace)

        if ls_it_max === nothing
            dfp.optimizer.ls_it_max = it_max
        else
            dfp.optimizer.ls_it_max = ls_it_max
        end

        if !dfp.optimizer.initialized[seed]
            init_run!(dfp, x0)
            dfp.optimizer.initialized[seed] = true
            if dfp.optimizer.line_search !== nothing
                reset!(dfp.optimizer.line_search, dfp.optimizer)
            end
        end

        it_criterion = dfp.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(dfp.optimizer.ls_it_max))")
        end

        while !check_convergence(dfp.optimizer)
            if dfp.optimizer.tolerance > 0
                dfp.optimizer.x_old_tol = copy(dfp.optimizer.x)
            end
            step!(dfp)
            save_checkpoint!(dfp.optimizer)

            if tqdm_iterations && dfp.optimizer.it % 100 == 0
                println("Iteration: $(dfp.optimizer.it)")
            end
        end

        append_seed_results!(dfp.optimizer.trace, seed)
        push!(dfp.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(dfp.optimizer.finished_seeds))/$(length(dfp.optimizer.seeds))")
        end
    end

    dfp.optimizer.seed = nothing
    return dfp.optimizer.trace
end