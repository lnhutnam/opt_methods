include("../optimizer.jl")

"""
Broyden–Fletcher–Goldfarb–Shanno algorithm. See
    https://arxiv.org/pdf/2004.14866.pdf
for a convergence proof and see
    https://en.wikipedia.org/wiki/BFGS
for a general description.

Arguments:
    L (float, optional): an upper bound on the smoothness constant
        to initialize the Hessian estimate
    hess_estim (Matrix, optional): initial Hessian estimate
    lr (float, optional): stepsize (default: 1)
    store_hess_estimate (bool, optional): whether to store the Hessian estimate (default: false)
"""
mutable struct BFGS
    optimizer::Optimizer
    L::Union{Float64, Nothing}
    B::Union{Matrix{Float64}, Nothing}
    lr::Float64
    store_hess_estimate::Bool

    # Internal state
    B_inv::Matrix{Float64}
    grad::Vector{Float64}

    function BFGS(loss; L=nothing, hess_estim=nothing, lr=1.0,
                  store_hess_estimate=false, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if L === nothing && hess_estim === nothing
            L_val = smoothness(loss)
            if L_val === nothing
                error("Either smoothness constant L or Hessian estimate must be provided")
            end
            L = L_val
        end

        B = hess_estim === nothing ? nothing : Matrix{Float64}(hess_estim)

        new(optimizer, L, B, lr, store_hess_estimate,
            Matrix{Float64}(undef, 0, 0), Float64[])
    end
end

function step!(bfgs::BFGS)
    bfgs.grad = gradient(bfgs.optimizer.loss, bfgs.optimizer.x)
    x_new = bfgs.optimizer.x .- bfgs.lr .* (bfgs.B_inv * bfgs.grad)

    if bfgs.optimizer.line_search !== nothing
        x_new = bfgs.optimizer.line_search(x=bfgs.optimizer.x, x_new=x_new, gradient=bfgs.grad)
    end

    s = x_new .- bfgs.optimizer.x
    grad_new = gradient(bfgs.optimizer.loss, x_new)
    y = grad_new .- bfgs.grad
    bfgs.grad = grad_new

    y_s = dot(y, s)

    # Update Hessian estimate if requested
    if bfgs.store_hess_estimate
        Bs = bfgs.B * s
        sBs = dot(s, Bs)
        bfgs.B .+= (y * y') ./ y_s .- (Bs * Bs') ./ sBs
    end

    # Update inverse Hessian estimate (BFGS update)
    if y_s > 0
        B_inv_y = bfgs.B_inv * y
        y_B_inv_y = dot(y, B_inv_y)

        bfgs.B_inv .+= (y_s + y_B_inv_y) .* (s * s') ./ y_s^2
        bfgs.B_inv .-= (B_inv_y * s' .+ s * B_inv_y') ./ y_s
    end

    bfgs.optimizer.x = x_new
end

function init_run!(bfgs::BFGS, x0; kwargs...)
    init_run!(bfgs.optimizer, x0; kwargs...)

    dim = length(bfgs.optimizer.x)

    if bfgs.B === nothing
        if bfgs.store_hess_estimate
            bfgs.B = bfgs.L * I(dim)
        end
        bfgs.B_inv = (1.0 / bfgs.L) * I(dim)
    else
        bfgs.B_inv = pinv(bfgs.B)
    end

    bfgs.grad = gradient(bfgs.optimizer.loss, bfgs.optimizer.x)
end

function run!(bfgs::BFGS, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(bfgs.optimizer.label): The number of iterations is set to $it_max.")
    end

    bfgs.optimizer.t_max = t_max
    bfgs.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(bfgs.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(bfgs.optimizer.seeds)) seeds...")
    end

    for seed in bfgs.optimizer.seeds
        if seed in bfgs.optimizer.finished_seeds
            continue
        end

        bfgs.optimizer.rng = MersenneTwister(seed)
        bfgs.optimizer.seed = seed
        loss_seed = rand(bfgs.optimizer.rng, 1:MAX_SEED)
        set_seed!(bfgs.optimizer.loss, loss_seed)
        init_seed!(bfgs.optimizer.trace)

        if ls_it_max === nothing
            bfgs.optimizer.ls_it_max = it_max
        else
            bfgs.optimizer.ls_it_max = ls_it_max
        end

        if !bfgs.optimizer.initialized[seed]
            init_run!(bfgs, x0)
            bfgs.optimizer.initialized[seed] = true
            if bfgs.optimizer.line_search !== nothing
                reset!(bfgs.optimizer.line_search, bfgs.optimizer)
            end
        end

        it_criterion = bfgs.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(bfgs.optimizer.ls_it_max))")
        end

        while !check_convergence(bfgs.optimizer)
            if bfgs.optimizer.tolerance > 0
                bfgs.optimizer.x_old_tol = copy(bfgs.optimizer.x)
            end
            step!(bfgs)
            save_checkpoint!(bfgs.optimizer)

            if tqdm_iterations && bfgs.optimizer.it % 100 == 0
                println("Iteration: $(bfgs.optimizer.it)")
            end
        end

        append_seed_results!(bfgs.optimizer.trace, seed)
        push!(bfgs.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(bfgs.optimizer.finished_seeds))/$(length(bfgs.optimizer.seeds))")
        end
    end

    bfgs.optimizer.seed = nothing
    return bfgs.optimizer.trace
end