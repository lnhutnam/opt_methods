include("../optimizer.jl")

"""
Quasi-Newton algorithm with Symmetric Rank 1 (SR1) update. See
    https://arxiv.org/pdf/2002.00657.pdf
for a formal description and convergence proof of a similar method.

The stability condition is from
    p. 145 in (J. Nocedal and S. J. Wright, "Numerical Optimization", 2nd edition)

Arguments:
    L (float, optional): an upper bound on the smoothness constant
        to initialize the Hessian estimate
    hess_estim (Matrix, optional): initial Hessian estimate
    lr (float, optional): stepsize (default: 1)
    stability_const (float, optional): a constant from [0, 1) that ensures a curvature
        condition before updating the Hessian-inverse estimate (default: 1e-8)
"""
mutable struct SR1
    optimizer::Optimizer
    L::Union{Float64, Nothing}
    B::Union{Matrix{Float64}, Nothing}
    lr::Float64
    stability_const::Float64

    # Internal state
    B_inv::Matrix{Float64}
    grad::Vector{Float64}

    function SR1(loss; L=nothing, hess_estim=nothing, lr=1.0,
                 stability_const=1e-8, kwargs...)
        optimizer = Optimizer(loss; kwargs...)

        if L === nothing && hess_estim === nothing
            L_val = smoothness(loss)
            if L_val === nothing
                error("Either smoothness constant L or Hessian estimate must be provided")
            end
            L = L_val
        end

        if !(0 <= stability_const < 1)
            error("Invalid stability parameter: $stability_const")
        end

        B = hess_estim === nothing ? nothing : Matrix{Float64}(hess_estim)

        new(optimizer, L, B, lr, stability_const,
            Matrix{Float64}(undef, 0, 0), Float64[])
    end
end

function step!(sr1::SR1)
    sr1.grad = gradient(sr1.optimizer.loss, sr1.optimizer.x)
    x_new = sr1.optimizer.x .- sr1.lr .* (sr1.B_inv * sr1.grad)

    if sr1.optimizer.line_search !== nothing
        x_new = sr1.optimizer.line_search(x=sr1.optimizer.x, x_new=x_new, gradient=sr1.grad)
    end

    s = x_new .- sr1.optimizer.x
    grad_new = gradient(sr1.optimizer.loss, x_new)
    y = grad_new .- sr1.grad
    sr1.grad = grad_new

    # SR1 updates with stability condition
    Bs = sr1.B * s
    sBs = dot(s, Bs)
    B_inv_y = sr1.B_inv * y
    y_B_inv_y = dot(y, B_inv_y)
    y_s = dot(y, s)

    # Stability condition from Nocedal & Wright
    stability_check = abs(y_s - sBs) > sr1.stability_const * norm(s) * norm(y .- Bs)

    if stability_check && y_s != y_B_inv_y
        # Update Hessian estimate (SR1 formula)
        if abs(y_s - sBs) > eps()
            update_vec = y .- Bs
            sr1.B .+= (update_vec * update_vec') ./ (y_s - sBs)
        end

        # Update inverse Hessian estimate (SR1 formula)
        if abs(y_s - y_B_inv_y) > eps()
            update_vec_inv = s .- B_inv_y
            sr1.B_inv .+= (update_vec_inv * update_vec_inv') ./ (y_s - y_B_inv_y)
        end
    end

    sr1.optimizer.x = x_new
end

function init_run!(sr1::SR1, x0; kwargs...)
    init_run!(sr1.optimizer, x0; kwargs...)

    dim = length(sr1.optimizer.x)

    if sr1.B === nothing
        sr1.B = sr1.L * I(dim)
        sr1.B_inv = (1.0 / sr1.L) * I(dim)
    else
        sr1.B_inv = pinv(sr1.B)
    end

    sr1.grad = gradient(sr1.optimizer.loss, sr1.optimizer.x)
end

function run!(sr1::SR1, x0; t_max=Inf, it_max=Inf, ls_it_max=nothing,
              tqdm_seeds=true, tqdm_iterations=true)
    if t_max == Inf && it_max == Inf
        it_max = 100
        println("$(sr1.optimizer.label): The number of iterations is set to $it_max.")
    end

    sr1.optimizer.t_max = t_max
    sr1.optimizer.it_max = it_max

    tqdm_seeds = tqdm_seeds && length(sr1.optimizer.seeds) > 1
    if tqdm_seeds
        println("Running $(length(sr1.optimizer.seeds)) seeds...")
    end

    for seed in sr1.optimizer.seeds
        if seed in sr1.optimizer.finished_seeds
            continue
        end

        sr1.optimizer.rng = MersenneTwister(seed)
        sr1.optimizer.seed = seed
        loss_seed = rand(sr1.optimizer.rng, 1:MAX_SEED)
        set_seed!(sr1.optimizer.loss, loss_seed)
        init_seed!(sr1.optimizer.trace)

        if ls_it_max === nothing
            sr1.optimizer.ls_it_max = it_max
        else
            sr1.optimizer.ls_it_max = ls_it_max
        end

        if !sr1.optimizer.initialized[seed]
            init_run!(sr1, x0)
            sr1.optimizer.initialized[seed] = true
            if sr1.optimizer.line_search !== nothing
                reset!(sr1.optimizer.line_search, sr1.optimizer)
            end
        end

        it_criterion = sr1.optimizer.ls_it_max != Inf
        if tqdm_iterations && it_criterion
            println("Starting optimization with max iterations: $(Int(sr1.optimizer.ls_it_max))")
        end

        while !check_convergence(sr1.optimizer)
            if sr1.optimizer.tolerance > 0
                sr1.optimizer.x_old_tol = copy(sr1.optimizer.x)
            end
            step!(sr1)
            save_checkpoint!(sr1.optimizer)

            if tqdm_iterations && sr1.optimizer.it % 100 == 0
                println("Iteration: $(sr1.optimizer.it)")
            end
        end

        append_seed_results!(sr1.optimizer.trace, seed)
        push!(sr1.optimizer.finished_seeds, seed)
        if tqdm_seeds
            println("Completed seed $(length(sr1.optimizer.finished_seeds))/$(length(sr1.optimizer.seeds))")
        end
    end

    sr1.optimizer.seed = nothing
    return sr1.optimizer.trace
end