using LinearAlgebra, SparseArrays

"""
Implement a+b compatible with different types of input.
Supports scalars, arrays and sparse objects.
"""
function safe_sparse_add(a, b)
    both_sparse = issparse(a) && issparse(b)
    one_is_scalar = isa(a, Number) || isa(b, Number)

    if both_sparse || one_is_scalar
        # both are sparse, keep the result sparse
        return a + b
    else
        # one of them is non-sparse, convert everything to dense
        if issparse(a)
            a = Array(a)
            if ndims(a) == 2 && ndims(b) == 1
                b = vec(b)
            end
        elseif issparse(b)
            b = Array(b)
            if ndims(b) == 2 && ndims(a) == 1
                b = vec(b)
            end
        end
        return a + b
    end
end

function safe_sparse_inner_prod(a, b)
    if issparse(a) && issparse(b)
        if ndims(a) == 2 && size(a, 2) == size(b, 1)
            return (a * b)[1, 1]
        elseif size(a, 1) == size(b, 1)
            return (a' * b)[1, 1]
        else
            return (a * b')[1, 1]
        end
    end

    if issparse(a)
        a = Array(a)
    elseif issparse(b)
        b = Array(b)
    end

    return dot(a, b)
end

function safe_sparse_multiply(a, b)
    if issparse(a) && issparse(b)
        return a .* b
    end

    if issparse(a)
        a = Array(a)
    elseif issparse(b)
        b = Array(b)
    end

    return a .* b
end

function safe_sparse_norm(a, ord=2)
    if issparse(a)
        if ord == 1
            return norm(nonzeros(a), 1)
        elseif ord == 2
            return norm(nonzeros(a), 2)
        else
            return norm(Array(a), ord)
        end
    end
    return norm(a, ord)
end