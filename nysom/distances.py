import numpy as np
import time
try:
    import cupy as cp
    default_xp = cp
    _cupy_available = True
except:
    default_xp = np
    _cupy_available = False

if _cupy_available:
    _manhattan_distance_kernel = cp.ReductionKernel(
        'T x, T w',
        'T y',
        'abs(x-w)',
        'a+b',
        'y = a',
        '0',
        'l1norm'
    )

def euclidean_squared_distance_part(x, w, w_flat_sq=None, xp=default_xp):
    """Calculate partial squared L2 distance
    This function does not sum x**2 to the result since it's not needed to 
    compute the best matching unit (it's not dependent on the neuron but
    it's a constant addition on the row).
    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    if w_flat_sq is None:
        w_flat_sq = xp.power(w_flat, 2).sum(axis=1, keepdims=True)
    cross_term = xp.dot(x, w_flat.T)
    return -2 * cross_term + w_flat_sq.T

def euclidean_distance(x, w, w_flat_sq=None, xp=default_xp):
    """Calculate L2 distance
    NB: result shape is (N,X*Y)
    """
    return xp.nan_to_num(
        xp.sqrt(
            euclidean_squared_distance(x, w, w_flat_sq, xp)
        )
    )

def manhattan_distance_legacy(x, w, xp=default_xp):
    """Calculate Manhattan distance

    It is very slow (~10x) compared to euclidean distance
    TODO: improve performance. Maybe a custom kernel is necessary

    NB: result shape is (N,X*Y)
    """

    if xp.__name__ == 'cupy':
        d = _manhattan_distance_kernel(
            x[:,xp.newaxis,xp.newaxis,:], 
            w[xp.newaxis,:,:,:], 
            axis=3
        )
        start = time.time()
        d = _manhattan_distance_kernel(
            x[:,xp.newaxis,xp.newaxis,:], 
            w[xp.newaxis,:,:,:], 
            axis=3
        )
        end = time.time()
        print(end - start)
        return d.reshape(x.shape[0], w.shape[0]*w.shape[1])
    else:
        d = xp.linalg.norm(
            x[:,xp.newaxis,xp.newaxis,:]-w[xp.newaxis,:,:,:], 
            ord=1,
            axis=3,
        )
        return d.reshape(x.shape[0], w.shape[0]*w.shape[1])

if _cupy_available:
    manhattan_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void l1_norm(const float* pixels, const float* weights, float* y, int n_samples, int n_weights, int n_dims) {
        int idx_pixel = threadIdx.x + blockDim.x * blockIdx.x;
        int idx_weight = threadIdx.y + blockDim.y * blockIdx.y;
        int idx_dim = threadIdx.z + blockDim.z * blockIdx.z;

        if(idx_pixel < n_samples && idx_weight < n_weights && idx_dim < n_dims)
        y[(idx_pixel * n_weights + idx_weight)] += fabsf(pixels[idx_pixel*n_dims+idx_dim] - weights[idx_weight*n_dims+idx_dim]);
    }
    ''', 'l1_norm')

def manhattan_distance(x, w, xp=default_xp):
    """Calculate Manhattan distance with a custom kernel.

    NB: result shape is (N,X*Y)
    """

    if xp.__name__ == 'cupy':
        n_samples = x.shape[0]
        n_dims = x.shape[1]
        n_weights = w.reshape(-1, n_dims).shape[0]

        start = time.time()
        d = xp.empty((x.shape[0], n_weights))
        manhattan_kernel(
            (n_samples,n_weights,n_dims), # the x, y and z block dimensions
            (1,), # a single thread
            (x, w.reshape(-1, n_dims), d, n_samples, n_weights, n_dims),
        )
        end = time.time()
        print(end - start)
        return d.reshape(x.shape[0], w.shape[0]*w.shape[1])
    else:
        d = xp.linalg.norm(
            x[:,xp.newaxis,xp.newaxis,:]-w[xp.newaxis,:,:,:], 
            ord=1,
            axis=3,
        )
        return d.reshape(x.shape[0], w.shape[0]*w.shape[1])
