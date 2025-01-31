import numpy as np
try:
    import cupy as cp
    default_xp = cp
except:
    print("WARNING: CuPy could not be imported")
    default_xp = np

def prepare_neig_func(func, *first_args):
    def _inner(*args, **kwargs):
        return func(*first_args, *args, **kwargs)
    return _inner

def gaussian_rect(neigx, neigy, std_coeff, compact_support, c, sigma, xp=default_xp):
    """Returns a Gaussian centered in c on a rect topology

    This function is optimized wrt the generic one.
    """
    d = 2*std_coeff**2*sigma**2

    nx = neigx[xp.newaxis,:]
    ny = neigy[xp.newaxis,:]
    cx = c[0][:,xp.newaxis]
    cy = c[1][:,xp.newaxis]

    ax = xp.exp(-xp.power(nx-cx, 2, dtype=xp.float32)/d)
    ay = xp.exp(-xp.power(ny-cy, 2, dtype=xp.float32)/d)

    if compact_support:
        ax *= xp.logical_and(nx > cx-sigma, nx < cx+sigma)
        ay *= xp.logical_and(ny > cy-sigma, ny < cy+sigma)

    return ax[:,:,xp.newaxis]*ay[:,xp.newaxis,:]