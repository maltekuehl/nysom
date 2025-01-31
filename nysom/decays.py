from numpy import exp, log

def exponential_decay(val0, valN, curr_iter, max_iter):
    """Exponential decay function of the learning process.
    Parameters
    ----------
    val0 : float
        initial value.

    valN : float
        final value.

    curr_iter : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    if valN == 0:
        diff = -log(0.1) / max_iter
    else:
        diff = -log(valN / val0) / max_iter
    return val0 * exp(-curr_iter * diff)
