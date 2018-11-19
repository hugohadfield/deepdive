
import numpy as np
import numba
import math
from numba import cuda

maxfloat = np.finfo(float).max


@cuda.jit(device=True)
def CUDA_COST_FUNC(x):
    return (x[0] + 2)**2 + (x[1] + 2)**2 + (x[2]-5.5)**2

CUDA_N_DIMS = 3


@cuda.jit(device=True)
def clear_array(x):
    for i in range(x.shape[0]):
        x[i] = 0


@cuda.jit(device=True)
def linesearch_device(x, direction, max_step_size):
    # Record our starting point
    x0 = cuda.local.array(CUDA_N_DIMS, dtype=numba.float64)
    for i in range(CUDA_N_DIMS):
        x0[i] = x[i]
    f0 = CUDA_COST_FUNC(x)
    # Evaluate a local gradient
    for i in range(CUDA_N_DIMS):
        x[i] = x0[i] + 0.0001*direction[i]
    gradfval = (CUDA_COST_FUNC(x)-f0)/0.0001
    if gradfval > 0:
        gradfval = -gradfval
        t = -max_step_size
    else:
        t = max_step_size
    # Start looking for smaller points
    for i in range(CUDA_N_DIMS):
        x[i] = x0[i] + t*direction[i]
    oldf = CUDA_COST_FUNC(x)
    gamma = 0.9
    for i in range(5000):
        # Update x
        for k in range(CUDA_N_DIMS):
            x[k] = x0[k] + t*direction[k]
        newf = CUDA_COST_FUNC(x)
        # Check if the new cost meets the stopping critera
        if newf <= f0 + t*gradfval:
            return newf
        else:
            # Check if our back tracking is actually increasing the function
            if newf > oldf:
                return newf
            t = gamma*t
            oldf = newf
    # If we get here then we have found no improvements
    for i in range(CUDA_N_DIMS):
        x[i] = x0[i]
    return f0


@cuda.jit(device=True)
def coordinate_descent_device(x, n_iterations, fthreshold):
    f0 = CUDA_COST_FUNC(x)
    fxold = f0
    direction = cuda.local.array(CUDA_N_DIMS, dtype=numba.float64)
    for i in range(n_iterations):
        for j in range(CUDA_N_DIMS):
            direction[j] = 1.0
            max_step_size = 1.0
            fx = linesearch_device(x, direction, max_step_size)
            direction[j] = 0.0
        if fxold -fx < fthreshold:
            return fx
        fold = fx
    return fx


@cuda.jit
def cuda_coordinate_descent_kernel(x, n_iterations, fthreshold, outputf):
    i = numba.cuda.grid(1)
    if i < x.shape[1]:
        inputx = x[:,i]
        fx = coordinate_descent_device(inputx, n_iterations, fthreshold)


@numba.njit
def linesearch(cost_func, x, direction, max_step_size):
    x0 = +x
    f0 = cost_func(x)
    gradfval = (cost_func(x+0.0001*direction)-f0)/0.0001
    if gradfval > 0:
        gradfval = -gradfval
        t = -max_step_size
    else:
        t = max_step_size
    oldf = cost_func(x0 + t*direction)
    gamma = 0.9
    for i in range(5000):
        newx = x0 + t*direction
        newf = cost_func(newx)
        if newf <= f0 + t*gradfval:
            x[:] = 0.0
            x += newx
            return newf
        else:
            if newf > oldf:
                x[:] = 0.0
                x += newx
                return newf
            t = gamma*t
            oldf = newf
    x[:] = 0.0
    x += x0
    return f0


@numba.njit
def coordinate_descent(cost_func, x, n_iterations, fthreshold):
    n_dims = x.shape[0]
    f0 = cost_func(x)
    fxold = f0
    direction = np.zeros(n_dims)
    for i in range(n_iterations):
        for j in range(n_dims):
            direction[j] = 1.0
            max_step_size = 1.0
            fx = linesearch(cost_func, x, direction, max_step_size)
            direction[j] = 0.0
        if fxold -fx < fthreshold:
            return fx
        fold = fx
    return fx


@numba.njit
def cost_func_x2(x):
    return (x[0] + 2)**2 + (x[1] + 2)**2 + (x[2]-5.5)**2


if __name__ == '__main__':

    import time

    x = np.zeros(3)
    fthreshold = 0.000001
    s = time.time()
    for i in range(10000):
        fx = coordinate_descent(cost_func_x2, x, 1000, fthreshold)
    print(time.time() - s)

    npar = 10000
    x = np.zeros((3,npar),dtype=np.float64)
    outputf = np.zeros((1,npar),dtype=np.float64)

    blockdim = 64
    griddim = int(math.ceil(x.shape[1] / blockdim))

    s = time.time()
    cuda_coordinate_descent_kernel[griddim,blockdim](x, 1000, fthreshold, outputf)
    print(time.time() - s)

    #print(x)
    #print(outputf)

