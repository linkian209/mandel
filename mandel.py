#!env\Scripts\python.exe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit

# Functions
def mandel_set(xmin, xmax, x_steps, ymin, ymax, y_steps, max_iter, func):
    # Cuda Setup
    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, x_steps).astype(np.complex64), 
        np.linspace(ymax, ymin, y_steps).astype(np.complex64)
    )
    z = X + 1.j * Y
    z_gpu = gpuarray.to_gpu(z)

    retval_gpu = gpuarray.zeros_like(z_gpu, dtype=np.int32)

    func(
        z_gpu, retval_gpu, np.int32(max_iter), np.int32(x_num_steps),
        np.int32(y_num_steps), block=block, grid=grid_size
    )

    return retval_gpu.get()

def on_zoom(axes):
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()

    retval = mandel_set(
        xmin, xmax, x_num_steps, ymin, ymax, 
        y_num_steps, max_iter, gpu_mandel
    )

    norm = colors.PowerNorm(.3)
    cs = axes.imshow(retval,cmap='hot', norm=norm, extent=[xmin,xmax,ymin,ymax])

# Get Starting Values
xmin = -2
xmax = 1
x_num_steps = 2688
ymin = -1.25
ymax = 1.25
y_num_steps = 2688
max_iter = 2048

# CUDA creation
BLOCK_SIZE = int(np.sqrt(cuda.Device(0).get_attribute(
    pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK
)))
block = (BLOCK_SIZE, BLOCK_SIZE, 1)
dx, mx = divmod(x_num_steps, BLOCK_SIZE)
dy, my = divmod(y_num_steps, BLOCK_SIZE)
grid_size = (
    (dx + (mx>0)), (dy + (my>0))
)

# Cuda SourceModule
mod = SourceModule(
'''
#include <pycuda-complex.hpp>
#include <stdio.h>

__global__ void mandel(pycuda::complex<float> *Z, int *retval, int max_iter, int cols, int rows)
{
    int x = threadIdx.x + (blockIdx.x * blockDim.x),
        y = threadIdx.y + (blockIdx.y * blockDim.y),
        gda = blockDim.x * gridDim.x;

    if(x < cols && y < rows)
    {
        int idx = y + x * gda;

        pycuda::complex<float> f,
                            c = Z[idx],
                            z = c;

        retval[idx] = 0;

        for(int i = 0; i < max_iter; ++i)
        {
            f = z * z + c;
            
            if(abs(f) > 2)
            {
                retval[idx] = i + 1;
            }
            z = f;
        }
    }
}
'''
)

gpu_mandel = mod.get_function('mandel')

# Create plot
fig = plt.figure()
mandel_plt = fig.add_subplot(1,1,1)

# Y update happens second
fig.axes[0].callbacks.connect('ylim_changed', on_zoom)

fig.axes[0].set_xlim(xmin, xmax)
fig.axes[0].set_ylim(ymin, ymax)

plt.show()