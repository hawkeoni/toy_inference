#include <stdio.h>
#include <cuda_runtime.h>

extern "C" {
#include "matmul_cuda.h"
}


#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void matmul_gpu(float *x, float *w, float *output, unsigned int dim_1, unsigned int dim_2, unsigned int dim_3) {
    // x - dim1 x dim2
    // w - dim2 x dim3
    // output - dim1 x dim3 = xA
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim_1 && j < dim_3) {
        for (unsigned int k = 0; k < dim_2; k++) {
            output[i * dim_3 + j] += x[i * dim_2 + k] * w[k * dim_3 + j];
        }
    }
}



extern "C"
void matmul_gpu_flat(float *w, float *x, float *output, size_t dim_1, size_t dim_2, size_t dim_3) {
    // x - dim1 x dim2
    // w - dim2 x dim3
    // output - dim1 x dim3 = xA
    float *w_device, *x_device, *output_device;
    gpuErrchk(cudaMalloc(&w_device, dim_2 * dim_3 * sizeof(float)));
    gpuErrchk(cudaMalloc(&x_device, dim_1 * dim_2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&output_device, dim_1 * dim_3 * sizeof(float)));

    gpuErrchk(cudaMemcpy(w_device, w, dim_2 * dim_3 * sizeof(float), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(x_device, x, dim_1 * dim_2 * sizeof(float), cudaMemcpyHostToDevice))
    dim3 threadDim(32, 32);
    dim3 blockDim(dim_1 / 32 + 1, dim_3 / 32 + 1);
    matmul_gpu<<<blockDim, threadDim>>>(x_device, w_device, output_device, dim_1, dim_2, dim_3);
    gpuErrchk(cudaMemcpy(output, output_device, dim_1 * dim_3 * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(w_device));
    gpuErrchk(cudaFree(x_device));
    gpuErrchk(cudaFree(output_device));
}
