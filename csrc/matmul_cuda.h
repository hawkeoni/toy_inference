#include <stdlib.h>

float* move_to_device(float *vec, size_t size, bool copy);
void matmul_gpu_flat(float *w, float *x, float *output, size_t dim_1, size_t dim_2, size_t dim_3);
void matmul_gpu_flat_only_move_x(float *w_device, float *x, float *output, float *output_device, size_t dim_1, size_t dim_2, size_t dim_3);