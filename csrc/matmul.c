#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "matmul_cuda.h"


float **create_matrix(size_t dim1, size_t dim2) {
    srand(0);
    float **res = (float**)malloc(sizeof(float*) * dim1);
    for (int i = 0; i < dim1; i++) {
        res[i] = (float*)malloc(sizeof(float) * dim2);
        for (int j = 0; j < dim2; j++) {
            res[i][j] = (float)(rand()) / RAND_MAX;
        }
    }
}
float *create_vector(size_t dim){
    float *res = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) 
    res[i] = (float)(rand()) / RAND_MAX;
    return res;
}

void print_vector(float *v, size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("%f ", v[i]);
    }
    printf("\n");
}

void print_matrix(float *v, size_t dim1, size_t dim2) {
    for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
            printf("%f ", v[i * dim2 + j]);
        }
        printf("\n");
    }
    

}

void matmul_cpu_flat(float *w, float *x, float *output, size_t dim1, size_t dim2, size_t dim3) {
    // x - dim1 x dim2
    // w - dim2 x dim3
    // output - dim1 x dim3

    // output_ij = Sum x_ik * w_jk
    for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim3; j++) {
            for (size_t k = 0; k < dim2; k++) {
                output[i * dim3 + j] += x[i * dim2 + k] * w[k * dim3 + j];
            }
        }
    }
}

int main(void) {
    size_t dim1 = 16, dim2 = 32, batch_size = 1;
    float *x = create_vector(batch_size * dim1);
    float *w = create_vector(dim1 * dim2);
    float *r_cpu = (float*)calloc(batch_size * dim2, sizeof(float));
    float *r_gpu = (float*)calloc(batch_size * dim2, sizeof(float));
    // print_matrix(x, batch_size, dim1);
    // printf("\n");
    // print_matrix(w, dim1, dim2);
    // printf("\n");
    // matmul_cpu_flat(w, x, r_cpu, batch_size, dim1, dim2);
    // print_matrix(r_cpu, batch_size, dim2);
    // printf("\n");

    matmul_gpu_flat(w, x, r_gpu, batch_size, dim1, dim2);
    print_matrix(r_gpu, batch_size, dim2);
    printf("\n");
    return 0;
}