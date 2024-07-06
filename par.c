#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 65536


int main(void) {
    float *a = (float*)malloc(sizeof(float) * N);
    float *b = (float*)malloc(sizeof(float) * N);
    int i;
    srand(0);
    for (i = 0; i < N; i++) {
        a[i] = (float)(rand()) / RAND_MAX;
        b[i] = (float)(rand()) / RAND_MAX;
    }

    float sum = 0;
    double start = omp_get_wtime();
    // clock_t start = clock();
    for (i = 0; i < N; i++) {
        sum += a[i] * b[i];
    }
    // printf("%f %f\n", (float)(clock() - start) / CLOCKS_PER_SEC, sum);
    printf("%f %f\n", omp_get_wtime() - start, sum);

    sum = 0;
    // start = clock();
    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < N; i++) {
        sum += a[i] * b[i];
    }
    // printf("%f %f\n", (float)(clock() - start) / CLOCKS_PER_SEC, sum);
    printf("%f %f\n", omp_get_wtime() - start, sum);
    free(a);
    free(b);
    return 0;
}