#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define N 32768


int main(void) {
    float *a = (float*)malloc(sizeof(float) * N);
    float *b = (float*)malloc(sizeof(float) * N);
    float *c = (float*)calloc(N, sizeof(float));
    if (c == NULL) {
        printf("Failed to malloc c\n");
        return 1;
    }
    int i, j;
    srand(0);
    for (i = 0; i < N; i++) {
        a[i] = (float)(rand()) / RAND_MAX;
        b[i] = (float)(rand()) / RAND_MAX;
    }

    float sum = 0;
    double start = omp_get_wtime();
    // clock_t start = clock();
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            c[i] += a[j] * b[j];
        }
    }
    printf("%f %f\n", omp_get_wtime() - start, c[N - 1]);

    sum = 0;
    // start = clock();
    memset(c, 0, sizeof(float) * N);
    start = omp_get_wtime();

    #pragma omp parallel for private(i,j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            c[i] += a[j] * b[j];
        }
    }

    printf("%f %f\n", omp_get_wtime() - start, c[N - 1]);
    free(a);
    free(b);
    return 0;
}