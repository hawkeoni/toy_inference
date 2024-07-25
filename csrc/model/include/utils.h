#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#ifndef _PHI_UTILS_UTILS
#define _PHI_UTILS_UTILS 

#define READ_AND_CHECK(fd, buf, size)                                  \
    do {                                                               \
        ssize_t _bytesRead = read((fd), (buf), (size));                \
        if (_bytesRead != (size)) {                                    \
            perror("Error reading from file descriptor");              \
            fprintf(stderr, "File: %s, Line: %d\n", __FILE__, __LINE__);\
            fprintf(stderr, "Expected %zu bytes, but got %zd\n",       \
                    (size_t)(size), _bytesRead);                       \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define TIME_FUNCTION_CALL(func, ...) \
    do { \
        time_t start, end; \
        time(&start); \
        func(__VA_ARGS__); \
        time(&end); \
        double dif = difftime(end, start); \
        printf("Function %s took %f seconds to run.\n", #func, dif); \
    } while (0)

#define TIME_FUNCTION_CALL_AVG(func, ...) \
    do { \
        int num_iters = 1; \
        double total_time = 0.0; \
        for (int i = 0; i < num_iters; ++i) { \
            time_t start, end; \
            time(&start); \
            func(__VA_ARGS__); \
            time(&end); \
            total_time += difftime(end, start); \
        } \
        double average_time = total_time / num_iters; \
        printf("Function %s took total: %fs, avg: %fs.\n", #func, total_time, average_time); \
    } while (0)
#endif