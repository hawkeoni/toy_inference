#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

#endif