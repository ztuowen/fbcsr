#ifndef VECTOR_H
#define VECTOR_H

#include"prefix.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct vector {
    int n;
    elem_t *val;
} vector;

int vector_equal(vector *a, vector *b);

void vector_init(vector *v, int n);

void vector_destroy(void *v);

void vector_memCpy(vector *src,vector *dst,enum DeviceCopyDIR dir);

void vector_CUDA_destroy(void *v);

#ifdef __cplusplus
}
#endif
#endif
