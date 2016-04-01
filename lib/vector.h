#ifndef VECTOR_H
#define VECTOR_H

#include"prefix.h"

typedef struct vector {
    int n;
    elem_t *val;
} vector;

int vector_equal(vector *a, vector *b);

void vector_init(vector *v, int n);

void vector_destroy(void *v);

#endif
