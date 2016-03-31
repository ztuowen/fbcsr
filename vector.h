#ifndef _VECTOR_H
#define _VECTOR_H

#include"prefix.h"

typedef struct vector {
    int n;
    int *val;
} vector;

int vector_equal(vector *a, vector *b);
int vector_init(vector *v,int n);
void vector_destroy(void *v);

#endif
