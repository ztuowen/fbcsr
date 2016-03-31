#ifndef _CSR_H
#define _CSR_H

#include "prefix.h"
#include "vector.h"

typedef struct csr {
    int n,m;
    int *ptr;       // Guarded
    int *indx;
    elem_t *val;
} csr;

void csr_destroy(void* c);
void csr_merge(csr *a, csr *b); // a=a+b also be used for deepcopy
void csr_makeEmpty(csr *c,int n,int m);
void csr_SpMV(csr *m, vector *v, vector *r);

#endif
