#ifndef CSR_H
#define CSR_H

#include "prefix.h"
#include "vector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct csr {
    int n, m;
    int nnz;
    int *ptr;       // Guarded
    int *indx;
    elem_t *val;
} csr;

void csr_destroy(void *c);

void csr_readFile(char *filename, csr *c);

void csr_merge(csr *a, csr *b); // a=a+b also be used for deepcopy

void csr_makeEmpty(csr *c, int n, int m);

void csr_SpMV(csr *m, vector *v, vector *r);

void csr_memCpy(csr *src, csr *dst, enum DeviceCopyDIR dir);

void csr_CUDA_SpMV(csr *m, vector *v, vector *r);

void csr_CUDA_destroy(void *c);

#ifdef __cplusplus
}
#endif

#endif
