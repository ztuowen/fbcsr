#ifndef UBCSR_H
#define UBCSR_H

#include "prefix.h"
#include "CSR.h"
#include "list.h"
#include "vector.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct ubcsr {
    int n, m;
    int c, r;
    int nr;
    int nnz;
    int nb;       // number of blocks
    int *rptr;      // Integer array, storing the row partitioning information
    void *optKernel; // Optimized kernel function, don't know if this is ok, but just use it for now
    // Null for default;
    elem_t *val;    // Scalar array, storing the values
    int *bindx;      // Integer array, storing the column indices of blocks
    int *bptr;      // Integer array; storing the pointer to the beginning of each block row
} ubcsr;

typedef void (*ubcsrSingle_SpMVKernel)(ubcsr *u, vector *v, vector *r);

void ubcsr_makeEmpty(ubcsr *u, int n, int m, int c, int r, void *optKrnl);

csr *csr_ubcsr(csr *c, list *l, float thresh);    // c,r information embedded in list of ubcsr
// Easier to just return the remainder, TODO it is against convention
void ubcsr_csr(list *l, csr *rem, csr *c);

void ubcsr_destroy(void *u);

void ubcsr_SpMV(list *l, vector *v, vector *r);

void ubcsr_memCpy(list *src, list *dst, enum DeviceCopyDIR dir);

void ubcsr_CUDA_SpMV(list *l, vector *v, vector *r);

void ubcsr_CUDA_destroy(void *u);

#ifdef __cplusplus
}
#endif
#endif
