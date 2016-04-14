//
// Created by joe on 4/8/16.
//

#ifndef MATRIXREP_FBCSR_H
#define MATRIXREP_FBCSR_H

#include"prefix.h"
#include"CSR.h"
#include"list.h"
#include"vector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fbcsr {
    float thresh;
    int n, m;
    int c, r;
    int nr;
    int nnz;
    int nb;       // number of blocks
    int nelem;    // number of elements
    int *rptr;      // Integer array, storing the row partitioning information
    void *optKernel; // Optimized kernel function, don't know if this is ok, but just use it for now
    // Null for default
    void *getCoo; // Take a element indx and return the row and column difference, Cannot be Null
    elem_t *val;    // Scalar array, storing the values
    int *bindx;      // Integer array, storing the column indices of blocks
    int *bptr;      // Integer array; storing the pointer to the beginning of each block row
} fbcsr;

typedef struct coo {
    int r, c;
} coo;

typedef void (*fbcsrSingle_SpMVKernel)(fbcsr *f, vector *v, vector *r);

typedef coo (*fbcsr_getCoo)(int elemIndx, int elemCnt);

void fbcsr_makeEmpty(fbcsr *f, int n, int m, int c, int r, int nelem, float thresh, void *optKrnl, void *getCoo);

coo fbcsr_row(int elemIndx, int elemCnt);

coo fbcsr_column(int elemIndx, int elemCnt);

coo fbcsr_forwardSlash(int elemIndx, int elemCnt);

coo fbcsr_backwardSlash(int elemIndx, int elemCnt);

csr *csr_fbcsr(csr *c, list *l);    // c,r information embedded in list of ubcsr
// Easier to just return the remainder, TODO it is against convention
void fbcsr_csr(list *l, csr *rem, csr *c);

void fbcsr_destroy(void *f);

void fbcsr_SpMV(list *l, vector *v, vector *r);

void fbcsr_memCpy(list *src, list *dst, enum DeviceCopyDIR dir);

void fbcsr_CUDA_SpMV(list *l, vector *v, vector *r);

void fbcsr_CUDA_destroy(void *f);

#ifdef __cplusplus
}
#endif
#endif //MATRIXREP_FBCSR_H
