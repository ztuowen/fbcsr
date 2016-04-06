#ifndef VBR_H
#define VBR_H

#include"CSR.h"
#include"prefix.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct vbr {
    int n, m;
    int nr;
    int nnz;
    int *rptr;  //Integer array. It contains the block row partitioning information--that is, the first row number of each block row.
    int *cptr;  //Integer array. It contains the block column partitioning information--that is, the first column number of each block column.
    elem_t *val;   //Scalar array. It contains the block entries of the matrix.
    int *indx;  //Integer array. It contains the pointers to the beginning of each block entry stored in val.
    int *bindx; //Integer array. It contains the block column indices of block entries of the matrix.
    int *bptr;  //Integer array. It contains the pointers to the beginning of each block row in bindx and val.
} vbr;

void csr_vbr(csr *c, vbr *v, float thresh);

void vbr_csr(vbr *v, csr *c);

void vbr_SpMV(vbr *m, vector *v, vector *r);

void vbr_destroy(void *v);   // vbr might not be a pointer

#ifdef __cplusplus
}
#endif
#endif
