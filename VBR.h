#ifndef _VBR_H
#define _VBR_H

#include"CSR.h"
#include"prefix.h"

typedef struct vbr {
    int n,m;
    int nr;
    int *rptr;  //Integer array. It contains the block row partitioning information--that is, the first row number of each block row.
    int *cptr;  //Integer array. It contains the block column partitioning information--that is, the first column number of each block column.
    elem_t *val;   //Scalar array. It contains the block entries of the matrix.
    int *indx;  //Integer array. It contains the pointers to the beginning of each block entry stored in val.
    int *bindx; //Integer array. It contains the block column indices of block entries of the matrix.
    int *bptr;  //Integer array. It contains the pointers to the beginning of each block row in bindx and val.
} vbr;

int csr_vbr(csr *c,vbr *v,float thresh);

#endif
