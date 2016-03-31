#ifndef _UBCSR_H
#define _UBCSR_H

#include "prefix.h"
#include "CSR.h"
#include "list.h"

typedef struct ubcsr {
    int n,m;
    int c,r;
    elem_t *val;    // Scalar array, storing the values
    int *indx;      // Integer array, storing the column indices of blocks
    int *rptr;      // Integer array, storing the row partitioning information
    int *bptr;      // Integer array; storing the pointer to the beginning of each block row
} ubcsr;

// Easier to just return the remainder
csr* csr_splitOnce(csr *c,ubcsr *u,csr *rem,float thresh);
csr* csr_split(csr *c,list *l,float thresh);    // c,r information embedded in list of ubcsr
void ubcsr_destroy(void* u);

#endif
