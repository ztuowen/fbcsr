#ifndef _UBCSR_H
#define _UBCSR_H

#include "prefix.h"
#include "CSR.h"
#include "list.h"

typedef struct ubcsr {
    int n,m;
    int c,r;
    int nr;
    int *rptr;      // Integer array, storing the row partitioning information
    elem_t *val;    // Scalar array, storing the values
    int *bindx;      // Integer array, storing the column indices of blocks
    int *bptr;      // Integer array; storing the pointer to the beginning of each block row
} ubcsr;

csr* csr_ubcsr(csr *c,list *l,float thresh);    // c,r information embedded in list of ubcsr
                                                // Easier to just return the remainder, TODO it is against convention
void ubcsr_csr(list *l, csr *rem,csr *c);
void ubcsr_destroy(void* u);

#endif
