#ifndef _CSR_H
#define _CSR_H

#include "prefix.h"

typedef struct csr {
    int n,m;
    elem_t *ptr,*indx,*val;
} csr;

#endif
