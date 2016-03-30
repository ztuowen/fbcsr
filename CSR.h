#ifndef _CSR_H
#define _CSR_H

#include "prefix.h"

typedef struct csr {
    int n,m;
    int *ptr;       // Guarded
    int *indx;
    elem_t *val;
} csr;

#endif
