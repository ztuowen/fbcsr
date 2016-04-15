//
// Created by joe on 4/15/16.
//

#include"../fix_csr.h"
#include<time.h>

elem_t fix_csr_random(void) {
    return 1 + (elem_t) rand() / RAND_MAX;
}

void fix_csr(csr *c) {
    srand((unsigned int) time(NULL));
    for (int i = 0; i < c->nnz; ++i)
        c->val[i] = fix_csr_random();
}