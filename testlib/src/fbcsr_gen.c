//
// Created by joe on 4/18/16.
//

#include"../fbcsr_gen.h"
#include<time.h>

elem_t fbcsr_gen_random(void) {
    return 1 + (elem_t) rand() / RAND_MAX;
}

void fbcsr_gen(fbcsr *f, int nr, int avg) {
    srand((unsigned int) time(NULL));
    f->nb = avg * nr;
    f->val = malloc(sizeof(elem_t) * f->nb * f->nelem);
    f->nr = nr;
    f->rptr = malloc(sizeof(int) * nr);
    f->bptr = malloc(sizeof(int) * (nr + 1));
    f->bindx = malloc(sizeof(int) * f->nb);
    for (int i = 0; i < f->nb * f->nelem; ++i)
        f->val[i] = fbcsr_gen_random();
    f->bptr[0] = 0;
    for (int i = 0; i < f->nr; ++i) {
        f->bptr[i + 1] = (i + 1) * avg;
        f->rptr[i] = i * f->r;
        for (int j = 0; j < avg; ++j)
            f->bindx[i * avg + j] = j * f->c;
    }
}