//
// Created by joe on 4/1/16.
//

#include "../vector_gen.h"
#include<time.h>

elem_t vector_gen_random(void) {
    return 1 + (elem_t) rand() / RAND_MAX;
}

void vector_gen(vector *v, int n, elem_t (*random)(void)) {
    if (random == NULL)
        srand((unsigned int) time(NULL));
    vector_init(v, n);
    for (int i = 0; i < n; ++i)
        if (random != NULL)
            v->val[i] = random();
        else
            v->val[i] = vector_gen_random();
}