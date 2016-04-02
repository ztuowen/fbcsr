//
// Created by joe on 4/1/16.
//

#include "../vector_gen.h"
#include<time.h>

#define DEFAULT_RNG 30

elem_t default_random(void) {
#ifdef USE_FLOAT
    return (elem_t) rand() / RAND_MAX;
#else
    return rand()%DEFAULT_RNG;
#endif
}

void vector_gen_random(vector *v, int n, elem_t (*random)(void)) {
    if (random == NULL)
        srand((unsigned int) time(NULL));
    vector_init(v, n);
    for (int i = 0; i < n; ++i)
        if (random != NULL)
            v->val[i] = random();
        else
            v->val[i] = default_random();
}