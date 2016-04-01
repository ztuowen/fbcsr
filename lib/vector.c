#include "vector.h"
#include<assert.h>
#include<string.h>
#include<math.h>

int vector_equal(vector *a, vector *b) {
    assert(a->n == b->n);
    int i;
    for (i = 0; i < a->n; ++i)
#ifdef USE_FLOAT
        if (fabs(a->val[i] - b->val[i])>_FLOAT_PREC)
            return 0;
#else
            if (a->val[i] != b->val[i])
                return 0;
#endif
    return 1;
}

int vector_init(vector *v, int n) {
    v->n = n;
    v->val = malloc(n * sizeof(elem_t));
    for (int i = 0; i < n; ++i)
        v->val[i] = 0;
}

void vector_destroy(void *v) {
    vector *vv = (vector *) v;
    safeFree(vv->val);
}