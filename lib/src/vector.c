#include "../vector.h"
#include<math.h>

int floatEqual(elem_t a, elem_t b) {
    elem_t absA = fabs(a);
    elem_t absB = fabs(b);
    elem_t diff = fabs(a - b);

    if (a == b) { // shortcut, handles infinities
        return 1;
    } else if ((absA + absB < FLOAT_PREC) && diff < FLOAT_PREC)
        return 1;
    else if (diff / (absA + absB) < FLOAT_PREC)
        return 1;
    return 0;
}

int vector_equal(vector *a, vector *b) {
    assert(a->n == b->n);
    int i;
    for (i = 0; i < a->n; ++i)
        if (!floatEqual(a->val[i], b->val[i])) {
            fprintf(stderr, "Not Equal: %f %f at %d", a->val[i], b->val[i], i);
            return 0;
        }
    return 1;
}

void vector_init(vector *v, int n) {
    v->n = n;
    v->val = malloc(n * sizeof(elem_t));
    for (int i = 0; i < n; ++i)
        v->val[i] = 0;
}

void vector_destroy(void *v) {
    vector *vv = (vector *) v;
    safeFree(vv->val);
}
