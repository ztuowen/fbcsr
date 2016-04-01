#include "CSR.h"
#include"string.h"
#include"assert.h"

void csr_destroy(void *c) {
    csr *cc = (csr *) c;
    safeFree(cc->val);
    safeFree(cc->indx);
    safeFree(cc->ptr);
}

void csr_makeEmpty(csr *c, int n, int m) {
    c->n = n;
    c->m = m;
    c->val = NULL;
    c->indx = NULL;
    c->ptr = malloc((n + 1) * sizeof(int));
    memset(c->ptr, 0, (n + 1) * sizeof(int));
}

void csr_merge(csr *a, csr *b) {
    assert(a->n == b->n);
    assert(a->m == b->m);
    int i, aj, bj;
    int cnt = a->ptr[a->n] + b->ptr[b->n];
    elem_t *val = malloc(cnt * sizeof(elem_t));
    int *indx = malloc(cnt * sizeof(int));
    int *ptr = malloc((a->n + 1) * sizeof(int));
    cnt = 0;
    for (i = 0; i < a->n; ++i) {
        ptr[i] = cnt;
        aj = a->ptr[i];
        bj = b->ptr[i];
        while (aj < a->ptr[i + 1] && bj < b->ptr[i + 1]) {
            if (a->indx[aj] < b->indx[bj]) {
                indx[cnt] = a->indx[aj];
                val[cnt] = a->val[aj];
                ++aj;
            } else {
                assert(a->indx[aj] > b->indx[bj]);
                indx[cnt] = b->indx[bj];
                val[cnt] = b->val[bj];
                ++bj;
            }
            ++cnt;
        }
        while (aj < a->ptr[i + 1]) {
            indx[cnt] = a->indx[aj];
            val[cnt] = a->val[aj];
            ++aj;
            ++cnt;
        }
        while (bj < b->ptr[i + 1]) {
            indx[cnt] = b->indx[bj];
            val[cnt] = b->val[bj];
            ++bj;
            ++cnt;
        }
    }
    ptr[a->n] = cnt;
    csr_destroy(a);
    a->ptr = ptr;
    a->indx = indx;
    a->val = val;
}

void csr_SpMV(csr *m, vector *v, vector *r) {
    assert(m->m == v->n);
    assert(m->n == r->n);
    int i, j;
    for (i = 0; i < m->n; ++i)
        for (j = m->ptr[i]; j < m->ptr[i + 1]; ++j)
            r->val[i] += v->val[m->indx[j]] * m->val[j];
}
