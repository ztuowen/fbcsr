#include "UBCSR.h"
#include "VBR.h"
#include<assert.h>

csr *csr_splitOnce(csr *c, ubcsr *u, float thresh) {
    vbr *v = malloc(sizeof(vbr));
    csr *rem = malloc(sizeof(csr));
    int i, j, k, lk;
    int rcnt, bcnt;
    csr_vbr(c, v, thresh);

    // Converting from VBR to UBCSR
    u->n = v->n;
    u->m = v->m;

    // Counting
    rcnt = 0; // row count
    bcnt = 0; // block count

    for (i = 0; i < v->nr; ++i) {
        for (j = 0; j + u->r <= v->rptr[i + 1] - v->rptr[i]; j += u->r) //row
        {
            ++rcnt;
            for (k = v->bptr[i]; k < v->bptr[i + 1]; ++k) {
                int rowsize = (v->cptr[v->bindx[k] + 1] - v->cptr[v->bindx[k]]);
                lk = 0;
                while (lk + u->c <= rowsize) {
                    int index, nz = 1;
                    index = v->indx[k] + lk + j * rowsize;
                    for (int cr = 0; cr < u->r && (nz); ++cr, index += (rowsize - u->c))
                        for (int cc = 0; cc < u->c && (nz); ++cc, ++index)
                            if (v->val[index] > 0) { // Encountered a nonzero adding
                                ++bcnt;
                                nz = 0; // just to break, how i wish to use goto
                            }
                    lk += u->c;
                }
            }
        }
    }

    // mallocs
    u->bindx = malloc(bcnt * sizeof(int));
    u->val = malloc(bcnt * u->c * u->r * sizeof(elem_t));
    u->bptr = malloc((rcnt + 1) * sizeof(int));
    u->rptr = malloc(rcnt * sizeof(int));
    u->nr = rcnt;
    bcnt = 0;
    rcnt = 0;
    // Now, add everything!
    for (i = 0; i < v->nr; ++i) {
        for (j = 0; j + u->r <= v->rptr[i + 1] - v->rptr[i]; j += u->r) //row
        {
            u->bptr[u->nr] = bcnt;
            u->rptr[(u->nr)++] = j + v->rptr[i];
            for (k = v->bptr[i]; k < v->bptr[i + 1]; ++k) {
                int rowsize = (v->cptr[v->bindx[k] + 1] - v->cptr[v->bindx[k]]);
                lk = 0;
                while (lk + u->c <= rowsize) {
                    int index, nz = 1;
                    index = v->indx[k] + lk + j * rowsize;
                    for (int cr = 0; cr < u->r && (nz); ++cr, index += (rowsize - u->c))
                        for (int cc = 0; cc < u->c && (nz); ++cc, ++index)
                            if (v->val[index] > 0) { // Encountered a nonzero adding
                                nz = 0; // just to break, how i wish to use goto
                            }
                    if (nz == 0) {
                        int st = bcnt * u->c * u->r;
                        index = v->indx[k] + lk + j * rowsize;
                        u->bindx[bcnt] = v->cptr[v->bindx[k]] + lk;
                        for (int cr = 0; cr < u->r; ++cr, index += (rowsize - u->c), ++st)
                            for (int cc = 0; cc < u->c; ++cc, ++index, ++st) {
                                u->val[st] = v->val[index];
                                v->val[index] = 0;
                            }
                        ++bcnt;
                    }
                    lk += u->c;
                }
            }
        }
    }
    assert(u->nr == rcnt);
    u->bptr[rcnt] = bcnt;
    // Get the remainder
    vbr_csr(v, rem);
    return rem;
}

csr *csr_ubcsr(csr *c, list *l, float thresh) {
    ubcsr *u;
    csr *r;
    csr *last = NULL;

    if (l != NULL) {
        u = (ubcsr *) list_get(l);
        last = csr_splitOnce(c, u, thresh);
        l = list_next(l);
        while (l != NULL) {
            u = (ubcsr *) list_get(l);

            r = csr_splitOnce(last, u, thresh);

            // setup for next iter
            l = list_next(l);
            csr_destroy(last);
            free(last);
            last = r;
        }
    }

    return last;
}

csr *ubcsrSingle_csr(ubcsr *u) {
    csr *c = malloc(sizeof(csr));
    c->n = u->n;
    c->m = u->m;
    int cnt = u->bptr[u->nr] * u->c * u->r;
    int i, vcnt, j, k;
    // Count
    vcnt = 0;
    for (i = 0; i < cnt; ++i)
        if (u->val[i] != 0)
            ++vcnt;
    // Malloc
    c->val = malloc(vcnt * sizeof(elem_t));
    c->indx = malloc(vcnt * sizeof(int));

    // Fill
    int row = 0;
    for (i = 0; i < u->nr; ++i) {
        for (j = 0; j < u->r; ++j) // row
        {
            int offset = j * u->c;
            int col;
            // write to ptr
            while (row < j + u->rptr[i]) {
                c->ptr[row] = cnt;
                ++row;
            }
            // write to indx & val
            for (k = u->bptr[i]; k < u->bptr[i + 1]; ++k) {
                for (col = 0; col < u->c; ++col)
                    if (u->val[u->c * u->r * k + offset + col] != 0) {
                        c->val[cnt] = u->val[u->c * u->r * k + offset + col];
                        c->indx[cnt] = u->bindx[k] + col;
                        ++cnt;
                    }
            }
        }
    }
    while (row < c->n) {
        c->ptr[row] = cnt;
        ++row;
    }
    c->ptr[row] = cnt;
    return c;
}

void ubcsr_csr(list *l, csr *rem, csr *c) {
    ubcsr *u;
    csr *tmp;
    csr_makeEmpty(c, rem->n, rem->m);
    csr_merge(c, rem);
    while (l != NULL) {
        u = (ubcsr *) list_get(l);

        tmp = ubcsrSingle_csr(u);
        csr_merge(c, tmp);

        // clean up & next
        csr_destroy(tmp);
        free(tmp);
        l = list_next(l);
    }
}

void ubcsr_destroy(void *u) {
    ubcsr *uu = (ubcsr *) u;
    safeFree(uu->rptr);
    safeFree(uu->val);
    safeFree(uu->bindx);
    safeFree(uu->bptr);
}

void ubcsrSingle_SpMV(ubcsr *u, vector *v, vector *r) {
    int i, j, k, l;
    int indx = 0;
    assert(u->m == v->n);
    assert(u->n == r->n);
    for (i = 0; i < u->nr; ++i)
        for (j = u->bptr[i]; j < u->bptr[i + 1]; ++j) {
            for (k = 0; k < u->r; ++k, ++indx)
                for (l = 0; l < u->c; ++l, ++indx)
                    r->val[k + u->rptr[i]] += v->val[l + u->bindx[j]] * u->val[indx];
        }
}

void ubcsr_SpMV(list *l, csr *rem, vector *v, vector *r) {
    ubcsr *u;
    csr_SpMV(rem, v, r);
    while (l != NULL) {
        u = (ubcsr *) list_get(l);
        if (u->optKernel == NULL)
            ubcsrSingle_SpMV(u, v, r);
        else {
            ubcsrSingle_SpMVKernel krnl = (ubcsrSingle_SpMVKernel) u->optKernel;
            krnl(u, v, r);
        }
        l = list_next(l);
    }
}

void ubcsr_makeEmpty(ubcsr *u,int n, int m, int c, int r, void *optKrnl) {
    u->n = n;
    u->m = m;
    u->c = c;
    u->r = r;
    u->optKernel = optKrnl;
}
