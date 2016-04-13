//
// Created by joe on 4/8/16.
//

#include"../FBCSR.h"
#include<string.h>

#define MAXCOL 1048576

void fbcsr_makeEmpty(fbcsr *f, int n, int m, int c, int r, int nelem, void *optKrnl, void *getCoo) {
    f->n = n;
    f->m = m;
    f->c = c;
    f->r = r;
    f->optKernel = optKrnl;
    f->nelem = nelem;
    f->optKernel = optKrnl;
    f->getCoo = getCoo;
}

coo fbcsr_row(int elemIndx, int elemCnt) {
    coo c;
    c.c = elemIndx;
    c.r = 0;
    return c;
}

coo fbcsr_column(int elemIndx, int elemCnt) {
    coo c;
    c.c = 0;
    c.r = elemIndx;
    return c;
}

coo fbcsr_forwardSlash(int elemIndx, int elemCnt) {
    coo c;
    c.c = elemIndx;
    c.r = elemCnt - elemIndx - 1;
    return c;
}

coo fbcsr_backwardSlash(int elemIndx, int elemCnt) {
    coo c;
    c.c = elemIndx;
    c.r = elemIndx;
    return c;
}

int *bestseq(int *score, int *ids, int n, int maxn, int con, int *tot) {
    int i;
    int *sel;
    int nxt;
    sel = malloc((n+1)*sizeof(int));
    memset(sel,0,(n+1)*sizeof(int));
    nxt = 0;
    ids[n] = maxn;
    for (i=0;i<n;++i) {
        while (nxt <= n && ids[nxt] < ids[i] + con)
            ++nxt;
        if (nxt <= n)
            sel[nxt] = max(sel[nxt], sel[i] + score[i]);
        sel[i+1] = max(sel[i+1],sel[i]);
    }
    i = n;
    int sc = sel[n];
    int li = n;
    nxt = n - 1;
    int lstp = maxn - con;
    *tot=0;
    while (nxt >= 0) {
        while (nxt > 0 && ids[nxt] > lstp)
            --nxt;
        if (ids[nxt] <= lstp && score[nxt] > 0 && sel[nxt] + score[nxt] == sc) {
            sc = sel[nxt];
            while (li > nxt)
                sel[--li] = 0;
            sel[nxt] = 1;
            lstp = ids[nxt] - con;
            ++(*tot);
        } else
            --nxt;
    }
    while (li>0)
        sel[--li]=0;
    return sel;
}

csr *fbcsrSingle_csr(fbcsr *f) {
    csr *c = malloc(sizeof(csr));
    c->n = f->n;
    c->m = f->m;
    fbcsr_getCoo getCoo = (fbcsr_getCoo) f->getCoo;
    int cnt = f->nb * f->nelem;
    int i, vcnt, j, k;
    // Count
    vcnt = 0;
    for (i = 0; i < cnt; ++i)
        if (f->val[i] != 0)
            ++vcnt;
    c->nnz = vcnt;
    // Malloc
    c->val = malloc(vcnt * sizeof(elem_t));
    c->indx = malloc(vcnt * sizeof(int));
    c->ptr = malloc((f->n + 1) * sizeof(int));

    // Fill
    int row = 0;
    cnt = 0;
    for (i = 0; i < f->nr; ++i) {
        int offset = 0;
        int offsetend = 0;
        for (j = 0; j < f->r && f->rptr[i] + j < f->n; ++j) // row
        {
            offset = offsetend = 0;
            while (offset < f->nelem && getCoo(offset, f->nelem).r != j)
                ++offset;
            offsetend = offset;
            while (offsetend < f->nelem && getCoo(offsetend, f->nelem).r == j)
                ++offsetend;
            int col;
            // write to ptr
            while (row <= j + f->rptr[i]) {
                c->ptr[row] = cnt;
                ++row;
            }
            // write to indx & val
            for (k = f->bptr[i]; k < f->bptr[i + 1]; ++k) {
                for (col = offset; col < offsetend; ++col)
                    if (f->val[k * f->nelem + col] != 0) {
                        c->val[cnt] = f->val[k * f->nelem + col];
                        c->indx[cnt] = f->bindx[k] + getCoo(col, f->nelem).c;
                        ++cnt;
                    }
            }
        }
    }
    while (row <= c->n) {
        c->ptr[row] = cnt;
        ++row;
    }
    return c;
}

void fbcsr_csr(list *l, csr *rem, csr *c) {
    fbcsr *f;
    csr *tmp;
    csr_makeEmpty(c, rem->n, rem->m);
    csr_merge(c, rem);
    while (l != NULL) {
        f = (fbcsr *) list_get(l);

        tmp = fbcsrSingle_csr(f);
        csr_merge(c, tmp);

        // clean up & next
        csr_destroy(tmp);
        free(tmp);
        l = list_next(l);
    }
}

void fbcsr_destroy(void *f) {
    fbcsr *ff = (fbcsr *) f;
    safeFree(ff->rptr);
    safeFree(ff->val);
    safeFree(ff->bindx);
    safeFree(ff->bptr);
}

void fbcsrSingle_SpMV(fbcsr *f, vector *v, vector *r) {
    int i, j, k;
    elem_t *val = f->val;
    fbcsr_getCoo getCoo = (fbcsr_getCoo) f->getCoo;
    assert(f->m == v->n);
    assert(f->n == r->n);
    for (i = 0; i < f->nr; ++i)
        for (j = f->bptr[i]; j < f->bptr[i + 1]; ++j) {
            for (k = 0; k < f->nelem; ++k, ++val) {
                coo c = getCoo(k, f->nelem);
                c.r += f->rptr[i];
                c.c += f->bindx[j];
                //if (c.r < f->n && c.c < f->m)
                r->val[c.r] += v->val[c.c] * (*val);
            }
        }
}

void fbcsr_SpMV(list *l, vector *v, vector *r) {
    fbcsr *f;
    while (l != NULL) {
        f = (fbcsr *) list_get(l);
        if (f->optKernel == NULL)
            fbcsrSingle_SpMV(f, v, r);
        else {
            fbcsrSingle_SpMVKernel krnl = (fbcsrSingle_SpMVKernel) f->optKernel;
            krnl(f, v, r);
        }
        l = list_next(l);
    }
}

int csr_lookFor(csr *c, coo pos, int *last, int *mincol, int colcor) {
    if (pos.r >= c->n)
        return 0;
    int rst = c->ptr[pos.r], red = c->ptr[pos.r + 1];
    if ((*last) < rst)
        *last = rst;
    if ((*last) >= red)
        *last = red - 1;
    while (*last > rst && c->indx[*last] > pos.c)
        --(*last);
    while (*last < red && c->indx[*last] < pos.c)
        ++(*last); // last >= pos.c
    if (*last >= rst && *last < red) {
        if (c->indx[*last] == pos.c)
            return 1;
        if (mincol != NULL)
            *mincol = min(*mincol, c->indx[*last] - colcor);
    }
    return 0;
}

csr *fbcsr_csr_splitOnce(csr *c, fbcsr *f, float thresh) {
    fbcsr_getCoo getCoo = (fbcsr_getCoo) f->getCoo;
    int row, col, idx, vcnt, cnt;
    int *findidx;
    int *minidx;
    int *sel;
    int tot;
    int *rowsc = malloc(f->m*sizeof(int));
    int *colsc = malloc(f->n*sizeof(int));
    int *ids = malloc((max(f->m, f->n) + 1) * sizeof(int));
    int *rowbk = malloc(f->n * sizeof(int));
    csr *last = malloc(sizeof(csr));
    csr_makeEmpty(last, c->n, c->m);
    csr_merge(last, c);
    f->n = c->n;
    f->m = c->m;

    findidx = malloc(f->r * sizeof(int));
    minidx = malloc(f->r * sizeof(int));
    for (idx = 0; idx < f->r; ++idx)
        minidx[idx] = MAXCOL;
    for (idx = 0; idx < f->nelem; ++idx) {
        coo pos = getCoo(idx, f->nelem);
        minidx[pos.r] = min(minidx[pos.r], pos.c);
    }

    // First we will inspect and get the total number of element registered
    int mincol;
    memset(rowsc,0,f->m*sizeof(int));
    for (row = 0; row < c->n; ++row) {
        // Here we use the simpler form that put the original kernel into the part
        // and assume that there is no contention in cols(we simply don't use row)
        memset(colsc,0,f->n*sizeof(int));
        memset(findidx, 0, f->r * sizeof(int));
        int selc = 0;
        for (col = 0; col < c->m;) {
            cnt = 0;
            mincol = MAXCOL;
            for (idx = 0; idx < f->nelem; ++idx) {
                coo pos = getCoo(idx, f->nelem);
                pos.r += row;
                pos.c += col;
                if (csr_lookFor(c, pos, &findidx[pos.r - row], &mincol, minidx[pos.r - row]))
                    ++cnt;
            }
            if (cnt >= thresh * f->nelem) {
                colsc[selc] = cnt;
                ids[selc++] = col;
            }
            col = mincol;
        }
        sel = bestseq(colsc, ids, selc, f->m, f->c, &tot);
        rowsc[row] = sel[selc];
        rowbk[row] = tot;
        free(sel);
    }

    // Get the best row decomposition
    for (row = 0; row < f->n; ++row)
        ids[row] = row;
    sel = bestseq(rowsc, ids, c->n, f->n, f->r, &tot);
    int r;
    if (0.1 * c->nnz > sel[c->n]) {
        f->nr = 0;
        f->nb = 0;
        f->rptr = malloc((f->nr) * sizeof(int));
    } else {
        f->nr = tot;
        f->nb = 0;
        for (row = 0; row < f->n; ++row)
            if (sel[row])
                f->nb += rowbk[row];
        f->rptr = malloc((f->nr) * sizeof(int));
        r = 0;
        for (row = 0; row < f->n; ++row)
            if (sel[row] == 1)
                f->rptr[r++] = row;
    }
    // Now we have how many columns and how many rows that it has.
    f->val = malloc(f->nelem * f->nb * sizeof(elem_t));
    f->bindx = malloc(f->nb * sizeof(int));
    f->bptr = malloc((f->nr + 1) * sizeof(int));
    f->bptr[0] = 0;
    vcnt = 0;
    for (r = 0; r < f->nr; ++r) {
        row = f->rptr[r];
        memset(findidx, 0, f->r * sizeof(int));
        memset(colsc, 0, f->n * sizeof(int));
        int selc = 0;
        for (col = 0; col < c->m;) {
            cnt = 0;
            mincol = MAXCOL;
            for (idx = 0; idx < f->nelem; ++idx) {
                coo pos = getCoo(idx, f->nelem);
                pos.r += row;
                pos.c += col;
                if (csr_lookFor(c, pos, &findidx[pos.r - row], &mincol, minidx[pos.r - row]))
                    ++cnt;
            }
            if (cnt >= thresh * f->nelem) {
                colsc[selc] = cnt;
                ids[selc++] = col;
            }
            col = mincol;
        }
        sel = bestseq(colsc, ids, selc, f->m, f->c, &tot);
        int cc;
        for (cc = 0; cc < selc; ++cc)
            if (sel[cc] == 1) {
                col = ids[cc];
                cnt = vcnt * f->nelem;
                f->bindx[vcnt] = col;
                for (idx = 0; idx < f->nelem; ++idx) {
                    coo pos = getCoo(idx, f->nelem);
                    pos.r += row;
                    pos.c += col;
                    if (csr_lookFor(c, pos, &findidx[pos.r - row], NULL, 0)) {
                        last->val[findidx[pos.r - row]] = 0;
                        f->val[cnt + idx] = c->val[findidx[pos.r - row]];
                    } else
                        f->val[cnt + idx] = 0;
                }
                ++vcnt;
            }
        f->bptr[r + 1] = vcnt;
        free(sel);
    }
    DEBUG_PRINT("Convert FBCSR: %d\n", vcnt * f->nelem);
    vcnt = 0;
    for (row = 0; row < last->n; ++row) {
        col = last->ptr[row];
        last->ptr[row] = vcnt;
        for (; col < last->ptr[row + 1]; ++col)
            if (last->val[col] != 0) {
                last->val[vcnt] = last->val[col];
                last->indx[vcnt++] = last->indx[col];
            }
    }
    last->ptr[last->n] = vcnt;
    last->nnz = vcnt;
    f->nnz = c->nnz - last->nnz;
    free(rowsc);
    free(colsc);
    free(ids);
    free(rowbk);
    return last;
}

csr *csr_fbcsr(csr *c, list *l, float thresh) {
    fbcsr *f;
    csr *r;
    csr *last = NULL;

    if (l != NULL) {
        f = (fbcsr *) list_get(l);
        last = fbcsr_csr_splitOnce(c, f, thresh);
        l = list_next(l);
        while (l != NULL) {
            f = (fbcsr *) list_get(l);

            r = fbcsr_csr_splitOnce(last, f, thresh);

            // setup for next iter
            l = list_next(l);
            csr_destroy(last);
            free(last);
            last = r;
        }
    }

    return last;
}
