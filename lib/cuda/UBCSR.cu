//
// Created by joe on 4/4/16.
//

#include"../UBCSR.h"
#include"cuHelper.h"

void ubcsrSingle_memCpy(ubcsr *src, ubcsr *dst, enum DeviceCopyDIR dir) {
    dst->n = src->n;
    dst->m = src->m;
    assert(dst->c == src->c);
    assert(dst->r == src->r);
    dst->nnz = src->nnz;
    dst->nr = src->nr;
    dst->nb = src->nb;
    memCopy((void **) &(dst->rptr), (void *) src->rptr, sizeof(int) * src->nr, dir);
    memCopy((void **) &(dst->bptr), (void *) src->bptr, sizeof(int) * (src->nr + 1), dir);
    memCopy((void **) &(dst->val), (void *) src->val, sizeof(elem_t) * src->nb * src->c * src->r, dir);
    memCopy((void **) &(dst->bindx), (void *) src->bindx, sizeof(int) * (src->nb), dir);
}

extern "C" void ubcsr_memCpy(list *src, list *dst, enum DeviceCopyDIR dir) {
    while (src != NULL && dst != NULL) {
        ubcsrSingle_memCpy((ubcsr *) list_get(src), (ubcsr *) list_get(dst), dir);
        src = list_next(src);
        dst = list_next(dst);
    }
    assert(dst == src);
}

/*  CPU Reference
 *   int i, j, k, l;
 *   int indx = 0;
 *   assert(u->m == v->n);
 *   assert(u->n == r->n);
 *   for (i = 0; i < u->nr; ++i) // BLOCK
 *       for (j = u->bptr[i]; j < u->bptr[i + 1]; ++j) { // UBCSR
 *           for (k = 0; k < u->r; ++k)
 *               for (l = 0; l < u->c; ++l, ++indx)
 *                   r->val[k + u->rptr[i]] += v->val[l + u->bindx[j]] * u->val[indx];
 *       }
 */
__global__ void ubcsrSingle_CUDA_SpMV_krnl(ubcsr u, vector v, vector r) {

    __shared__
    elem_t val[32];
    int row_offset = u.rptr[blockIdx.x];
    int rowst = u.bptr[blockIdx.x] + threadIdx.x;
    int rowed = u.bptr[blockIdx.x + 1];
    int indxoffset = 0;
    int cr = u.c * u.r;
    for (int k = 0; k < u.r; ++k) {
        elem_t tmp = 0;
        for (int j = rowst; j < rowed; j += blockDim.x) {
            int col = u.bindx[j];
            int indx = j * cr;
            for (int l = 0; l < u.c; ++l, ++indxoffset)
                tmp += v.val[l + col] * u.val[indx + indxoffset];
        }
        val[threadIdx.x] = tmp;
        for (int top = blockDim.x >> 1; top > 0; top >>= 1)
            if (threadIdx.x < top)
                val[threadIdx.x] += val[threadIdx.x + top];
        if (threadIdx.x == 0)
            r.val[row_offset + k] += val[0];
    }
}

void ubcsrSingle_CUDA_SpMV(ubcsr *u, vector *v, vector *r) {
    dim3 grid(u->nr), block(32);
    ubcsrSingle_CUDA_SpMV_krnl << < grid, block >> > (*u, *v, *r);
}

extern "C" void ubcsr_CUDA_SpMV(list *l, vector *v, vector *r) {
    ubcsr *u;
    while (l != NULL) {
        u = (ubcsr *) list_get(l);
        if (u->optKernel == NULL)
            ubcsrSingle_CUDA_SpMV(u, v, r);
        else {
            ubcsrSingle_SpMVKernel krnl = (ubcsrSingle_SpMVKernel) u->optKernel;
            krnl(u, v, r);
        }
        l = list_next(l);
    }
}

extern "C" void ubcsr_CUDA_destroy(void *u) {
    ubcsr *uu = (ubcsr *) u;
    safeCudaFree(uu->rptr);
    safeCudaFree(uu->val);
    safeCudaFree(uu->bindx);
    safeCudaFree(uu->bptr);
}
