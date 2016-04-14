//
// Created by joe on 4/10/16.
//
#include"../FBCSR.h"
#include"cuHelper.h"


void fbcsrSingle_memCpy(fbcsr *src, fbcsr *dst, enum DeviceCopyDIR dir) {
    dst->n = src->n;
    dst->m = src->m;
    assert(dst->c == src->c);
    assert(dst->r == src->r);
    dst->nnz = src->nnz;
    dst->nr = src->nr;
    dst->nb = src->nb;
    dst->nelem = src->nelem;
    dst->thresh = src->thresh;
    memCopy((void **) &(dst->rptr), (void *) src->rptr, sizeof(int) * src->nr, dir);
    memCopy((void **) &(dst->bptr), (void *) src->bptr, sizeof(int) * (src->nr + 1), dir);
    memCopy((void **) &(dst->val), (void *) src->val, sizeof(elem_t) * src->nb * src->nelem, dir);
    memCopy((void **) &(dst->bindx), (void *) src->bindx, sizeof(int) * (src->nb), dir);
}

extern "C" void fbcsr_memCpy(list *src, list *dst, enum DeviceCopyDIR dir) {
    while (src != NULL && dst != NULL) {
        fbcsrSingle_memCpy((fbcsr *) list_get(src), (fbcsr *) list_get(dst), dir);
        src = list_next(src);
        dst = list_next(dst);
    }
    assert(dst == src);
}

extern "C" void fbcsr_CUDA_SpMV(list *l, vector *v, vector *r) {
    fbcsr *f;
    while (l != NULL) {
        f = (fbcsr *) list_get(l);
        if (f->optKernel == NULL) {
            fprintf(stderr, "Cannot pass NULL as optkernel for fbcsr CUDA\n");
            exit(-1);
        } else {
            fbcsrSingle_SpMVKernel krnl = (fbcsrSingle_SpMVKernel) f->optKernel;
            if (f->nr > 0)
                krnl(f, v, r);
            cuCheck(cudaGetLastError());
        }
        l = list_next(l);
    }
}

extern "C" void fbcsr_CUDA_destroy(void *f) {
    fbcsr *ff = (fbcsr *) f;
    safeCudaFree(ff->rptr);
    safeCudaFree(ff->val);
    safeCudaFree(ff->bindx);
    safeCudaFree(ff->bptr);
}
