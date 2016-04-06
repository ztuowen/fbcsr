//
// Created by joe on 4/4/16.
//

#include"../UBCSR.h"
#include"cuHelper.h"

void ubcsrSingle_memCpy(ubcsr *src,ubcsr *dst,enum DeviceCopyDIR dir) {
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

extern "C" void ubcsr_memCpy(list *src,list *dst, enum DeviceCopyDIR dir){
    while (src != NULL && dst != NULL) {
        ubcsrSingle_memCpy((ubcsr *) list_get(src), (ubcsr *) list_get(dst), dir);
        src = list_next(src);
        dst = list_next(dst);
    }
    assert(dst == src);
}

void ubcsrSingle_CUDA_SpMV(ubcsr *u, vector *v, vector *r) {

}

extern "C" void ubcsr_CUDA_SpMV(list *l, vector *v, vector *r){

}

extern "C" void ubcsr_CUDA_destroy(void *u){
    ubcsr *uu = (ubcsr *) u;
    safeCudaFree(uu->rptr);
    safeCudaFree(uu->val);
    safeCudaFree(uu->bindx);
    safeCudaFree(uu->bptr);
}
