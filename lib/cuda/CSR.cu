//
// Created by joe on 4/5/16.
//

#include"../CSR.h"
#include"cuHelper.h"
#include<cusparse.h>

extern "C" void csr_memCpy(csr *src, csr *dst, enum DeviceCopyDIR dir) {
    dst->m = src->m;
    dst->n = src->n;
    dst->nnz = src->nnz;
    memCopy((void **) &(dst->ptr), (void *) src->ptr, sizeof(int) * (dst->n + 1), dir);
    memCopy((void **) &(dst->indx), (void *) src->indx, sizeof(int) * (src->ptr[src->n]), dir);
    memCopy((void **) &(dst->val), (void *) src->val, sizeof(elem_t) * (src->ptr[src->n]), dir);
}

extern "C" void csr_CUDA_SpMV(csr *m, vector *v, vector *r) {
    cusparseMatDescr_t descr = 0;
    cuSparseCheck(cusparseCreateMatDescr(&descr));
    cuSparseCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    cuSparseCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    cusparseHandle_t handle;
    cuSparseCheck(cusparseCreate(&handle));
    elem_t unit = 1;
    cuSparseCheck(
            cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m->n, m->m, m->nnz, &unit, descr, m->val, m->ptr,
                           m->indx, v->val, &unit, r->val));
    cuSparseCheck(cusparseDestroy(handle));
    cuSparseCheck(cusparseDestroyMatDescr(descr));
}

extern "C" void csr_CUDA_destroy(void *c) {
    csr *cc = (csr *) c;
    safeCudaFree(cc->val);
    safeCudaFree(cc->indx);
    safeCudaFree(cc->ptr);
}
