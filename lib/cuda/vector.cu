//
// Created by joe on 4/5/16.
//

#include"../vector.h"
#include"cuHelper.h"

extern "C" void vector_memCpy(vector *src, vector *dst, enum DeviceCopyDIR dir) {
    dst->n = src->n;
    memCopy((void **) &(dst->val), (void *) src->val, sizeof(elem_t) * dst->n, dir);
}

extern "C" void vector_CUDA_destroy(void *v) {
    vector *vv = (vector *) v;
    safeCudaFree(vv->val);
}
