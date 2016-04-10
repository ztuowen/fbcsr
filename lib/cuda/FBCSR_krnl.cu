//
// Created by joe on 4/10/16.
//
#include"../FBCSR_krnl.h"

#define workGroup (blockSize/nelem)

template< int blockSize,
int nelem,
int rof,
int rowm,
int colm
>

__global__ void FBCSR_general_krnl(fbcsr *f, vector *v, vector *r) {
    int me = threadIdx.x / nelem;
    int row = blockIdx.x;
    int rr = f->rptr[row];
    int rc;
    int idx = blockDim.x % nelem;
    int nrow = rowm * nelem + rof;
    int ncol = colm * nelem;
    rr += nrow;
    elem_t acc = 0;
    for (int i = f->bptr[row] + me; i < f->bptr[row + 1]; i += workGroup) {
        rc = f->bindx[i] + ncol;
        acc += v->val[rc] * f->val[i * nelem + idx];
    }
    if (rowm == 0) { // there is no column difference
        __shared__
        elem_t red[blockSize];
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int tot = blockSize >> 1; threadIdx.x < tot; tot >>= 1) {
            red[threadIdx.x] += red[threadIdx.x + tot];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            r->val[rr] += red[0];
    } else { // column is increasing
        __shared__
        elem_t red[workGroup][nelem];
        red[me][idx] = acc;
        __syncthreads();
        for (int tot = workGroup >> 1; me < tot; tot >>= 1) {
            red[me][ncol] += red[me + tot][ncol];
            __syncthreads();
        }
        if (me == 0)
            r->val[rr] += red[0][ncol];
    }
};

extern "C" void fbcsr_row_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(256);
    FBCSR_general_krnl < 256, 16, 0, 0, 1 ><<<block, thread>>>(f, v, r);
}

extern "C" void fbcsr_row_krnl_32(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(256);
    FBCSR_general_krnl < 256, 32, 0, 0, 1 ><<<block, thread>>>(f, v, r);
}

extern "C" void fbcsr_col_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(256);
    FBCSR_general_krnl < 256, 16, 0, 1, 0 ><<<block, thread>>>(f, v, r);
}

extern "C" void fbcsr_col_krnl_32(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(256);
    FBCSR_general_krnl < 256, 32, 0, 1, 0 ><<<block, thread>>>(f, v, r);
}

extern "C" void fbcsr_fslash_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(256);
    FBCSR_general_krnl < 256, 16, 15, -1, 1 ><<<block, thread>>>(f, v, r);
}

extern "C" void fbcsr_fslash_krnl_32(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(256);
    FBCSR_general_krnl < 256, 32, 31, -1, 1 ><<<block, thread>>>(f, v, r);
}

extern "C" void fbcsr_bslash_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(256);
    FBCSR_general_krnl < 256, 16, 0, 1, 1 ><<<block, thread>>>(f, v, r);
}

extern "C" void fbcsr_bslash_krnl_32(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(256);
    FBCSR_general_krnl < 256, 32, 0, 1, 1 ><<<block, thread>>>(f, v, r);
}

