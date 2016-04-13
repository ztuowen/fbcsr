//
// Created by joe on 4/10/16.
//
#include"../FBCSR_krnl.h"

#define workGroup (blockSize/nelem)

template< int step
>

__device__ inline void linereduce(elem_t *x, int idx) {
    if (idx < step) {
        x[idx] += x[idx + step];
        if (step > 32)
            __syncthreads();
    }
}

template< int blockSize,
int nelem,
int rof,
int rowm,
int colm,
int rowwork
>

__global__ void FBCSR_general_krnl(fbcsr f, vector v, vector r) {
    int me = threadIdx.x / nelem;
    int row = blockIdx.x;
    while (row < f.nr) {
        int rr = f.rptr[row];
        int rc;
        int idx = threadIdx.x % nelem;
        int nrow = rowm * idx + rof;
        int ncol = colm * idx;
        rr += nrow;
        elem_t acc = 0;
        for (int i = f.bptr[row] + me; i < f.bptr[row + 1]; i += workGroup) {
            rc = f.bindx[i] + ncol;
            acc += v.val[rc] * f.val[i * nelem + idx];
        }
        if (rowm == 0) { // there is no row difference
            __shared__
            elem_t red[blockSize];
            red[threadIdx.x] = acc;
            if (blockSize > 32)
            __syncthreads();
            if (blockSize > 128)
                linereduce < 128 > (red, threadIdx.x);
            if (blockSize > 64)
                linereduce < 64 > (red, threadIdx.x);
            if (blockSize > 32)
                linereduce < 32 > (red, threadIdx.x);
            linereduce < 16 > (red, threadIdx.x);
            linereduce < 8 > (red, threadIdx.x);
            linereduce < 4 > (red, threadIdx.x);
            linereduce < 2 > (red, threadIdx.x);
            linereduce < 1 > (red, threadIdx.x);
            if (threadIdx.x == 0)
                atomicAdd(&r.val[rr], red[0]);
        } else { // row is increasing
            __shared__
            elem_t red[workGroup][nelem];
            red[me][nrow] = acc;
            __syncthreads();
            for (int tot = workGroup >> 1; me < tot; tot >>= 1) {
                red[me][idx] += red[me + tot][idx];
                __syncthreads();
            }
            if (me == 0) {
                atomicAdd(&r.val[rr], red[0][nrow]);
            }
        }
        row += gridDim.x;
    }
};

extern "C" void fbcsr_row_krnl_16(fbcsr *f, vector *v, vector *r) {
    if (f->nnz / f->nr > 512) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 16, 0, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 256) {
        dim3 block(f->nr), thread(64);
        FBCSR_general_krnl < 64, 16, 0, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 128) {
        dim3 block(f->nr), thread(32);
        FBCSR_general_krnl < 32, 16, 0, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 64) {
        dim3 block((f->nr + 1) / 2), thread(32);
        FBCSR_general_krnl < 32, 16, 0, 0, 1, 2 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 32 || f->nr < 400) {
        dim3 block((f->nr + 3) / 4), thread(32);
        FBCSR_general_krnl < 32, 16, 0, 0, 1, 4 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 7) / 8), thread(32);
        FBCSR_general_krnl < 32, 16, 0, 0, 1, 8 ><<<block, thread>>>(*f, *v, *r);
    }
}

extern "C" void fbcsr_row_krnl_32(fbcsr *f, vector *v, vector *r) {
    if (f->nnz / f->nr > 512) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 32, 0, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 256) {
        dim3 block(f->nr), thread(64);
        FBCSR_general_krnl < 64, 32, 0, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 128) {
        dim3 block(f->nr), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 64) {
        dim3 block((f->nr + 1) / 2), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 0, 1, 2 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 32 || f->nr < 400) {
        dim3 block((f->nr + 3) / 4), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 0, 1, 4 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 7) / 8), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 0, 1, 8 ><<<block, thread>>>(*f, *v, *r);
    }
}

extern "C" void fbcsr_col_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(128);
    FBCSR_general_krnl < 128, 16, 0, 1, 0, 1 ><<<block, thread>>>(*f, *v, *r);
}

extern "C" void fbcsr_col_krnl_32(fbcsr *f, vector *v, vector *r) {
    if (f->nnz / f->nr > 512) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 32, 0, 1, 0, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 256) {
        dim3 block(f->nr), thread(64);
        FBCSR_general_krnl < 64, 32, 0, 1, 0, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 128) {
        dim3 block(f->nr), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 1, 0, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 64) {
        dim3 block((f->nr + 1) / 2), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 1, 0, 2 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 32 || f->nr < 400) {
        dim3 block((f->nr + 3) / 4), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 1, 0, 4 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 7) / 8), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 1, 0, 8 ><<<block, thread>>>(*f, *v, *r);
    }
}

extern "C" void fbcsr_fslash_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(128);
    FBCSR_general_krnl < 128, 16, 15, -1, 1, 1 ><<<block, thread>>>(*f, *v, *r);
}

extern "C" void fbcsr_fslash_krnl_32(fbcsr *f, vector *v, vector *r) {
    if (f->nnz / f->nr > 512) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 32, 31, -1, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 256) {
        dim3 block(f->nr), thread(64);
        FBCSR_general_krnl < 64, 32, 31, -1, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 128) {
        dim3 block(f->nr), thread(32);
        FBCSR_general_krnl < 32, 32, 31, -1, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 64) {
        dim3 block((f->nr + 1) / 2), thread(32);
        FBCSR_general_krnl < 32, 32, 31, -1, 1, 2 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 32 || f->nr < 400) {
        dim3 block((f->nr + 3) / 4), thread(32);
        FBCSR_general_krnl < 32, 32, 31, -1, 1, 4 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 7) / 8), thread(32);
        FBCSR_general_krnl < 32, 32, 31, -1, 1, 8 ><<<block, thread>>>(*f, *v, *r);
    }
}

extern "C" void fbcsr_bslash_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(128);
    FBCSR_general_krnl < 128, 16, 0, 1, 1, 1 ><<<block, thread>>>(*f, *v, *r);
}

extern "C" void fbcsr_bslash_krnl_32(fbcsr *f, vector *v, vector *r) {
    if (f->nnz / f->nr > 512) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 32, 0, 1, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 256) {
        dim3 block(f->nr), thread(64);
        FBCSR_general_krnl < 64, 32, 0, 1, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 128) {
        dim3 block(f->nr), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 1, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 64) {
        dim3 block((f->nr + 1) / 2), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 1, 1, 2 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 32 || f->nr < 400) {
        dim3 block((f->nr + 3) / 4), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 1, 1, 4 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 7) / 8), thread(32);
        FBCSR_general_krnl < 32, 32, 0, 1, 1, 8 ><<<block, thread>>>(*f, *v, *r);
    }
}

