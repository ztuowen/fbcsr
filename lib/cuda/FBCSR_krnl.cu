//
// Created by joe on 4/10/16.
//
#include <device_launch_parameters.h>
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
int colm
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
                //atomicAdd(&r.val[rr], red[0]);
                r.val[rr] += red[0];
        } else { // row is increasing
            __shared__
            elem_t red[workGroup][nelem];
            red[me][nrow] = acc;
            if (blockSize > 32)
                __syncthreads();
            if (workGroup > 1)
            for (int tot = workGroup >> 1; me < tot; tot >>= 1) {
                red[me][idx] += red[me + tot][idx];
                __syncthreads();
            }
            if (me == 0) {
                //atomicAdd(&r.val[rr], red[0][nrow]);
                r.val[rr] += red[0][nrow];
            }
        }
        row += gridDim.x;
    }
};
template< int blockSize,
int nelem,
int lineAssign,
int rof,
int rowm,
int colm
>

__global__ void FBCSR_general_col_krnl(fbcsr f, vector v, vector r) {
    int rowoff = threadIdx.x / lineAssign;
    int row = blockIdx.x * (blockSize / lineAssign) + rowoff;
    __shared__ elem_t red[blockSize / lineAssign][lineAssign];
    int rowIdx = threadIdx.x % lineAssign;
    if (row < f.nr) {
        int rc;
        int idx = threadIdx.x % nelem;
        int nrow = rowm * idx + rof;
        int ncol = colm * idx;
        elem_t acc = 0;
        for (int i = f.bptr[row]+rowIdx/ nelem; i < f.bptr[row + 1]; i+=lineAssign/nelem) {
            rc = f.bindx[i] + ncol;
            acc += v.val[rc] * f.val[i * nelem + idx];
        }
        if (lineAssign>nelem){
            red[rowoff][rowIdx] = acc;
            __syncthreads();
            if (rowIdx<nelem) 
                for(int i=rowIdx+nelem; i<lineAssign;i+=nelem)
                    acc+=red[rowoff][i];
        }
        if (rowIdx<nelem)
        {
            int rr = f.rptr[row] + nrow;
            r.val[rr] += acc;
        }
    }
};

template< int blockSize,
int nelem,
int lineAssigned,
int totWork
>

__global__ void FBCSR_row_krnl(fbcsr f, vector v, vector r) {
    int blidx = threadIdx.x % lineAssigned;
    int idx = blidx % nelem;
    int me = blidx / nelem;
    int blrow = threadIdx.x / lineAssigned;
    int rowoff = blockIdx.x * totWork;
    int row;
    int workelem = blrow;
    __shared__ elem_t work[totWork];
    __shared__ elem_t red[blockSize / lineAssigned][lineAssigned + 1];
    for (; workelem < totWork && (row = rowoff + workelem) < f.nr; workelem += blockSize / lineAssigned) {
        int rc;
        elem_t acc = 0;
        for (int i = f.bptr[row] + me; i < f.bptr[row + 1]; i += lineAssigned / nelem) {
            rc = f.bindx[i] + idx;
            acc += v.val[rc] * f.val[i * nelem + idx];
        }
        red[blrow][blidx] = acc;
        if (lineAssigned > 32)
            __syncthreads();
        if (lineAssigned > 128)
            linereduce < 128 > (red[blrow], blidx);
        if (lineAssigned > 64)
            linereduce < 64 > (red[blrow], blidx);
        if (lineAssigned > 32)
            linereduce < 32 > (red[blrow], blidx);
        linereduce < 16 > (red[blrow], blidx);
        linereduce < 8 > (red[blrow], blidx);
        linereduce < 4 > (red[blrow], blidx);
        linereduce < 2 > (red[blrow], blidx);
        linereduce < 1 > (red[blrow], blidx);
        if (blidx == 0) {
            work[workelem] = red[blrow][0];
        }
    }
    __syncthreads();
    if (threadIdx.x < totWork && rowoff + threadIdx.x < f.nr) {
        int rr = f.rptr[rowoff + threadIdx.x];
        r.val[rr] += work[threadIdx.x];
    }
};

template< int blockSize,
int nelem,
int lineAssigned,
int totWork,
int redtime
>
// lineAssigned = 32
__global__ void FBCSR_row_nred_krnl(fbcsr f, vector v, vector r) {
    int blidx = threadIdx.x % lineAssigned;
    int idx = blidx % nelem;
    int blrow = threadIdx.x / lineAssigned;
    int rowoff = blockIdx.x * totWork;
    int row;
    int workelem = blrow;
    __shared__ elem_t work[(totWork + lineAssigned * 3 + 1) * (blockSize / lineAssigned)];
    int st = (totWork + lineAssigned * 3 + 1) * blrow;
    int l = totWork + lineAssigned - 1;
    int np = (blidx % 2) * (l + 1) + (blidx / 2);
    int storp = lineAssigned / 2 - 1 + (blidx % 2) * l + (blidx / 2);
    for (; workelem < totWork && (row = rowoff + workelem) < f.nr; workelem += blockSize / lineAssigned) {
        int rc;
        elem_t acc = 0;
        for (int i = f.bptr[row] + blidx/32; i < f.bptr[row + 1]; i+=lineAssigned/32) {
            rc = f.bindx[i] + idx;
            acc += v.val[rc] * f.val[i * nelem + idx];
        }
        work[st + storp] = acc;
        if (lineAssigned>32) __syncthreads();
        if (blidx < lineAssigned - 1) {
            elem_t tmp = work[blidx + st] + work[blidx + st + l];
            if (lineAssigned>32) __syncthreads();
            work[st + np] = tmp;
            if (lineAssigned>32) __syncthreads();
        }
        ++st;
    }
    for (int i = 0; i < redtime; ++i) { // hardcoded 2^5 = 32
        work[st + storp] = 0;
        if (lineAssigned>32) __syncthreads();
        if (blidx < lineAssigned - 1) {
            elem_t tmp = work[blidx + st] + work[blidx + st + l];
            if (lineAssigned>32) __syncthreads();
            work[st + np] = tmp;
            if (lineAssigned>32) __syncthreads();
        }
        ++st;
    }
    __syncthreads();
    if (threadIdx.x < totWork && rowoff + threadIdx.x < f.nr) {
        int rr = (threadIdx.x % (blockSize / lineAssigned)) * (totWork + lineAssigned * 3 + 1) +
                 threadIdx.x / (blockSize / lineAssigned) + redtime;
        row = f.rptr[rowoff + threadIdx.x];
        r.val[row] += work[rr];
    }
};

template< int blockSize
> __global__ void FBCSR_square_krnl(fbcsr f, vector v, vector r) {
    int me = threadIdx.x / 32;
    int idx = threadIdx.x % 32;
    __shared__ elem_t tmp[blockSize / 32][32];
    elem_t acc = 0;
    for (int i = f.bptr[blockIdx.x] + me; i < f.bptr[blockIdx.x + 1]; i += blockSize / 32) {
        int offset = i * f.nelem;
        // read vector
        tmp[me][idx] = v.val[f.bindx[i] + idx];

        for (int j = 0; j < 32; ++j, offset += 32)
            acc += tmp[me][j] * f.val[offset + idx];
    }
    tmp[me][idx] = acc;
    if (blockSize > 32)
        __syncthreads();
    if (me == 0) {
        for (int i = 1; i < blockSize / 32; ++i)
            acc += tmp[i][idx];
        int rr = f.rptr[blockIdx.x];
        r.val[rr + idx] += acc;
    }
}

extern "C" void fbcsr_row_krnl_16(fbcsr *f, vector *v, vector *r) {
    if (f->nnz / f->nr > 512) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 16, 0, 0, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 256) {
        dim3 block(f->nr), thread(64);
        FBCSR_general_krnl < 64, 16, 0, 0, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 128) {
        dim3 block(f->nr), thread(32);
        FBCSR_general_krnl < 32, 16, 0, 0, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 64) {
        dim3 block((f->nr + 1) / 2), thread(32);
        FBCSR_general_krnl < 32, 16, 0, 0, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nnz / f->nr > 32 || f->nr < 400) {
        dim3 block((f->nr + 3) / 4), thread(32);
        FBCSR_general_krnl < 32, 16, 0, 0, 1 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 7) / 8), thread(32);
        FBCSR_general_krnl < 32, 16, 0, 0, 1 ><<<block, thread>>>(*f, *v, *r);
    }
}

extern "C" void fbcsr_row_krnl_32(fbcsr *f, vector *v, vector *r) {
    if (f->nb / f->nr > 32) {
        dim3 block(f->nr), thread(256);
        FBCSR_row_nred_krnl < 256, 32, 256, 1, 7 ><<<block, thread>>>(*f, *v, *r);
    } else
    if (f->nb / f->nr > 16) {
        dim3 block(f->nr), thread(128);
        FBCSR_row_nred_krnl < 128, 32, 128, 1, 6 ><<<block, thread>>>(*f, *v, *r);
    } else
    if (f->nb / f->nr > 4) {
        dim3 block((f->nr + 3) / 4), thread(64);
        FBCSR_row_nred_krnl < 64, 32, 32, 4, 4 ><<<block, thread>>>(*f, *v, *r);
    } else
     if (f->nb / f->nr > 2 || f->nr < 256 * 32) {
        dim3 block((f->nr + 7) / 8), thread(64);
        FBCSR_row_nred_krnl < 64, 32, 32, 8 ,4><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 31) / 32), thread(64);
        FBCSR_row_nred_krnl < 64, 32, 32, 32 ,4><<<block, thread>>>(*f, *v, *r);
    }
}

extern "C" void fbcsr_col_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(128);
    FBCSR_general_krnl < 128, 16, 0, 1, 0 ><<<block, thread>>>(*f, *v, *r);
}

extern "C" void fbcsr_col_krnl_32(fbcsr *f, vector *v, vector *r) {
    /*if (f->nb / f->nr > 16) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 32, 0, 1, 0 ><<<block, thread>>>(*f, *v, *r);
    } else */if (f->nb / f->nr > 64) {
        dim3 block(f->nr), thread(256);
        FBCSR_general_col_krnl < 256, 32, 256, 0, 1, 0 ><<<block, thread>>>(*f, *v, *r);
    } else
    if (f->nb / f->nr > 32) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_col_krnl < 128, 32, 128, 0, 1, 0 ><<<block, thread>>>(*f, *v, *r);
    } else
    if (f->nb / f->nr > 2 || f->nr < 4 * 512) {
        dim3 block((f->nr + 1) / 2), thread(64);
        FBCSR_general_col_krnl < 64, 32, 32, 0, 1, 0 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 3) / 4), thread(128);
        FBCSR_general_col_krnl < 128, 32, 32, 0, 1, 0 ><<<block, thread>>>(*f, *v, *r);
    }
}

extern "C" void fbcsr_fslash_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(128);
    FBCSR_general_krnl < 128, 16, 15, -1, 1 ><<<block, thread>>>(*f, *v, *r);
}

extern "C" void fbcsr_fslash_krnl_32(fbcsr *f, vector *v, vector *r) {
    /*if (f->nb / f->nr > 16) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 32, 31, -1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nb / f->nr > 4) {
        dim3 block(f->nr), thread(64);
        FBCSR_general_krnl < 64, 32, 31, -1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else*/ if (f->nb / f->nr > 64) {
        dim3 block(f->nr), thread(256);
        FBCSR_general_col_krnl < 256, 32, 256, 31, -1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else
    if (f->nb / f->nr > 32) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_col_krnl < 128, 32, 128, 31, -1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else
    if (f->nb / f->nr > 2 || f->nr < 4 * 512) {
        dim3 block((f->nr+1)/2 ), thread(64);
        FBCSR_general_col_krnl < 64, 32, 32, 31, -1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 3) / 4), thread(128);
        FBCSR_general_col_krnl < 128, 32, 32, 31, -1, 1 ><<<block, thread>>>(*f, *v, *r);
    }
}

extern "C" void fbcsr_bslash_krnl_16(fbcsr *f, vector *v, vector *r) {
    dim3 block(f->nr), thread(128);
    FBCSR_general_krnl < 128, 16, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
}

extern "C" void fbcsr_bslash_krnl_32(fbcsr *f, vector *v, vector *r) {
    /*if (f->nb / f->nr > 16) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 32, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nb / f->nr > 4) {
        dim3 block(f->nr), thread(64);
        FBCSR_general_krnl < 64, 32, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else*/ if (f->nb / f->nr > 64) {
        dim3 block(f->nr), thread(256);
        FBCSR_general_krnl < 256, 32, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else
    if (f->nb / f->nr > 32) {
        dim3 block(f->nr), thread(128);
        FBCSR_general_krnl < 128, 32, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else
    if (f->nb / f->nr > 2 || f->nr < 4 * 512) {
        dim3 block((f->nr+1)/2 ), thread(64);
        FBCSR_general_col_krnl < 64, 32, 32, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block((f->nr + 3) / 4), thread(128);
        FBCSR_general_col_krnl < 128, 32, 32, 0, 1, 1 ><<<block, thread>>>(*f, *v, *r);
    }
}


extern "C" void fbcsr_square_krnl(fbcsr *f, vector *v, vector *r) {
    if (f->nb / f->nr > 128) {
        dim3 block(f->nr), thread(1024);
        FBCSR_square_krnl < 1024 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nb / f->nr > 64) {
        dim3 block(f->nr), thread(512);
        FBCSR_square_krnl < 512 ><<<block, thread>>>(*f, *v, *r);
    } else if (f->nb / f->nr > 32) {
        dim3 block(f->nr), thread(256);
        FBCSR_square_krnl < 256 ><<<block, thread>>>(*f, *v, *r);
    } else {
        dim3 block(f->nr), thread(128);
        FBCSR_square_krnl < 128 ><<<block, thread>>>(*f, *v, *r);
    }
}
