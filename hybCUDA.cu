//
// Created by joe on 4/6/16.
//

#include"prefix.h"
#include"testlib/vector_gen.h"
#include"CSR.h"
#include"FBCSR.h"
#include"FBCSR_krnl.h"
#include<cusparse.h>

#define TOTALRUNS 1000

typedef void (*testFunc)(void);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s <matrix.csr> <opt>", argv[0]);
        return -1;
    }
    int opt = 1;
    if (argc > 2)
        switch (argv[2][0]) {
            case 'd':
                opt = 0;
                break;
            case 'g':
                opt = 2;
                break;
            default:
                opt = 1;
        }
    csr c;
    vector vec;
    vector ref;

    csr_readFile(argv[1], &c);
    vector_gen_random(&vec, c.m, NULL);
    vector_init(&ref, c.n);

    // Make reference
    {
        vector cuv;
        vector cur;
        csr cum;
        vector_init(&ref, c.n);
        csr_memCpy(&c, &cum, cpyHostToDevice);

        vector_memCpy(&vec, &cuv, cpyHostToDevice);
        vector_memCpy(&ref, &cur, cpyHostToDevice);
        cusparseMatDescr_t descr = 0;
        cusparseHandle_t handle = 0;
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseCreate(&handle);
        cusparseHybMat_t hybMat;
        cusparseHybPartition_t hybPart = CUSPARSE_HYB_PARTITION_AUTO;
        cusparseCreateHybMat(&hybMat);

        cusparseScsr2hyb(handle, cum.n, cum.m, descr, cum.val, cum.ptr, cum.indx, hybMat, 0, hybPart);

        if (opt) {
            cudaEvent_t st, ed;
            float eltime;
            cudaEventCreate(&st);
            cudaEventCreate(&ed);
            cudaEventRecord(st, 0);
            float unit = 1;
            for (int i = 0; i < TOTALRUNS; ++i) {
                cusparseShybmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &unit, descr, hybMat, cuv.val, &unit, cur.val);
            }
            cudaEventRecord(ed, 0);
            cudaEventSynchronize(ed);
            cudaEventElapsedTime(&eltime, st, ed);

            if (opt == 1)
                printf("%f\n", eltime / TOTALRUNS);
            else
                printf("%f\n", c.nnz / (eltime * 2000));
            cudaEventDestroy(st);
            cudaEventDestroy(ed);
        }
        vector_destroy(&ref);
        vector_memCpy(&cur, &ref, cpyDeviceToHost);

        csr_CUDA_destroy(&cum);
        vector_CUDA_destroy(&cuv);
        vector_CUDA_destroy(&cur);
        cusparseDestroy(handle);
        cusparseDestroyMatDescr(descr);
        cusparseDestroyHybMat(hybMat);
    }
    csr_destroy(&c);
    vector_destroy(&vec);
    vector_destroy(&ref);
    return 0;
}

