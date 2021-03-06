//
// Created by joe on 4/6/16.
//

#include"prefix.h"
#include"testlib/vector_gen.h"
#include"CSR.h"
#include"FBCSR.h"
#include"FBCSR_krnl.h"

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
            case 'b':
                opt = 3;
                break;
            default:
                opt = 1;
        }
    csr c;
    vector vec;
    vector ref;

    csr_readFile(argv[1], &c);
    vector_gen(&vec, c.m, NULL);
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

        if (opt) {
            cudaEvent_t st, ed;
            float eltime;
            cudaEventCreate(&st);
            cudaEventCreate(&ed);
            cudaEventRecord(st, 0);
            for (int i = 0; i < TOTALRUNS; ++i) {
                csr_CUDA_SpMV(&cum, &cuv, &cur);
            }
            cudaEventRecord(ed, 0);
            cudaEventSynchronize(ed);
            cudaEventElapsedTime(&eltime, st, ed);
            float cnt = 0;

            cnt += 2 * sizeof(int) * c.n;     // row pointer
            cnt += 1 * sizeof(int) * c.nnz; // column index
            cnt += 2 * sizeof(elem_t) * c.nnz; // A[i,j] and x[j]
            cnt += 2 * sizeof(elem_t) * c.n;

            switch (opt) {
                case 1:
                default:
                    printf("%f\n", eltime / TOTALRUNS);
                    break;
                case 2:
                    printf("%f\n", 2 * c.nnz / (eltime * (1000000 / TOTALRUNS)));
                    break;
                case 3:
                    printf("%f\n", cnt / (eltime * (1000000 / TOTALRUNS)));
                    break;
            }
            cudaEventDestroy(st);
            cudaEventDestroy(ed);
        }
        vector_destroy(&ref);
        vector_memCpy(&cur, &ref, cpyDeviceToHost);

        csr_CUDA_destroy(&cum);
        vector_CUDA_destroy(&cuv);
        vector_CUDA_destroy(&cur);
    }
    csr_destroy(&c);
    vector_destroy(&vec);
    vector_destroy(&ref);
    return 0;
}

