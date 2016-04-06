//
// Created by joe on 4/6/16.
//

#include"prefix.h"
#include"testlib/vector_gen.h"
#include"CSR.h"
#include"UBCSR.h"


typedef void (*testFunc)(void);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s <matrix.csr>", argv[0]);
        return -1;
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

        csr_CUDA_SpMV(&cum, &cuv, &cur);

        vector_destroy(&ref);
        vector_memCpy(&cur, &ref, cpyDeviceToHost);

        csr_CUDA_destroy(&cum);
        vector_CUDA_destroy(&cuv);
        vector_CUDA_destroy(&cur);
    }
    // UBCSR
    {
        list *l = NULL;
        list *cul = NULL;
        vector cuv;
        vector cur;
        ubcsr *u;
        csr *rem, curem;
        vector res;
        float eltime;
        cudaEvent_t st, ed;
        cudaEventCreate(&st);
        cudaEventCreate(&ed);
        vector_init(&res, c.n);
        u = (ubcsr *) malloc(sizeof(ubcsr));
        ubcsr_makeEmpty(u, c.n, c.m, 1, 2, NULL);
        l = list_add(l, u);

        u = (ubcsr *) malloc(sizeof(ubcsr));
        ubcsr_makeEmpty(u, c.n, c.m, 1, 2, NULL);
        cul = list_add(cul, u);

        rem = csr_ubcsr(&c, l, 0.8);

        vector_memCpy(&vec, &cuv, cpyHostToDevice);
        vector_memCpy(&res, &cur, cpyHostToDevice);
        csr_memCpy(rem, &curem, cpyHostToDevice);
        ubcsr_memCpy(l, cul, cpyHostToDevice);

        cudaEventRecord(st, 0);

        csr_CUDA_SpMV(&curem, &cuv, &cur);
        ubcsr_CUDA_SpMV(cul, &cuv, &cur);

        cudaEventRecord(ed, 0);
        cudaEventSynchronize(ed);
        cudaEventElapsedTime(&eltime, st, ed);
        printf("%f\n", eltime);

        vector_destroy(&res);
        vector_memCpy(&cur, &res, cpyDeviceToHost);

        if (!vector_equal(&ref, &res))
            return -1;

        cudaEventDestroy(st);
        cudaEventDestroy(ed);
        list_destroy(l, ubcsr_destroy);
        csr_destroy(rem);
        free(rem);
        vector_destroy(&res);
        csr_CUDA_destroy(&curem);
        vector_CUDA_destroy(&cuv);
        vector_CUDA_destroy(&cur);
        list_destroy(cul, ubcsr_CUDA_destroy);
    }
    csr_destroy(&c);
    vector_destroy(&vec);
    vector_destroy(&ref);
    return 0;
}

