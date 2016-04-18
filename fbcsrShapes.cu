//
// Created by joe on 4/6/16.
//

#include"prefix.h"
#include"testlib/vector_gen.h"
#include"FBCSR.h"
#include"FBCSR_krnl.h"
#include"testlib/fbcsr_gen.h"

#define TOTALRUNS 1000

typedef void (*testFunc)(void);

int n = 100000;
int m = 100000;

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "USAGE: %s <sel> <nr> <avg>", argv[0]);
        return -1;
    }
    int nr, avg, sel;
    if (sscanf(argv[1], "%d", &sel) != 1)
        exit(-1);
    if (sscanf(argv[2], "%d", &nr) != 1)
        exit(-1);
    if (sscanf(argv[3], "%d", &avg) != 1)
        exit(-1);
    vector vec;
    vector ref;

    vector_gen(&vec, m, NULL);
    vector_init(&ref, n);

    // FBCSR
    {
        list *l = NULL;
        list *cul = NULL;
        vector cuv;
        vector cur;
        fbcsr *f;
        vector res;
        float eltime;

        vector_init(&res, n);
        switch (sel) {
            case 0:
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 1, 32, 32, 0.6, (void *) fbcsr_bslash_krnl_32, (void *) fbcsr_backwardSlash);
                cul = list_add(cul, f);
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 1, 32, 32, 0.6, NULL, (void *) fbcsr_backwardSlash);
                l = list_add(l, f);
                break;
            case 1:
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 1, 32, 32, 0.6, (void *) fbcsr_fslash_krnl_32, (void *) fbcsr_forwardSlash);
                cul = list_add(cul, f);
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 1, 32, 32, 0.6, NULL, (void *) fbcsr_forwardSlash);
                l = list_add(l, f);
                break;
            case 2:
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 32, 1, 32, 0.6, (void *) fbcsr_row_krnl_32, (void *) fbcsr_row);
                cul = list_add(cul, f);
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 32, 1, 32, 0.6, NULL, (void *) fbcsr_row);
                l = list_add(l, f);
                break;
            case 3:
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 1, 32, 32, 0.6, (void *) fbcsr_col_krnl_32, (void *) fbcsr_column);
                cul = list_add(cul, f);
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 1, 32, 32, 0.6, NULL, (void *) fbcsr_column);
                l = list_add(l, f);
                break;
            case 4:
            default:
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 32, 32, 1024, 0.4, (void *) fbcsr_square_krnl, (void *) fbcsr_square);
                cul = list_add(cul, f);
                f = (fbcsr *) malloc(sizeof(fbcsr));
                fbcsr_makeEmpty(f, n, m, 32, 32, 1024, 0.4, NULL, (void *) fbcsr_square);
                l = list_add(l, f);
                break;
        }
        fbcsr_gen(f, nr, avg);
        fbcsr_SpMV(l, &vec, &ref);

        vector_memCpy(&vec, &cuv, cpyHostToDevice);
        vector_memCpy(&res, &cur, cpyHostToDevice);
        fbcsr_memCpy(l, cul, cpyHostToDevice);

        {
            cudaEvent_t st, ed;
            cudaEventCreate(&st);
            cudaEventCreate(&ed);
            cudaEventRecord(st, 0);
            for (int i = 0; i < TOTALRUNS; ++i) {
                fbcsr_CUDA_SpMV(cul, &cuv, &cur);
            }
            cudaEventRecord(ed, 0);
            cudaEventSynchronize(ed);
            cudaEventElapsedTime(&eltime, st, ed);

            printf("%f\n", 2 * avg * nr * f->nelem / (eltime * (1000000 / TOTALRUNS)));
            cudaEventDestroy(st);
            cudaEventDestroy(ed);
        }

        vector_memCpy(&res, &cur, cpyHostToDevice);

        fbcsr_CUDA_SpMV(cul, &cuv, &cur);

        vector_destroy(&res);
        vector_memCpy(&cur, &res, cpyDeviceToHost);

        if (!vector_equal(&ref, &res)) {
            fprintf(stderr, "Result mismatch\n");
        }

        list_destroy(l, fbcsr_destroy);
        vector_destroy(&res);
        vector_CUDA_destroy(&cuv);
        vector_CUDA_destroy(&cur);
        list_destroy(cul, fbcsr_CUDA_destroy);
    }
    vector_destroy(&vec);
    vector_destroy(&ref);
    return 0;
}

