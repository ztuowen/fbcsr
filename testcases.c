//
// Created by joe on 4/1/16.
// This includes a lot of tests
// Add more test case to it when build
//

#include"prefix.h"
#include"testlib/test_prefix.h"
#include"testlib/vector_gen.h"
#include"CSR.h"
#include"VBR.h"
#include"UBCSR.h"
#include"FBCSR.h"
#include"timing.h"
#include<unistd.h>
#include<assert.h>
#include<math.h>

csr *c;
vector *vec;
vector *ref;

typedef void (*testFunc)(void);

void SpMV_csr_ref(void) {
    // Always assumes that CSR is correct
    csr_SpMV(c, vec, ref);
}

void SpMV_vbr(void) {
    vector *res;
    res = (vector *) malloc(sizeof(vector));
    vector_init(res, c->n);
    vbr *v = (vbr *) malloc(sizeof(vbr));

    csr_vbr(c, v, 0.8);
    vbr_SpMV(v, vec, res);

    assert((vector_equal(ref, res)));

    vector_destroy(res);
    free(res);
    vbr_destroy(v);
    free(v);
}

void SpMV_ubcsr() {
    list *l = NULL;
    ubcsr *u;
    csr *rem;
    vector res;
    vector_init(&res, c->n);
    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 1, 4, NULL);
    l = list_add(l, u);
    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 4, 1, NULL);
    l = list_add(l, u);
    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 4, 4, NULL);
    l = list_add(l, u);

    rem = csr_ubcsr(c, l, 0.8);
    csr_SpMV(rem, vec, &res);
    ubcsr_SpMV(l, vec, &res);

    assert((vector_equal(ref, &res)));

    list_destroy(l, ubcsr_destroy);
    csr_destroy(rem);
    free(rem);
    vector_destroy(&res);
}

void SpMV_CUDA_csr() {
    vector cuv;
    vector cur;
    csr cum;
    vector res;
    vector_init(&res, c->n);
    csr_memCpy(c, &cum, cpyHostToDevice);

    vector_memCpy(vec, &cuv, cpyHostToDevice);
    vector_memCpy(&res, &cur, cpyHostToDevice);

    csr_CUDA_SpMV(&cum, &cuv, &cur);

    vector_destroy(&res);
    vector_memCpy(&cur, &res, cpyDeviceToHost);

    assert((vector_equal(ref, &res)));

    vector_destroy(&res);
    csr_CUDA_destroy(&cum);
    vector_CUDA_destroy(&cuv);
    vector_CUDA_destroy(&cur);
}

void SpMV_CUDA_ubcsr() {
    list *l = NULL;
    list *cul = NULL;
    vector cuv;
    vector cur;
    ubcsr *u;
    csr *rem, curem;
    vector res;
    vector_init(&res, c->n);

    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 1, 4, NULL);
    l = list_add(l, u);
    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 4, 1, NULL);
    l = list_add(l, u);
    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 4, 4, NULL);
    l = list_add(l, u);

    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 1, 4, NULL);
    cul = list_add(cul, u);
    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 4, 1, NULL);
    cul = list_add(cul, u);
    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 4, 4, NULL);
    cul = list_add(cul, u);

    rem = csr_ubcsr(c, l, 0.8);

    vector_memCpy(vec, &cuv, cpyHostToDevice);
    vector_memCpy(&res, &cur, cpyHostToDevice);
    csr_memCpy(rem, &curem, cpyHostToDevice);
    ubcsr_memCpy(l, cul, cpyHostToDevice);

    csr_CUDA_SpMV(&curem, &cuv, &cur);
    ubcsr_CUDA_SpMV(cul, &cuv, &cur);

    vector_destroy(&res);
    vector_memCpy(&cur, &res, cpyDeviceToHost);

    assert((vector_equal(ref, &res)));

    list_destroy(l, ubcsr_destroy);
    csr_destroy(rem);
    free(rem);
    vector_destroy(&res);
    csr_CUDA_destroy(&curem);
    vector_CUDA_destroy(&cuv);
    vector_CUDA_destroy(&cur);
    list_destroy(cul, ubcsr_CUDA_destroy);
}

void trans_vbr(void) {
    vector *res;
    res = (vector *) malloc(sizeof(vector));
    csr *nc = (csr *) malloc(sizeof(csr));
    vector_init(res, c->n);
    vbr *v = (vbr *) malloc(sizeof(vbr));

    csr_vbr(c, v, 0.8);
    vbr_csr(v, nc);
    csr_SpMV(nc, vec, res);

    assert((vector_equal(ref, res)));

    csr_destroy(nc);
    free(nc);
    vector_destroy(res);
    free(res);
    vbr_destroy(v);
    free(v);
}

void trans_ubcsr() {
    list *l = NULL;
    ubcsr *u;
    csr *rem;
    csr *nc = (csr *) malloc(sizeof(csr));
    vector *res = (vector *) malloc(sizeof(vector));
    vector_init(res, c->n);
    u = (ubcsr *) malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 1, 2, NULL);
    l = list_add(l, u);

    rem = csr_ubcsr(c, l, 0.8);
    ubcsr_csr(l, rem, nc);
    csr_SpMV(nc, vec, res);

    assert((vector_equal(ref, res)));

    list_destroy(l, ubcsr_destroy);
    csr_destroy(nc);
    free(nc);
    csr_destroy(rem);
    free(rem);
    vector_destroy(res);
    free(res);
}

void SpMV_fbcsr() {
    list *l = NULL;
    fbcsr *f;
    csr *rem;
    vector *res = (vector *) malloc(sizeof(vector));
    vector_init(res, c->n);
    f = (fbcsr *) malloc(sizeof(fbcsr));
    fbcsr_makeEmpty(f, c->n, c->m, 1, 4, 4, NULL, fbcsr_column);
    l = list_add(l, f);

    rem = csr_fbcsr(c, l, 0.7);
    csr_SpMV(rem, vec, res);
    fbcsr_SpMV(l, vec, res);
    assert((vector_equal(ref, res)));

    list_destroy(l, fbcsr_destroy);
    csr_destroy(rem);
    free(rem);
    vector_destroy(res);
    free(res);
}

void trans_fbcsr() {
    list *l = NULL;
    fbcsr *f;
    csr *rem;
    csr *nc = (csr *) malloc(sizeof(csr));
    vector *res = (vector *) malloc(sizeof(vector));
    vector_init(res, c->n);
    f = (fbcsr *) malloc(sizeof(fbcsr));
    fbcsr_makeEmpty(f, c->n, c->m, 1, 4, 4, NULL, fbcsr_column);
    l = list_add(l, f);

    rem = csr_fbcsr(c, l, 0.7);
    fbcsr_csr(l, rem, nc);
    csr_SpMV(nc, vec, res);
    assert((vector_equal(ref, res)));

    list_destroy(l, fbcsr_destroy);
    csr_destroy(nc);
    free(nc);
    csr_destroy(rem);
    free(rem);
    vector_destroy(res);
    free(res);
}

void timing() {
    struct timespec b, e;
    long sec, nsec;
    float msec;
    GET_TIME(b);
    sleep(1);
    GET_TIME(e);
    msec = elapsed_time_msec(&b, &e, &sec, &nsec);
    // Allow some uncertainty for sleep
    assert(fabs(msec - 1000) < 10);
}

char *tNames[] = {
        "SpMV using CSR as ref",
        "SpMV using VBR",
        "SpMV using UBCSR",
        "SpMV using FBCSR",
        "Translate using FBCSR",
        "Translate to VBR",
        "Translate to UBCSR",
        "Timing",
        "SpMV using CSR+CUDA",
        "SpMV using UBCSR+CUDA",
        NULL};

testFunc tFuncs[] = {
        SpMV_csr_ref,
        SpMV_vbr,
        SpMV_ubcsr,
        SpMV_fbcsr,
        trans_fbcsr,
        trans_vbr,
        trans_ubcsr,
        timing,
        SpMV_CUDA_csr,
        SpMV_CUDA_ubcsr,
        NULL};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s <matrix.csr>", argv[0]);
        return -1;
    }
    c = (csr *) malloc(sizeof(csr));
    vec = (vector *) malloc(sizeof(vector));
    ref = (vector *) malloc(sizeof(vector));

    csr_readFile(argv[1], c);
    vector_gen_random(vec, c->m, NULL);
    vector_init(ref, c->n);
    int i = 0;
    while (tNames[i] != NULL) {
        printf("Testing: %s ... ", tNames[i]);
        fflush(stdout);
        tFuncs[i]();
        printf("OK\n");
        ++i;
    }
    printf("Hooray~!!!!! It is ready to exit, that means:\n1. We don't have any errors.\n2. The tests are not comprehensive.\n");
    csr_destroy(c);
    free(c);
    vector_destroy(vec);
    free(vec);
    vector_destroy(ref);
    free(ref);
    return 0;
}
