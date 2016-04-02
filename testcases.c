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
#include<stdio.h>
#include<assert.h>

csr *c;
vector *vec;
vector *ref;

typedef void (*testFunc)(void);

void readTest(void) {
    csr_readFile(TESTMATRIX, c);
}

void SpMV_csr_ref(void) {
    // Always assumes that CSR is correct
    csr_SpMV(c, vec, ref);
}

void SpMV_vbr(void) {
    vector *res;
    res = malloc(sizeof(vector));
    vector_init(res, c->n);
    vbr *v = malloc(sizeof(vbr));

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
    vector *res = malloc(sizeof(vector));
    vector_init(res, c->n);
    u = malloc(sizeof(ubcsr));
    ubcsr_makeEmpty(u, c->n, c->m, 1, 2, NULL);
    l = list_add(l, u);

    rem = csr_ubcsr(c, l, 0.8);
    ubcsr_SpMV(l, rem, vec, res);

    assert((vector_equal(ref, res)));

    list_destroy(l, ubcsr_destroy);
    csr_destroy(rem);
    free(rem);
    vector_destroy(res);
    free(res);
}

void trans_vbr(void) {
    vector *res;
    res = malloc(sizeof(vector));
    csr *nc = malloc(sizeof(csr));
    vector_init(res, c->n);
    vbr *v = malloc(sizeof(vbr));

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
    csr *nc = malloc(sizeof(csr));
    vector *res = malloc(sizeof(vector));
    vector_init(res, c->n);
    u = malloc(sizeof(ubcsr));
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

char *tNames[] = {
        "SpMV using CSR as ref",
        "SpMV using VBR",
        "SpMV using UBCSR",
        "Translate to VBR",
        "Translate to UBCSR",
        NULL};
testFunc tFuncs[] = {
        SpMV_csr_ref,
        SpMV_vbr,
        SpMV_ubcsr,
        trans_vbr,
        trans_ubcsr,
        NULL};

int main() {
    c = malloc(sizeof(csr));
    vec = malloc(sizeof(vector));
    ref = malloc(sizeof(vector));

    readTest();
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
    printf("Hooray~!!!!! It is ready to exit, that means:\n1. We don't have any errors.\n2.The tests are not comprehensive.\n");
    csr_destroy(c);
    free(c);
    vector_destroy(vec);
    free(vec);
    vector_destroy(ref);
    free(ref);
    return 0;
}
