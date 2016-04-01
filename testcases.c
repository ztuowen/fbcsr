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
list *u;

void readTest(void){
    csr_readFile(TESTMATRIX,c);
}

void SpMV_csr_ref(void) {
    // Always assumes that CSR is correct
    csr_SpMV(c,vec,ref);
}

void trans_vbr(void) {
    vector *res;
    res = malloc(sizeof(vector));
    csr *nc = malloc(sizeof(csr));
    vector_init(res, c->n);
    vbr *v = malloc(sizeof(vbr));
    csr_vbr(c,v,0.8);
    vbr_csr(v,nc);
    csr_SpMV(nc,vec,res);
    assert((vector_equal(ref,res)));
    csr_destroy(nc);
    free(nc);
    vector_destroy(res);
    free(res);
    vbr_destroy(v);
    free(v);
}

int main() {
    c = malloc(sizeof(csr));
    vec = malloc(sizeof(vector));
    ref = malloc(sizeof(vector));

    readTest();
    vector_gen_random(vec,c->m,NULL);
    vector_init(ref,c->n);
    SpMV_csr_ref();
    printf("SpMV test\n");
    printf("Translation test\n");
    trans_vbr();
    printf("Hooray~!!!!! It is ready to exit, that means:\n1. We don't have any errors.\n2.The tests are not comprehensive.\n");
    csr_destroy(c);
    free(c);
    vector_destroy(vec);
    free(vec);
    vector_destroy(ref);
    free(ref);
}
