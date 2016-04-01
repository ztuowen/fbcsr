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

csr *c;
vector *vec;
vector *ref;
vector *res;
vbr *v;
list *u;

void readTest(void){
    csr_readFile(TESTMATRIX,c);
}

void SpMV_csr_ref(void) {
    // Always assumes that CSR is correct
    csr_SpMV(c,vec,ref);
}

int main() {
    c = malloc(sizeof(csr));
    vec = malloc(sizeof(vector));
    ref = malloc(sizeof(vector));
    res = malloc(sizeof(vector));
    v = malloc(sizeof(vbr));

    readTest();
    vector_gen_random(vec,c->m,NULL);
    vector_init(ref,c->n);
    vector_init(res,c->n);
    SpMV_csr_ref();

}
