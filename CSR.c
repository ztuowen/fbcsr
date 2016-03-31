#include"CSR.h"

void csr_destroy(void* c)
{
    csr *cc = (csr*) c;
    free(cc->val);
    free(cc->indx);
    free(cc->ptr);
}
