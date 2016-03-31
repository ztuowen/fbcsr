#include"UBCSR.h"

csr* csr_splitOnce(csr *c,ubcsr *u, float thresh)
{
    // TODO implement!
    return NULL;
}

csr* csr_split(csr *c,list *l,float thresh)
{
    ubcsr *u;
    csr *r;
    csr *last;

    if (l!=NULL)
    {
        u = (ubcsr*)list_get(l);
        last = csr_splitOnce(c,u,thresh);
        l=list_next(l);
    }
    
    while (l!=NULL)
    {
        u = (ubcsr*)list_get(l);

        rem = csr_splitOnce(last,u,thresh);

        // setup for next iter
        l=list_next(l);
        csr_destroy(last);
        free(last);
        last = rem;
    }

    return last;
}
