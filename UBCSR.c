#include"UBCSR.h"
#include"VBR.h"
#include<assert.h>

csr* csr_splitOnce(csr *c,ubcsr *u, float thresh)
{
    // TODO implement!
    vbr *v = malloc(sizeof(vbr));
    csr *rem = malloc(sizeof(csr));
    int i,j,k,lk;
    int rcnt,bcnt;
    csr_vbr(c,v,thresh);

    // Converting from VBR to UBCSR
    u->n = v->n;
    u->m = v->m;
    
    // Counting
    rcnt=0; // row count
    bcnt=0; // block count

    for (i = 0;i<nr;++i)
    {
        for (j=0;j+u->r<=v->rptr[i+1]-v->rptr[i];j+=u->r) //row
        {
            ++rcnt;
            for (k = v->bptr[i];k<v->bptr[i+1];++k)
            {
                int rowsize = (v->cptr[v->bindx[k]+1] - v->cptr[v->bindx[k]]);
                lk=0;
                while (lk + u->c <= rowsize)
                {
                    int index,nz=1;
                    index = v->indx[k] + lk + j*rowsize;
                    for (int cr = 0;cr<u->r && (nz);++cr,index+=(rowsize-c))
                        for (int cc = 0;cc<u->c && (nz);++cc,++index)
                            if (v->val[index]>0){ // Encountered a nonzero adding
                                ++bcnt;
                                nz=0; // just to break, how i wish to use goto
                            }
                    lk+=c;
                }
            }
        }
    }
    
    // mallocs
    u -> bindx = malloc(bcnt*sizeof(int));
    u -> val = malloc(bcnt*c*r*sizeof(elem_t));
    u -> bptr = malloc((rcnt+1)*sizeof(int));
    u -> rptr = malloc(rcnt*sizeof(int));
    u -> nr = rcnt;
    bcnt=0;
    rcnt=0;
    // Now, add everything!
    for (i = 0;i<nr;++i)
    {
        for (j=0;j+u->r<=v->rptr[i+1]-v->rptr[i];j+=u->r) //row
        {
            u->bptr[u->nr] = bcnt;
            u->rptr[(u->nr)++] = j+v->rptr[i];
            for (k = v->bptr[i];k<v->bptr[i+1];++k)
            {
                int rowsize = (v->cptr[v->bindx[k]+1] - v->cptr[v->bindx[k]]);
                lk=0;
                while (lk + u->c <= rowsize)
                {
                    int index,nz=1;
                    index = v->indx[k] + lk + j*rowsize;
                    for (int cr = 0;cr<u->r && (nz);++cr,index+=(rowsize-c))
                        for (int cc = 0;cc<u->c && (nz);++cc,++index)
                            if (v->val[index]>0){ // Encountered a nonzero adding
                                nz=0; // just to break, how i wish to use goto
                            }
                    if (nz==0)
                    {
                        int st = bcnt*c*r;
                        index = v->indx[k] + lk + j*rowsize;
                        u->bindx[bcnt] = v->cptr[v->bindx[k]] + lk;
                        for (int cr = 0;cr<u->r;++cr,index+=(rowsize-c),++st)
                            for (int cc = 0;cc<u->c;++cc,++index,++st)
                            {
                                u->val[st] = v->val[index];
                                v->val[index]=0;
                            }
                        ++bcnt;
                    }
                    lk+=c;
                }
            }
        }
    }
    assert(u->nr == rcnt);
    u->bptr[rcnt]=bcnt;
    // Get the remainder
    vbr_csr(v,rem);
    return rem;
}

csr* csr_ubcsr(csr *c,list *l,float thresh)
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
