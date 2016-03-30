#include"VBR.h"
#include"string.h"

int condense(int *e,int *c_e,int len, float thresh, int **out)
{
    int cnt=len,j=0;
    for (int i=1;i<len;++i)
        if (elems[i]*elem[i-1]>=thresh * c_e[i] * c_e[i-1])
            --cnt;
    *out = malloc(cnt*sizeof(int));
    *out[0]=0;
    for (int i=1;i<len;++i)
        if (elems[i]*elem[i-1]<thresh * c_e[i] * c_e[i-1])
            *out[++j] = i;
    *out[++j] = len;
    return cnt - 1;
}

int csr_vbr(csr *c,vbr *v,float thresh)
{
    int *k_row,*row,*k_col,*col; // similarity count
    int i,j,lj,lk;
    int nr,nc;
    v->n = c->n;
    v->m = c->m;
    k_col = malloc(v->m*sizeof(int));
    col = malloc(v->m*sizeof(int));
    k_row = malloc(v->n*sizeof(int));
    row = malloc(v->n*sizeof(int));
    memset(k_col,0,v->m*sizeof(int));
    memset(col,0,v->m*sizeof(int));
    memset(k_row,0,v->n*sizeof(int));
    memset(row,0,v->n*sizeof(int));
    // calculate similarity between blocks
    j = 0;
    lj = 0; // last row place
    for (i=0;i<n;++i)
    {
        lk =-2; // last column count
        for (;j<c->ptr[i+1];++j)
        {
            ++k_row[i];
            ++k_col[c->indx[j]];
            // Check similarity with last row
            if (c->indx[j] == lk+1)
                ++col[c->indx[j]];
            while (lj<ptr[i] && c->indx[lj] < c->indx[j])
                ++lj;
            if (lj < ptr[i] && c->indx[lj] < c->indx[j])
                ++row[i];
        }
        lj=ptr[i];
    }
    // condense row
    nr = v->nr = condense(row,k_row,v->n,thresh,v->rptr);
    v-> bptr = malloc(nr*sizeof(int));
    // condense column
    nc = condense(col,k_col,v->m,thresh,v->cptr);
c
    int cnt = 0; // The total number of elements needs to be added
    int bcnt = 0;   // The total number of blocks
    v->bptr[0]=0;
    for (i = 0;i<nr;++i) {
        memset(col,0,nc*sizeof(int));
        for (j=v->rptr[i];j<v->rptr[i+1];++j) // j is the row
        {
            lk=1;
            for (lj = c->ptr[j];lj<c->ptr[j+1];++lj)
            {
                while (v->cptr[lk]<=c->indx[lj])++lk;
                if (col[lk]==0)
                {
                    cnt += (v->cptr[lk] - v->cptr[lk-1])*(v->rptr[i+1]-v->rptr[i]);
                    ++bcnt;
                }
                ++col[lk];
            }
        }
        v -> bptr[i+1] = bcnt;
    }
    v -> bindx = malloc(bcnt*sizeof(int));
    v -> indx = malloc((bcnt+1)*sizeof(int)); // 1 is added as guard
    v -> val = malloc(cnt*sizeof(elem_t));
    // clear val to 0 - cannot memset because might not be int
    v->indx[bcnt] = cnt;
    for (i=0;i<cnt;++i)
        v->val[i] = 0; 

    // Finally the constructing part
    cnt = 0;    // val counter
    for (i=0; i<nr;++i)
    {
        memset(col,0,nc*sizeof(int));
        for (j=v->rptr[i];j<v->rptr[i+1];++j) // j is the row
        {
            lk=1;
            for (lj = c->ptr[j];lj<c->ptr[j+1];++lj)
            {
                while (v->cptr[lk]<=c->indx[lj])++lk;
                ++col[lk];
            }
        }
        lj=v->bptr[i];
        for (j=1;j<nc;++j)
            if (col[j]) {
                v->bindx[lj] = v->cptr[j-1];
                v->indx[lj] = cnt;
                col[j]=lj;
                cnt += (v->cptr[j]-v->cptr[j-1])*(v->rptr[i+1]-v->rptr[i]);
                ++lj;
            }
        assert(lj==v->bptr[i+1]);
        for (j=v->rptr[i];j<v->rptr[i+1];++j) // j is the row
        {
            lk=1;
            for (lj = c->ptr[j];lj<c->ptr[j+1];++lj)
            {
                while (v->cptr[lk]<=c->indx[lj]i)++lk;
                bcnt = (c->indx[lj] - v->cptr[lk-1]) + (j-v->rptr[i])*(v->cptr[lk]-v->cptr[lk-1]); //index
                v->val[bcnt] = c->val[lj];
            }
        }
    }
    // Thus we should have all
    free(k_row);
    free(row);
    free(k_col);
    free(col);
}

int vbr_csr(vbr *v,csr *c)
{
    int cnt = v->indx[v->bptr[v->nr+1]];
    int i,vcnt;
    c->ptr = malloc((v->n+1)*sizeof(int));
    vcnt=0;
    for (i=0;i<cnt;++i)
        if (v->val[i]!=0)
            ++vcnt;
    c->val = malloc(vcnt*sizeof(elem_t));
    c->indx = malloc(vcnt*sizeof(int));
    vcnt = 0;
    c->indx[0]=0;
    for (i=0;i<nr;++i)
    {
        for (r=0;r<v->rptr[i+1] - v->rptr[i];++r)
            for (j=v->bptr[i];j<v->bptr[i];++j)
            {
                int cc = v->bindx[j];
                int col = v->cptr[cc+1]-v->cptr[cc];
                for (k=0;k<col;++k)
                    if (v->val[v->indx[j]+r*col+k]!=0)
                    {
                        c->val[vcnt] = v->val[v->indx[j]+r*col+k];
                        c->indx[vcnt] = v->cptr[cc] + k;
                        ++vcnt;
                    }
            }
        c->indx[v->rptr[i]+r+1] = vcnt;
    }
}
