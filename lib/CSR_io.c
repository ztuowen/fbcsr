#include "CSR.h"
#include <stdio.h>
#include <stdlib.h>
#include<string.h>
#include<assert.h>


void load_sparse_matrix(char *filename, csr *c) {
    FILE *in;
    char data[1024];
    int i;
    int ret;
    int nnz;
    char *tmp;
    in = fopen(filename, "r");
    if (in == NULL) {
        printf("something might be wrong with the file\n");
    }
    tmp = fgets(data, 1024, in);
    fprintf(stderr, "%s", data);    // TODO What is this fprintf for, Debugging?
    ret = fscanf(in, "%d %d %d\n", &(c->n), &(c->m), &nnz); // TODO I just ignore nnz for now
    if (DEBUG)
        fprintf(stderr, "load_sparse_matrix:: rows = %d, cols= %d nnz = %d\n", A->nrows, A->ncols, nnz);

    c->ptr = (int *) malloc(sizeof(int) * ((c->n) + 1));
    c->indx = (int *) malloc(sizeof(int) * (c->m));
    c->val = malloc(sizeof(elem_t) * nnz);
    if (DEBUG)
        fprintf(stderr, "load_sparse_matrix::reading row index\n");

    for (i = 0; i <= c->n; i++) {
        int temp;
        ret = fscanf(in, "%d", &temp);
        --temp;
        c->ptr[i] = temp;

        assert(temp<=nnz);
    }

    if (DEBUG)
        fprintf(stderr, "load_sparse_matrix::reading column index\n");

    for (i = 0; i < nnz; i++) {
        int temp;
        ret = fscanf(in, "%d", &temp);
        temp--;
        c->indx[i] = temp;
        assert(temp>=0);
    }

    if (DEBUG)
        fprintf(stderr, "load_sparse_matrix::reading values\n");

    for (i = 0; i < nnz; i++) {
        // should I use a separate function to multiplex
#ifdef USE_FLOAT
        ret = fscanf(in, "%f", &(c->val[i]));
#else
        ret = fscanf(in, "%d", &(c->val[i]));
#endif
    }

    if (DEBUG)
        fprintf(stderr, "load_sparse_matrix::Loading sparse matrix done\n");
}

