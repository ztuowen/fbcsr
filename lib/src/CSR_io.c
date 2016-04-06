#include "../CSR.h"

void csr_readFile(char *filename, csr *c) {
    FILE *in;
    char data[1024];
    int i;
    int ret;
    in = fopen(filename, "r");
    if (in == NULL) {
        fprintf(stderr, "something might be wrong with the file\n");
        exit(-1);
    }
    fgets(data, 1024, in);
    DEBUG_PRINT("Matrix: %s", data);
    ret = fscanf(in, "%d %d %d\n", &(c->n), &(c->m), &(c->nnz));
    assert(ret == 3);
    DEBUG_PRINT("load_sparse_matrix:: rows = %d, cols= %d nnz = %d\n", c->n, c->m, c->nnz);

    c->ptr = (int *) malloc(sizeof(int) * ((c->n) + 1));
    c->indx = (int *) malloc(sizeof(int) * c->nnz);
    c->val = malloc(sizeof(elem_t) * c->nnz);
    DEBUG_PRINT("load_sparse_matrix::reading row index\n");

    for (i = 0; i <= c->n; i++) {
        int temp;
        ret = fscanf(in, "%d", &temp);
        assert(ret == 1);
        --temp;
        c->ptr[i] = temp;

        assert(temp <= c->nnz);
    }

    DEBUG_PRINT("load_sparse_matrix::reading column index\n");

    for (i = 0; i < c->nnz; i++) {
        int temp;
        ret = fscanf(in, "%d", &temp);
        assert(ret == 1);
        temp--;
        c->indx[i] = temp;
        assert(temp >= 0);
    }

    DEBUG_PRINT("load_sparse_matrix::reading values\n");

    for (i = 0; i < c->nnz; i++) {
        // should I use a separate function to multiplex
        ret = fscanf(in, "%f", &(c->val[i]));
        assert(ret == 1);
    }

    DEBUG_PRINT("load_sparse_matrix::Loading sparse matrix done\n");
}

