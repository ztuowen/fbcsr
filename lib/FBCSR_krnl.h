//
// Created by joe on 4/10/16.
//

#ifndef MATRIXREP_FBCSR_KRNL_H
#define MATRIXREP_FBCSR_KRNL_H

#include"FBCSR.h"

#ifdef __cplusplus
extern "C" {
#endif

void fbcsr_row_krnl_16(fbcsr *f, vector *v, vector *r);

void fbcsr_row_krnl_32(fbcsr *f, vector *v, vector *r);

void fbcsr_col_krnl_16(fbcsr *f, vector *v, vector *r);

void fbcsr_col_krnl_32(fbcsr *f, vector *v, vector *r);

void fbcsr_fslash_krnl_16(fbcsr *f, vector *v, vector *r);

void fbcsr_fslash_krnl_32(fbcsr *f, vector *v, vector *r);

void fbcsr_bslash_krnl_16(fbcsr *f, vector *v, vector *r);

void fbcsr_bslash_krnl_32(fbcsr *f, vector *v, vector *r);

void fbcsr_square_krnl(fbcsr *f, vector *v, vector *r);

#ifdef __cplusplus
}
#endif
#endif //MATRIXREP_FBCSR_KRNL_H
