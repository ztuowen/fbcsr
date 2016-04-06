//
// Created by joe on 4/5/16.
//

#ifndef MATRIXREP_CUHELPER_H_H
#define MATRIXREP_CUHELPER_H_H

#include "../prefix.h"

#define cuCheck(x) { cudaError e = (x); if (e!=cudaSuccess) {printf("Error: cudaStat %d, %s\n", e, cudaGetErrorString(e)); exit(-1);} }
#define safeCudaFree(x) {if (x) cuCheck(cudaFree(x));}
#define cuSparseCheck(x) { cusparseStatus_t e = (x); if (e!=CUSPARSE_STATUS_SUCCESS) {fprintf(stderr,"CUDA CALL FAILED");exit(-1);} }

void memCopy(void **dst, void *src, size_t size, enum DeviceCopyDIR dir);

#endif //MATRIXREP_CUHELPER_H_H
