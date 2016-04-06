//
// Created by joe on 4/4/16.
//

#include"../UBCSR.h"
#include"cuHelper.h"

void ubcsrSingle_memCpy(ubcsr *src,ubcsr *dst,enum DeviceCopyDIR dir) {

}

extern "C" void ubcsr_memCpy(list *src,list *dst, enum DeviceCopyDIR dir){

}

extern "C" void ubcsr_CUDA_SpMV(list *l, vector *v, vector *r){

}

extern "C" void ubcsr_CUDA_destroy(void *u){
    ubcsr *uu = (ubcsr *) u;
    safeCudaFree(uu->rptr);
    safeCudaFree(uu->val);
    safeCudaFree(uu->bindx);
    safeCudaFree(uu->bptr);
}
