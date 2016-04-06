#include"cuHelper.h"

void memCopy(void *src,void **dst,int numBytes,enum DeviceCopyDIR dir){
    switch (dir){
        case cpyHostToDevice:
            cuCheck(cudaMalloc(dst,numBytes));
            cuCheck(cudaMemcpy(*dst,src,numBytes,cudaMemcpyHostToDevice));
            break;
        case cpyDeviceToHost:
            *dst = malloc(numBytes);
            cuCheck(cudaMemcpy(*dst,src,numBytes,cudaMemcpyDeviceToHost));
            break;
        default:
            fprintf(stderr,"Unexpected memcpy");
            exit(-1);
    }
}
