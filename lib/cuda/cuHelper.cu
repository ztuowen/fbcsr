#include"cuHelper.h"

void memCopy(void **dst, void *src, size_t numBytes, enum DeviceCopyDIR dir) {
    switch (dir) {
        case cpyHostToDevice: cuCheck(cudaMalloc(dst, numBytes));
            cuCheck(cudaMemcpy(*dst, src, numBytes, cudaMemcpyHostToDevice));
            break;
        case cpyDeviceToHost:
            *dst = malloc(numBytes);
            cuCheck(cudaMemcpy(*dst, src, numBytes, cudaMemcpyDeviceToHost));
            break;
        default:
            fprintf(stderr, "Unexpected memcpy");
            exit(-1);
    }
}
