#ifndef PREFIX_H
#define PREFIX_H

#include<stdlib.h>
#include<stdio.h>
#include<assert.h>

#define safeFree(x) {if (x) free(x);}

#define USE_FLOAT
#define DEBUG


// Only if support vargs in macros can this work
// Really complex structure, turn off DEBUG_PRINT when in release
#ifdef NDEBUG
#define DEBUG_PRINT(...) do {}while(0)
#else
#ifdef DEBUG
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_PRINT(...) do {}while(0)
#endif
#endif

#define _FLOAT_PREC (1e-9)
typedef float elem_t;

enum DeviceCopyDIR {
    cpyHostToDevice,
    cpyDeviceToHost
};

#endif
