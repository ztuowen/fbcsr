#ifndef PREFIX_H
#define PREFIX_H

#include<stdlib.h>

#define safeFree(x) {if (x) free(x);}

#define USE_FLOAT
#define DEBUG 0

#ifdef USE_FLOAT
#define _FLOAT_PREC (1e-9)
typedef float elem_t;
#else
typedef int elem_t;
#endif

#endif
