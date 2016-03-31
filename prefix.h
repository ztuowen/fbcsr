#ifndef _PREFIX_H
#define _PREFIX_H

#define safeFree(x) {if (x) free(x);}

// #define USE_FLOAT

#ifdef USE_FLOAT
#define _FLOAT_PREC (1e-9)
#endif

typedef int elem_t;

#endif
