//
// Created by joe on 4/1/16.
//

#ifndef MATRIXREP_RANDOM_GEN_H_H
#define MATRIXREP_RANDOM_GEN_H_H

#include"prefix.h"
#include"vector.h"

#ifdef __cplusplus
extern "C" {
#endif

void vector_gen(vector *v, int n, elem_t (*random)(void));

#ifdef __cplusplus
}
#endif
#endif //MATRIXREP_RANDOM_GEN_H_H
