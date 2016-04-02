//
// Created by joe on 4/1/16.
//

#ifndef MATRIXREP_RANDOM_GEN_H_H
#define MATRIXREP_RANDOM_GEN_H_H

#include"prefix.h"
#include"vector.h"

void vector_gen_random(vector *v, int n, elem_t (*random)(void));

#endif //MATRIXREP_RANDOM_GEN_H_H
