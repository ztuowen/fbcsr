/*
Author: Tharindu Rusira
Date: 04/05/15
*/
#ifndef TIMING_H
#define TIMING_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define GET_TIME(x) if(clock_gettime(CLOCK_MONOTONIC,&(x))<0){perror("clock_gettime():");exit(EXIT_FAILURE);}

float elapsed_time_msec(struct timespec *begin, struct timespec *end, long *sec,long *nsec);

#endif
