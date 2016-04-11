/*
Author: Tharindu Rusira
Date: 04/05/15
*/

#include "../timing.h"

float elapsed_time_msec(struct timespec *begin, struct timespec *end, long *sec,long *nsec)
{
	if(end->tv_nsec < begin->tv_nsec)
	{
		*nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
		*sec = end->tv_sec - begin->tv_sec -1;
	}
	else
	{
		*nsec = end->tv_nsec - begin->tv_nsec;
		*sec = end->tv_sec - begin->tv_sec;
	}
	return (float)(*sec)*1000 + ((float)(*nsec))/1000000;
}


/*
Timing code example
-------------------------------------------------------
#include <stdio.h>
#include <>
struct timespec s,e;
long sec, nsec;
float time;

GET_TIME(s);
// do your work
GET_TIME(e);

time = elapsed_time_msec(&s,&e,&sec, &nsec);
-------------------------------------------------------
Compiling
-------------------------------------------------------
gcc src/timing.c <other_source_files> <options>
*/