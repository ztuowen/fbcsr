/*
Author: Tharindu Rusira
Date: 04/05/15
*/
#include "timing.h"

float elapsed_time_msec(struct timespec *begin, struct timespec *end,long *sec, long *nsec)
{
	if(end->tv_nsec < begin->tv_nsec)
	{
		*nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
		*sec = end->tv_sec - begin->tv-sec -1;
	}
	else
	{
		*nsec = end->tv_nsec - begin->tv_nsec;
		*sec = end->tv_sec - begin->tv_sec;
	}
	return (float)(*sec)*1000 + ((float)(*nsec))/1000000;
}
