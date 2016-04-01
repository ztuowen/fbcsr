#ifndef LIST_H
#define LIST_H

#include"prefix.h"

typedef struct list {
    struct list *nxt;
    void *dat; // this have to be multiplexed
} list;

list *list_add(list *l, void *d);

list *list_next(list *l);   // return null if end of list
void *list_get(list *l);

void *list_destroy(list *l, void (*destroy)(void *d)); // pass null to do nothing

#endif
