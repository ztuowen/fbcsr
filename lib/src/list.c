#include "../list.h"

list *list_add(list *l, void *d) {
    list *nl = malloc(sizeof(list));
    nl->nxt = l;
    nl->dat = d;
    return nl;
}

list *list_next(list *l) {
    return l->nxt;
}

void *list_get(list *l) {
    return l->dat;
}

void list_destroy(list *l, void (*destory)(void *)) {
    list *nl;
    void *dat;
    while (l != NULL) {
        nl = l;
        l = list_next(l);
        dat = list_get(nl);
        if (destory != NULL)
            destory(dat);
        free(dat);
        free(nl);
    }
}
