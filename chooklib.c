#define _GNU_SOURCE

#include <stdio.h>
#include <dlfcn.h>

#include "hooklib.hpp"

static void* (*lmalloc)(size_t)=NULL;
static void (*lfree)(void *)=NULL;

static void mtrace_init(void)
{
    lmalloc = dlsym(RTLD_NEXT, "malloc");
    if (NULL == lmalloc) DEBUG_PRINT("Error in `dlsym`: %s\n", dlerror());
    

    lfree = dlsym(RTLD_NEXT, "free");
    if (NULL == lfree) DEBUG_PRINT("Error in 'dlsym': %s\n", dlerror());

}

void *malloc(size_t size)
{
    if(lmalloc==NULL) {
        mtrace_init();
    }

    void *p = NULL;
    p = lmalloc(size);

    //DEBUG_PRINT("malloc [%d]\n",size);

    return p;
}

void free(void* Ptr){
    if(lfree==NULL){
        mtrace_init();
    }
    //DEBUG_PRINT("free\n");
    lfree(Ptr);
}