#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>

using namespace std;

int main(void){
    
    int *offset;
    cudaMalloc(&offset, sizeof(char));

    void *d_a[10];
    void *a;
    a = (void *)malloc(sizeof(int)*1000);

    for(int i = 0; i < 5; i++){
        cudaMalloc(&d_a[i], sizeof(int)*1000);
        printf("%d th address: %p\n",i, d_a[i]);
    }

    cudaFree(d_a[0]);
    cudaFree(d_a[2]);
    cudaFree(d_a[4]);
    cudaFree(d_a[1]);
    for(int i = 0; i < 5; i++){
    void *c;
    cudaMalloc(&c, sizeof(int));
    printf("target address: %p\n", c);
    }
    printf("dummy %p\n",d_a[0]);
    printf("dummy %p\n",d_a[2]);
    printf("dummy %p\n",d_a[4]);
    return 0;
}