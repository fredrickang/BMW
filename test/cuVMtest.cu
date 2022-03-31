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
    
    int *d_a[10];
    int *a[10];
   
    for(int i = 0; i < 10; i++){
        a[i] = (int *)malloc(sizeof(int)*1000);
        for(int j = 0; j < 1000; j++) a[i][j] = i;
        cudaMalloc(&d_a[i], sizeof(int)*1000);
        cudaMemcpy(d_a[i], a, sizeof(int)*1000, cudaMemcpyHostToDevice);
        printf("%d th address: %p\n",i, d_a[i]);
    }

    // for(int i = 0; i <10; i++){
    //     if(i % 2 == 0) cudaFree(d_a[i]);
    // }

    // cudaFree(d_a[0]);
    // cudaFree(d_a[2]);
    // cudaFree(d_a[4]);
    
    for(int i = 0; i < 5; i++){
        void *c;
        cudaMalloc(&c, sizeof(int));
        printf("target address: %p\n", c);
        cudaFree(c);
    }
    
    return 0;
}