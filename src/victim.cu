#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>

#define MAX_ITER 10

int main(void){
    sigset_t myset;
    (void) sigemptyset(&myset);
    int *d_a[MAX_ITER];

    for(int i = 0; i < MAX_ITER; i++){
        cudaMalloc(&d_a[i],sizeof(int)*1000);
        if(i % 2 == 0) cudaFree(d_a[i]);
    }
    
    cudaDeviceSynchronize();
    while(1){
        sigsuspend(&myset);
    }
}