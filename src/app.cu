#include <stdio.h>
#include <stdlib.h>

#define MAX_ITER 10

int main(void){
    
    int *d_a[MAX_ITER];

    for(int i = 0; i < MAX_ITER; i++){
        cudaMalloc(&d_a[i],sizeof(int)*1000);
        if(i % 2 == 0) cudaFree(d_a[i]);
    }
    
    cudaDeviceSynchronize();
}