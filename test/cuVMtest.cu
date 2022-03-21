#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


int main(void){
    void *d_a[10];
    for(int i = 0; i < 10; i++){
        cudaMalloc(&d_a[i], sizeof(char)*1000);
        printf("%d th address: %p\n",i, d_a[i]);
    }
    printf("%d\n",(int *)d_a[1] -(int *)d_a[0]);

    cudaFree(d_a[1]);
    int *c;
    cudaMalloc(&c, sizeof(int)*1024);
    printf("target address: %p\n", c);

    return 0;
}