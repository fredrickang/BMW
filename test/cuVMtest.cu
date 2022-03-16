#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


int main(void){
    int ret;
    int *a, *d_a, *d_b, *b;
    a = (int *)malloc(sizeof(int)*10);
    b = (int *)malloc(sizeof(int)*10);

    cudaMalloc(&d_a, sizeof(int)*10);
    
    for(int i = 0; i < 10; i++) a[i] = i;

    cudaMemcpy(d_a, a, sizeof(int)*10, cudaMemcpyHostToDevice);
    d_b = d_a;

    cudaFree(d_a);
    cudaMalloc(&d_b, sizeof(int)*10);
    cudaMemcpy(d_b, a, sizeof(int)*10, cudaMemcpyHostToDevice);

    ret = cudaMemcpy(b, d_a, sizeof(int)*10, cudaMemcpyDeviceToHost);
    printf("cudamemcpy ret: %d\n", ret);
    for(int i =0; i < 10; i++){
        printf("%d,",b[i]);
    }
    printf("\n");

    return 0;
}