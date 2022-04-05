#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>

using namespace std;

__global__ void test(int * tmp){
    for(int i = 0; i < 10; i++){
        printf("Say hello\n");
    }
}


int main(void){
    dim3 gridDim;
    dim3 blockDim;
    int * tmp;
    cudaMalloc(&tmp, sizeof(int)*100);
    printf("host space : %p\n", tmp);
    test<<<gridDim, blockDim>>>(tmp);
    void *args[] = {(void*)tmp};
    cudaLaunchKernel((void*)test,gridDim,blockDim,args,0,NULL);

    return 0;
}