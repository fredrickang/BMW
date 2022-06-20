#include <stdio.h>
#include <stdlib.h>


int main(void){
    int count = 2097;
    int ret;
    int ***ddarray = (int ***)malloc(sizeof(int **)*1000);
    for(int i = 0; i < 1000; i++){
        ddarray[i] = (int **)malloc(sizeof(int *)*count);
        for(int j = 0; j < count; j++){
            ret = cudaMalloc(&ddarray[i][j], sizeof(int)*128);
            printf("%d,%d: ReT:%d\n", i,j,ret);
        }
    }
    while(1){
    }
}
