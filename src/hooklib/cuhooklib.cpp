#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>

#include <math.h>
#include <cuda.h>

#include <iostream>
#include <list>
#include <map>
#include <algorithm>
#include <cassert>
#include <climits>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "hooklib.hpp"

using namespace std;

size_t swap_out_sz_tot = 0;
size_t swap_in_sz_tot = 0;
double swap_out_time = 0.0;
double swap_in_time = 0.0;

static size_t SWAPOUT_SIZE = 0;


/* =====CUDA Hooking APIs===== */

cudaError_t cudaMalloc(void **devPtr, size_t size){   
    cudaError_t err;
    if(!init){
        Init();
        init = 1;
        return cudaSuccess;
    }

    DEBUG_PRINT(BLUE "cudaMalloc [%d]\n" RESET, size);

    SendRequest(*devPtr, _cudaMalloc_, size);
    err = lcudaMalloc(devPtr, size);
    add_entry(&gpu_entry_list, entry_index, *devPtr, size);
    entry_index++;
    
    if(pagetable.size() != 0){ /* app-level page table exist */
        void *new_address = *devPtr;
        void *old_address = (void *)fake_address;
        fake_address--;
        pagetable[old_address] = new_address;
        *devPtr = old_address;
        DEBUG_PRINT_PAGETABLE();
    }
    return err;
}

cudaError_t cudaMalloc(void **devPtr, size_t size, int index){   
    cudaError_t err;
    if(!init){
        Init();
        init = 1;
    }

    DEBUG_PRINT(BLUE "cudaMalloc [%d]\n" RESET, size);

    SendRequest(*devPtr, _cudaMalloc_, size, index);
    err = lcudaMalloc(devPtr, size);
    add_entry(&gpu_entry_list, index, *devPtr, size);
    return err;
}

cudaError_t cudaFree(void* devPtr){ /* free */
    
    if(pagetable.size() != 0 && pagetable.find(devPtr) != pagetable.end()){
        void *new_address = pagetable[devPtr];
        pagetable.erase(devPtr);
        
        devPtr = new_address;
        DEBUG_PRINT(RED"cudaFree: %p\n"RESET,devPtr);
        DEBUG_PRINT_PAGETABLE();
    }

    DEBUG_PRINT(BLUE "cudaFree\n" RESET);
    SendRequest(devPtr, _cudaFree_, 0);
    del_entry(&gpu_entry_list, devPtr);

    return lcudaFree(devPtr);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind){
    void* remapped_dst = dst;
    const void * remapped_src = src;

    if(pagetable.size() != 0){
        if(kind == cudaMemcpyHostToDevice){
            if (pagetable.find(dst) != pagetable.end()) {
                remapped_dst = pagetable[dst];
                DEBUG_PRINT(RED "Re-directed %p -> %p\n" RESET, dst, remapped_dst);
            }else{
                remapped_dst = check_pointer_arithmetic(dst, __func__);
            }
        }else{
            if (pagetable.find((void *)src) != pagetable.end() ){
                remapped_src = (const void*)pagetable[(void *)src];
                DEBUG_PRINT(RED "Re-directed %p -> %p\n" RESET, src, remapped_src);
            }else{
                remapped_src = (const void *)check_pointer_arithmetic((void *)src, __func__);
            }
        }
    }
    cudaError_t err = cudaSuccess;
    CHECK_CUDA(lcudaMemcpy(remapped_dst, remapped_src, size, kind));
    return err;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream){
    void* remapped_dev = devPtr;
    if(pagetable.size() != 0 && pagetable.find(devPtr) != pagetable.end()){
        remapped_dev = pagetable[devPtr];
        DEBUG_PRINT(RED "Re-directed %p -> %p\n" RESET, devPtr, remapped_dev);
    }else{
        remapped_dev = check_pointer_arithmetic(devPtr, __func__);
    }
    cudaError_t err = cudaSuccess;
    CHECK_CUDA(lcudaMemsetAsync(remapped_dev, value, count, stream));
    return err;
}
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc){
    void * remapped_A = (void *)A;
    void * remapped_B = (void *)B;
    void * remapped_C = (void *)C;
    if(pagetable.size() != 0){
        if(pagetable.find((void *)A) != pagetable.end()){
            remapped_A = pagetable[(void *)A];
        }else{
            remapped_A = check_pointer_arithmetic((void *)A, __func__);
        }
        if(pagetable.find((void *)B) != pagetable.end()){
            remapped_B = pagetable[(void *)B];
        }else{
            remapped_B = check_pointer_arithmetic((void *)B, __func__);
        }
        if(pagetable.find((void *)C) != pagetable.end()){
            remapped_C = pagetable[(void *)C];
        }else{
            remapped_C = check_pointer_arithmetic((void *)C, __func__);
        }
    }

    cublasStatus_t err = lcublasSgemm(handle, transa, transb, m, n, k, alpha, (const float *)remapped_A, lda, (const float *)remapped_B, ldb, (const float *)beta, (float *)remapped_C, ldc);
    
    return err;
}


cudaError_t cudaLaunchKernel( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream){
    cudaError_t err;
    if(pagetable.size() != 0){   
        int arg_size = *(int *)args[0];

        int index;
        int pointer_bit[arg_size];
        
        
        for (int i = 0; i < arg_size; i++){
            index = i+1;
            pointer_bit[i] = *(int *)args[index];
        }
     
        for(int i = 0; i < arg_size; i++){
            index = i + arg_size + 1;
            if(pagetable.find(*(void **)args[index]) != pagetable.end()){
                *(void **)args[index] = pagetable[*(void **)args[index]];
            }
            else{
                if(pointer_bit[i] == 1){
                    void * correct = check_pointer_arithmetic(*(void **)args[index], __func__);
                    *(void **)args[index] = correct;
                }
            }
        }
    }

    err = cudaSuccess;
    CHECK_CUDA(lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
    return err;
}


/* =====BMW core===== */

void* swapThread(void *vargsp){

    sigset_t sigsetmask;
    int signum, ack;

    sigemptyset(&sigsetmask);
    sigaddset(&sigsetmask, SIGUSR1);
    sigaddset(&sigsetmask, SIGUSR2);
    sigaddset(&sigsetmask, SIGTERM);

    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));
    msg->type = _Done_;
    msg->entry_index = -1;
    msg->size = 0;

    while(1){
        if(sigwait(&sigsetmask, &signum) > 0){
            DEBUG_PRINT(RED "SIGWAIT Error\n" RESET);
            exit(-1);
        }
        if(signum == SIGUSR1){            
            swapout(signum);
            CHECK_COMM(write(request_fd, msg, sizeof(req_msg)));
            DEBUG_PRINT(GREEN "Swap-out Complete\n" RESET);
            SWAP_OUT = true; // swapped flag on
        }
        if(signum == SIGUSR2){
            if(SWAP_OUT) swapin(signum);
            CHECK_COMM(write(request_fd, msg, sizeof(req_msg)));
            DEBUG_PRINT(GREEN "Swap-in Complete\n" RESET);
            if(SWAPOUT_SIZE == 0) SWAP_OUT = false;   // swapped flag off
        }
        if(signum == SIGTERM){
            DEBUG_PRINT(GREEN "Swap Thread Terminating\n" RESET);
            break;
        }
    }

}

/* Swap-in handler */
void swapin(int signum){
    DEBUG_PRINT(GREEN "Swap-in (SIGUSR2) handler callback\n" RESET);
    
    in_msg *msg = (in_msg *)malloc(sizeof(in_msg));
    CHECK_COMM(read(decision_fd, msg, sizeof(in_msg)));
    DEBUG_PRINT(GREEN"MSG type(%d) Size(%lu)\n"RESET, msg->type, msg->size);

    size_t swapin_size, process_size;
    swapin_size = msg->type == 1 ? SWAPOUT_SIZE : msg->size;
    process_size = 0;
    
    
    void *dummy;
    SendRequest(dummy,_SWAPIN_,swapin_size);
    DEBUG_PRINT(GREEN"Space(%lu) reservation done\n"RESET, swapin_size);
    
    list<int> elist;
    for(auto iter = swap_entry_list.rbegin(); iter != swap_entry_list.rend(); ++iter){
        int index = iter->first;
        size_t size = iter->second.size;
        void* new_address;
        void* old_address = iter->second.gpu_address;
        void* orig_address = iter->second.origin_address;
        char* hostPtr = (char *)iter->second.cpu_address;

        elist.push_back(index);
        CHECK_CUDA(lcudaMalloc(&new_address, size));
        CHECK_CUDA(lcudaMemcpy(new_address, hostPtr, size, cudaMemcpyHostToDevice));

        SendRequest(old_address,_cudaMalloc_, size, index);
        
        add_entry(&gpu_entry_list, index, orig_address, size);

        free(hostPtr);
        DEBUG_PRINT(GREEN "Swap in Addr: %p, Size: %lu\n" RESET, new_address, size);
        
        /* page table update */
        process_size += size;
        pagetable[orig_address] = new_address;
        if(process_size >= swapin_size) break;        
    } 
    
    SWAPOUT_SIZE -= process_size;

    for(auto iter = elist.begin(); iter != elist.end(); iter++){
        swap_entry_list.erase(swap_entry_list.find(*iter));
    }
    
    DEBUG_PRINT_SWAP();
    DEBUG_PRINT_ENTRY();
    DEBUG_PRINT_PAGETABLE();
}

/* Swap out handler */
void swapout(int signum){
    DEBUG_PRINT(GREEN "Swap-out (SIGUSR1) handler callback\n" RESET);

    int ack;
    cudaError_t err;
    
    evict_msg *msg = (evict_msg *)malloc(sizeof(evict_msg));
    CHECK_COMM(read(decision_fd, msg, sizeof(evict_msg)));
    DEBUG_PRINT(GREEN "Swap-out Range [%d, %d]\n" RESET,msg->start_idx, msg->end_idx);
    
    for(int i = msg->start_idx; i <=msg->end_idx; i++){
        int index = i;
        size_t size = gpu_entry_list[index].size;
        void * oldaddress = gpu_entry_list[index].address;
        void * newaddress = oldaddress;

        if (pagetable.size() != 0 && pagetable.find(oldaddress)!=pagetable.end()){
            newaddress = pagetable[oldaddress];
        }
        
        void * hostPtr = (char *)malloc(size);
        CHECK_CUDA(lcudaMemcpy(hostPtr, newaddress, size, cudaMemcpyDeviceToHost));

        add_swap_entry(&swap_entry_list, index, oldaddress, newaddress, hostPtr, size);
        CHECK_CUDA(lcudaFree(newaddress));
        SendRequest(oldaddress, _cudaFree_, 0);
        del_entry(&gpu_entry_list, oldaddress);    
        SWAPOUT_SIZE += size;
        
        DEBUG_PRINT(GREEN "Swap out Addr: %p, Size: %d\n" RESET, newaddress, size);   
        if(pagetable.size() != 0){
            pagetable.erase(oldaddress);
        }
    }
    DEBUG_PRINT_ENTRY();
    DEBUG_PRINT_SWAP();
    DEBUG_PRINT_PAGETABLE();
}

void * check_pointer_arithmetic(void *devPtr, const char* func_name){
    void *retPtr = devPtr;
    int closest_idx = floorSearch(devPtr);
    if(!exist_in_entry(&gpu_entry_list, devPtr) && closest_idx != -1){
        void * base_pointer = gpu_entry_list[closest_idx].address;
        if(pagetable.find(base_pointer) != pagetable.end()){
            size_t dist = (char *)devPtr - (char *)base_pointer;
            if(dist < gpu_entry_list[closest_idx].size){                    
                void * corrected = (char *)pagetable[base_pointer] + dist;
                retPtr = (void *)corrected;
            }
        }
    }
    return retPtr;
}

/* =====BMW Interface===== */

void Init(){

    sigset_t sigsetmask_main;
    sigfillset(&sigsetmask_main);
    sigdelset(&sigsetmask_main, SIGINT);
    pthread_sigmask(SIG_SETMASK, &sigsetmask_main, NULL);

    char request[30];
    char decision[30];

    snprintf(request, 30, "/tmp/mm_request_%d",getpid());
    snprintf(decision, 30, "/tmp/mm_decision_%d",getpid());

    while((request_fd = open(request,O_WRONLY)) < 0);
    while((decision_fd = open(decision,O_RDONLY)) < 0);
    DEBUG_PRINT(BLUE "Request/Decision channel opened\n" RESET);

    atexit(Cleanup);
    DEBUG_PRINT(BLUE "Termination function registered\n" RESET);
    
    pthread_create(&swap_thread_id, NULL, swapThread, NULL);
    DEBUG_PRINT(BLUE "Generating Swap Threads\n" RESET);

    DEBUG_PRINT(GREEN "==Initialization Sequence Done==\n" RESET);
}

void Cleanup(){
    DEBUG_PRINT(BLUE "Cleaning up...\n" RESET);
    int ack;
    
    pthread_kill(swap_thread_id, SIGTERM);
    pthread_join(swap_thread_id, NULL);
    DEBUG_PRINT(BLUE "Swap Thread terminated\n" RESET);
    DEBUG_PRINT(GREEN "==BMW Termination Sequence Done==\n" RESET);
}

int SendRequest(void* devPtr, cudaAPI type, size_t size){
    DEBUG_PRINT_ENTRY();
    
    int ack;
    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));
    
    msg -> type = type;
    msg -> size = size;
    
    if(type == _cudaMalloc_)  msg -> entry_index = entry_index;
    if(type == _cudaFree_)  msg -> entry_index = find_index_by_ptr(&gpu_entry_list, devPtr);
    if(type == _SWAPIN_) msg -> entry_index = -1;
    
    CHECK_COMM(write(request_fd, msg, sizeof(req_msg)));
    CHECK_COMM(read(decision_fd, &ack, sizeof(int)));
}

int SendRequest(void* devPtr, cudaAPI type, size_t size, int index){
    DEBUG_PRINT_ENTRY();
    
    int ack;
    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));
    
    msg -> type = type;
    msg -> size = size;
    
    if(type == _cudaMalloc_)  msg -> entry_index = index;
    if(type == _cudaFree_)  msg -> entry_index = find_index_by_ptr(&gpu_entry_list, devPtr);
    
    CHECK_COMM(write(request_fd, msg, sizeof(req_msg)));
    CHECK_COMM(read(decision_fd, &ack, sizeof(int)));
}

void add_entry(map<int,entry> *entry_list, int index, void* devPtr, size_t size){
    DEBUG_PRINT(BLUE "Add: {%d, [%p, %d]}\n" RESET, index, devPtr, size);
    entry tmp;
    tmp.address = devPtr;
    tmp.size = size;
    (*entry_list).insert({index, tmp});
}

void del_entry(map<int,entry> *entry_list, void* devPtr){
    DEBUG_PRINT(BLUE "Del: %p\n" RESET, devPtr);
    (*entry_list).erase(find_index_by_ptr(entry_list, devPtr));
}


int find_index_by_ptr(map<int,entry> *entry_list, void* ptr){
    auto iter = (*entry_list).begin();

    while(iter != (*entry_list).end() && iter->second.address != ptr ){
        ++iter;
    }

    if(iter ==(*entry_list).end() && iter->second.address != ptr) {
        DEBUG_PRINT(RED "Can't find ptr inside entry\n" RESET);
        exit(-1);
    }
    return iter->first;
}

bool exist_in_entry(map<int,entry> *entry_list, void *ptr){
    auto iter = (*entry_list).begin();
    while(iter != (*entry_list).end() && iter->second.address != ptr){
        ++ iter;
    }
    if (iter == (*entry_list).end() && iter->second.address != ptr) return false;
    return true;
}

void add_swap_entry(map<int,gswap>* entry_list, int index, void* origPrt, void* gpuPtr, void* cpuPtr, size_t size){
    DEBUG_PRINT(BLUE "Add (Swap): {%d, [%p, %p, %p, %lu]}\n" RESET, index, origPrt, gpuPtr, cpuPtr, size);
    gswap tmp;
    tmp.origin_address = origPrt;
    tmp.gpu_address = gpuPtr;
    tmp.cpu_address = cpuPtr;
    tmp.size = size;
    (*entry_list).insert({index, tmp});
}


/*  ===== Utils ===== */

#ifdef DEBUG2
void DEBUG_PRINT_ENTRY(){
    DEBUG_PRINT(BLUE "Current GPU Entry: ");
    auto iter = gpu_entry_list.begin();
    while(iter != gpu_entry_list.end()){
        fprintf(stderr, "{%d, [%p, %d]} ",iter->first, iter->second.address, iter->second.size);
        ++iter;
    }
    fprintf(stderr,"\n" RESET);
}
#else
void DEBUG_PRINT_ENTRY(){

}
#endif

#ifdef DEBUG2
void DEBUG_PRINT_SWAP(){
    DEBUG_PRINT(BLUE "Current SWAP Entry: ");
    auto iter = swap_entry_list.begin();
    while(iter != swap_entry_list.end()){
        fprintf(stderr, "{%d, [%p, %p, %d]} ",iter->first, iter->second.gpu_address, iter->second.cpu_address , iter->second.size);
        ++iter;
    }
    fprintf(stderr,"\n" RESET);
}
#else
void DEBUG_PRINT_SWAP(){

}
#endif


#ifdef DEBUG
void DEBUG_PRINT_PAGETABLE(){
    DEBUG_PRINT(GREEN "Current Page Table Entry: ");
    auto iter = pagetable.begin();
    while(iter != pagetable.end()){
        fprintf(stderr,"{Old: %p, New: %p}",iter->first, iter->second);
        ++iter;
    }
    fprintf(stderr,"\n" RESET);
}
#else
void DEBUG_PRINT_PAGETABLE(){

}
#endif


float checksum(float * input, int size){
    int items = size/sizeof(float);
    float sum = 0;
    for(int i = 0; i < items; i++){
        sum += input[i];
    }
    return sum;
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

char * getcudaAPIString(cudaAPI type){
    switch (type){
        case _cudaMalloc_:
            return string(_cudaMalloc_);
        case _cudaFree_:
            return string(_cudaFree_);
    }
}


int floorSearch(void * addr){
    int floor_idx = -1;
    unsigned long long int floor_dist = LLONG_MAX;

    for(auto iter = gpu_entry_list.begin(); iter !=gpu_entry_list.end(); iter++){
        int index = iter->first;
        void * key = iter->second.address;
        if(key < addr && floor_dist > ((char *)addr - (char *)key)){
            floor_idx = index;
            floor_dist = ((char *)addr - (char *)key);
        }
    }
    return floor_idx;
}