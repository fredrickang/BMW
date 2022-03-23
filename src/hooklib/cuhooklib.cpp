#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <signal.h>
#include <map>
#include <vector>
#include <list>
#include <pthread.h>
#include <math.h>
#include <algorithm>

#include "hooklib.hpp"

void * GPU_OFFSET_PTR = NULL;

using namespace std;

cudaError_t cudaMalloc(void **devPtr, size_t size){   
    cudaError_t err;
    if(!init){
        Init();
        init = 1;
    }

    DEBUG_PRINT(BLUE "cudaMalloc [%d]\n" RESET, size);

    SendRequest(*devPtr, _cudaMalloc_, size);
    err = lcudaMalloc(devPtr, size);
    add_entry(&gpu_entry_list, entry_index, devPtr, size);
    entry_index++;
    return err;
}

cudaError_t cudaMalloc(void **devPtr, void **idxPtr, size_t size, int index){   
    cudaError_t err;
    if(!init){
        Init();
        init = 1;
    }

    DEBUG_PRINT(BLUE "cudaMalloc [%d]\n" RESET, size);

    SendRequest(*devPtr, _cudaMalloc_, size, index);
    err = lcudaMalloc(devPtr, size);
    add_entry(&gpu_entry_list, index, idxPtr, size);
    return err;
}

cudaError_t cudaFree(void* devPtr){ /* free */
    
    DEBUG_PRINT(BLUE "cudaFree\n" RESET);
    SendRequest(devPtr, _cudaFree_, 0);
    del_entry(&gpu_entry_list,devPtr);

    return lcudaFree(devPtr);
}

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
        switch(signum){
            case SIGUSR1:
                swapout(signum);
                commErrchk(write(request_fd, msg, sizeof(int)*REQ_MSG_SIZE));
                DEBUG_PRINT(GREEN "Swap-out Complete\n" RESET);
                SWAP_OUT = true; // swapped flag on
                break;
            case SIGUSR2:
                if(SWAP_OUT) swapin(signum);
                commErrchk(write(request_fd, msg, sizeof(int)*REQ_MSG_SIZE));
                DEBUG_PRINT(GREEN "Swap-in Complete\n" RESET);
                SWAP_OUT = false;   // swapped flag off
                break;
            case SIGTERM:
                DEBUG_PRINT(BLUE "Swap Thread Terminating\n" RESET);
                exit(EXIT_SUCCESS);
            default:
                break;
        }
    }

}

void Init(){
    // Block other signals except SIGINT
    sigset_t sigsetmask_main;
    sigfillset(&sigsetmask_main);
    sigdelset(&sigsetmask_main, SIGINT);
    pthread_sigmask(SIG_SETMASK, &sigsetmask_main, NULL);

    if((register_fd = open(REGISTRATION, O_WRONLY)) < 0){
        DEBUG_PRINT(RED "REGISTRATION CHANNEL OPEN FAIL\n" RESET);
        exit(-1);
    }

    reg_msg *reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->reg_type = 1;
    reg->pid = getpid();

    commErrchk(write(register_fd, reg, sizeof(int)*2))
    
    DEBUG_PRINT(BLUE "Registrated\n" RESET);

    char request[30];
    char decision[30];

    snprintf(request, 30, "/tmp/request_%d",getpid());
    snprintf(decision, 30, "/tmp/decision_%d",getpid());

    while((request_fd = open(request,O_WRONLY)) < 0);
    while((decision_fd = open(decision,O_RDONLY)) < 0);
    DEBUG_PRINT(BLUE "Request/Decision channel opened\n" RESET);

    atexit(Cleanup);
    DEBUG_PRINT(BLUE "Termination function registered\n" RESET);
    
    pthread_create(&swap_thread_id, NULL, swapThread, NULL);
    DEBUG_PRINT(BLUE "Generating Swap Threads\n" RESET);

    DEBUG_PRINT(GREEN "==Initialization Sequence Done==\n" RESET);
}

int SendRequest(void* devPtr, cudaAPI type, size_t size){
    DEBUG_PRINT_ENTRY();
    
    int ack;
    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));
    
    msg -> type = type;
    msg -> size = size;
    
    if(type == _cudaMalloc_)  msg -> entry_index = entry_index;
    if(type == _cudaFree_)  msg -> entry_index = find_index_by_ptr(&gpu_entry_list, devPtr);
    
    commErrchk(write(request_fd, msg, sizeof(int)*3));
    commErrchk(read(decision_fd, &ack, sizeof(int)));
}
int SendRequest(void* devPtr, cudaAPI type, size_t size, int index){
    DEBUG_PRINT_ENTRY();
    
    int ack;
    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));
    
    msg -> type = type;
    msg -> size = size;
    
    if(type == _cudaMalloc_)  msg -> entry_index = index;
    if(type == _cudaFree_)  msg -> entry_index = find_index_by_ptr(&gpu_entry_list, devPtr);
    
    commErrchk(write(request_fd, msg, sizeof(int)*3));
    commErrchk(read(decision_fd, &ack, sizeof(int)));
}

#ifdef DEBUG2
void DEBUG_PRINT_ENTRY(){
    DEBUG_PRINT(BLUE "Current GPU Entry: ");
    auto iter = gpu_entry_list.begin();
    while(iter != gpu_entry_list.end()){
        fprintf(stderr, "{%d, [%p, %p, %d]} ",iter->first, iter->second.address, *(iter->second.address), iter->second.size);
        ++iter;
    }
    fprintf(stderr,"\n" RESET);
}
#else
void DEBUG_PRINT_ENTRY(){

}
#endif

#ifdef DEBUG2
void DEBUG_SWAP_PRINT(){
    DEBUG_PRINT(BLUE "Current SWAP Entry: ");
    auto iter = swap_entry_list.begin();
    while(iter != swap_entry_list.end()){
        fprintf(stderr, "{%d, [%p, %p, %d]} ",iter->first, iter->second.gpu_address, iter->second.cpu_address , iter->second.size);
        ++iter;
    }
    fprintf(stderr,"\n" RESET);
}
#else
void DEBUG_SWAP_PRINT(){

}
#endif

void add_entry(map<int,entry> *entry_list, int index, void** devPtr, size_t size){
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

    while(iter != (*entry_list).end() && *(iter->second.address) != ptr ){
        ++iter;
    }

    if(iter ==(*entry_list).end() && *(iter->second.address) != ptr) {
        DEBUG_PRINT(RED "Can't find ptr inside entry\n" RESET);
        exit(-1);
    }
    return iter->first;
}


void Cleanup(){
    DEBUG_PRINT(BLUE "Cleaning up...\n" RESET);
    
    reg_msg *reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->reg_type = 0;
    reg->pid = getpid();
    commErrchk(write(register_fd, reg, sizeof(int)*2));
    DEBUG_PRINT(BLUE "==De-registration done==\n" RESET);
    
    kill(0, SIGTERM);
    pthread_join(swap_thread_id, NULL);
    DEBUG_PRINT(BLUE "Swap Thread terminated\n" RESET);

    DEBUG_PRINT(GREEN "==Termination Sequence Done==\n" RESET);
}

/* Swap in handler */
void swapin(int signum){
    DEBUG_PRINT(GREEN "Swap-in (SIGUSR2) handler callback\n" RESET);
    // iterate swap entry list
    for(auto iter = swap_entry_list.begin(); iter != swap_entry_list.end(); iter++){
        int index = iter->first;
        size_t size = iter->second.size;

        void * devPtr = 0ULL;
        void ** ddevPtr = iter->second.gpu_address;
        char * hosPtr = (char *)iter->second.cpu_address;
        
        cudaMalloc(&devPtr,ddevPtr,size,index);
        cudaMemcpy(devPtr, hosPtr, size, cudaMemcpyHostToDevice);
        *ddevPtr = devPtr;
        free(hosPtr);
    }
    swap_entry_list.clear();
    DEBUG_SWAP_PRINT();
    DEBUG_PRINT_ENTRY();
} 

/* Swap out handler */
void swapout(int signum){
    DEBUG_PRINT(GREEN "Swap-out (SIGUSR1) handler callback\n" RESET);

    int ack;
    cudaError_t err;
    
    evict_msg *msg = (evict_msg *)malloc(sizeof(evict_msg));
    commErrchk(read(decision_fd, msg, sizeof(int)*2));
    
    DEBUG_PRINT(GREEN "Swap-out Range [%d, %d]\n" RESET,msg->start_idx, msg->end_idx);
    
    for(auto iter = gpu_entry_list.cbegin(); iter != gpu_entry_list.cend(); ){
        int index = iter->first;
        size_t size = iter->second.size;
        if(index >= msg->start_idx && index <= msg->end_idx){
            void** ddevPtr = iter->second.address;
            char * hosPtr =(char *)malloc(size);

            CHECK_CUDA(cudaMemcpy(hosPtr, *ddevPtr, size, cudaMemcpyDeviceToHost));
            add_swap_entry(&swap_entry_list, index, ddevPtr, hosPtr, size);
            
            // MAP Erase sucks... it must be done in this way 
            DEBUG_PRINT(BLUE "cudaFree\n" RESET);
            SendRequest(*ddevPtr, _cudaFree_, 0);
            CHECK_CUDA(lcudaFree(*ddevPtr));
            gpu_entry_list.erase(iter++);

            DEBUG_PRINT(GREEN "Swap out Addr: %p, Size: %d\n" RESET, *ddevPtr, size);
        }else{
            ++iter;
        }
    }
    DEBUG_PRINT_ENTRY();
    DEBUG_SWAP_PRINT();
}

void add_swap_entry(map<int,gswap>* entry_list, int index, void** gpuPtr, void* cpuPtr, size_t size){
    DEBUG_PRINT(BLUE "Add (Swap): {%d, [%p, %p, %d]}\n" RESET, index, gpuPtr, cpuPtr, size);
    gswap tmp;
    tmp.gpu_address = gpuPtr;
    tmp.cpu_address = cpuPtr;
    tmp.size = size;
    (*entry_list).insert({index, tmp});
}

char * getcudaAPIString(cudaAPI type){
    switch (type){
        case _cudaMalloc_:
            return string(_cudaMalloc_);
        case _cudaFree_:
            return string(_cudaFree_);
    }
}