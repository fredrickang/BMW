#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <signal.h>
#include <map>
#include <list>
#include <pthread.h>

#include "hooklib.hpp"

using namespace std;

cudaError_t cudaMalloc (void **devPtr, size_t size){   
    cudaError_t err;
    if(!init){
        Init();
        init = 1;
    }

    DEBUG_PRINT("cudaMalloc [%d]\n", size);

    SendRequest((const void *)*devPtr, _cudaMalloc_, size);
    err = lcudaMalloc(devPtr, size);
    add_entry(&gpu_entry_list, entry_index, (const void *)*devPtr, size);
    entry_index++;
    return err;
}

cudaError_t cudaFree(void* devPtr){ /* free */
    
    DEBUG_PRINT("cudaFree\n");
    SendRequest((const void *)devPtr, _cudaFree_, 0);
    del_entry(&gpu_entry_list,(const void *)devPtr);

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
            DEBUG_PRINT(RED"SIGWAIT Error\n"RESET);
            exit(-1);
        }
        switch(signum){
            case SIGUSR1:
                swapout(signum);
                commErrchk(write(request_fd, &msg, sizeof(int)*REQ_MSG_SIZE));
                SWAP_OUT = true; // swapped flag on
                break;
            case SIGUSR2:
                if(SWAP_OUT){
                    swapin(signum);
                } 
                commErrchk(write(request_fd, &msg, sizeof(int)*REQ_MSG_SIZE));
                SWAP_OUT = false;   // swapped flag off
                break;
            case SIGTERM:
                DEBUG_PRINT("Swap Thread Terminating\n");
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
        DEBUG_PRINT("\x1b[31m""REGISTRATION CHANNEL OPEN FAIL\n""\x1b[0m");
        exit(-1);
    }

    reg_msg *reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->reg_type = 1;
    reg->pid = getpid();

    commErrchk(write(register_fd, reg, sizeof(int)*2))
    
    DEBUG_PRINT("Registrated\n");

    char request[30];
    char decision[30];

    snprintf(request, 30, "/tmp/request_%d",getpid());
    snprintf(decision, 30, "/tmp/decision_%d",getpid());

    while((request_fd = open(request,O_WRONLY)) < 0);
    while((decision_fd = open(decision,O_RDONLY)) < 0);
    DEBUG_PRINT("Request/Decision channel opened\n");

    atexit(Cleanup);
    DEBUG_PRINT("Termination function registered\n");
    
    pthread_create(&swap_thread_id, NULL, swapThread, NULL);
    DEBUG_PRINT("Generating Swap Threads\n");

    DEBUG_PRINT_ENTRY();
}

int SendRequest(const void* devPtr, cudaAPI type, size_t size){
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

#ifdef DEBUG
void DEBUG_PRINT_ENTRY(){
    DEBUG_PRINT("Current Entry: ");
    auto iter = gpu_entry_list.begin();
    while(iter != gpu_entry_list.end()){
        fprintf(stderr, "{%d, [%p, %d]} ",iter->first, iter->second.address, iter->second.size);
        ++iter;
    }
    fprintf(stderr,"\n");
}
#else
void DEBUG_PRINT_ENTRY(){

}
#endif

void add_entry(map<int,entry> *entry_list, int index, const void* devPtr, size_t size){
    DEBUG_PRINT("add entry: {%d, [%p, %d]}\n", index, devPtr, size);
    entry tmp;
    tmp.address = devPtr;
    tmp.size = size;
    (*entry_list).insert({index, tmp});
}

void del_entry(map<int,entry> *entry_list, const void* devPtr){
    DEBUG_PRINT("del entry: %p\n", devPtr);
    (*entry_list).erase(find_index_by_ptr(entry_list, devPtr));
}


int find_index_by_ptr(map<int,entry> *entry_list, const void* ptr){
    // DEBUG_PRINT_ENTRY();
    auto iter = (*entry_list).begin();

    while(iter != (*entry_list).end() && iter->second.address != ptr ){
        ++iter;
    }

    if(iter ==(*entry_list).end() && iter->second.address != ptr) {
        DEBUG_PRINT("\x1b[31m""Can't find ptr inside entry\n""\x1b[0m");
        exit(-1);
    }

    
    return iter->first;
}


void Cleanup(){
    DEBUG_PRINT("Cleaning up...\n");

    // kill(0, SIGTERM);
    pthread_join(swap_thread_id, NULL);
    DEBUG_PRINT("Swap Thread terminated\n");

    reg_msg *reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->reg_type = 0;
    reg->pid = getpid();
    commErrchk(write(register_fd, reg, sizeof(int)*2));
    DEBUG_PRINT(" De-registration done\n");

}

/* Swap in handler */
void swapin(int signum){
    DEBUG_PRINT("\x1b[31m""SIGUSR2 (Swap in) handler callback\n""\x1b[0m");
    // iterate swap entry list
    for(auto iter = swap_entry_list.cbegin(); iter != swap_entry_list.cend(); ){
        int index = iter->first;
        void * devPtr = (void *)iter->second.gpu_address;
        char * hosPtr = (char *)iter->second.cpu_address;
        size_t size = iter->second.size;

        cudaMalloc(&devPtr,size);
        cudaMemcpy(devPtr, hosPtr, size, cudaMemcpyHostToDevice);
        free(hosPtr);
        swap_entry_list.erase(iter++);
    }
}


/* Swap out handler */
void swapout(int signum){
    DEBUG_PRINT("\x1b[31m""SIGUSR1 (Swap out) handler callback\n""\x1b[0m");

    int ack;
    cudaError_t err;
    list<int> swap_list;
    
    evict_msg *msg = (evict_msg *)malloc(sizeof(evict_msg));
    commErrchk(read(decision_fd, msg, sizeof(int)*2));


    int start_idx = msg->start_idx;
    auto begin_iter = gpu_entry_list.find(msg->start_idx);
    auto end_iter = gpu_entry_list.find(msg->end_idx);
    end_iter++;
    
    DEBUG_PRINT("\x1b[31m""Swap out request [%d, %d]\n""\x1b[0m",msg->start_idx, msg->end_idx);

    for(auto iter = begin_iter; iter!=end_iter; iter++){
        int index = iter->first;
        swap_list.push_back(index);            
        const void* ptr = iter->second.address;
        size_t size = iter->second.size;
        
        /* Synchronous version */
        char * cpu =(char *)malloc(size);
        err = cudaMemcpy(cpu, ptr, size, cudaMemcpyDeviceToHost); // error check logic need to add
        add_swap_entry(&swap_entry_list, index, (const void *)ptr, (const void *)cpu, size);
        cudaFree((void *)ptr);
        DEBUG_PRINT("\x1b[31m""Swap out Address: %p\n""\x1b[0m", ptr);
    }
    
    // erasing entry inside above for loop cuase an error
    for(auto iter = swap_list.begin(); iter != swap_list.end(); iter++){
        gpu_entry_list.erase(*iter);
    }
}

void add_swap_entry(map<int,gswap>* entry_list, int index, const void* gpuPtr, const void* cpuPtr, size_t size){
    DEBUG_PRINT("add swap entry: {%d, [%p, %p, %d]}\n", index, gpuPtr, cpuPtr, size);
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