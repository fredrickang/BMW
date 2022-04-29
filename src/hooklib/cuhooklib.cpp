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
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "hooklib.hpp"

using namespace std;

unsigned long long int swap_out_sz_tot = 0;
unsigned long long int swap_in_sz_tot = 0;
double swap_out_time = 0.0;
double swap_in_time = 0.0;

extern int prio;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}



cudaError_t cudaMalloc(void **devPtr, size_t size){   
    cudaError_t err;
    if(!init){
        Init();
        init = 1;
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
    
    if(pagetable.size() != 0){
        
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
                DEBUG_PRINT(RED "Re-direceted %p -> %p\n" RESET, dst, remapped_dst);
            }
        }else{
            if (pagetable.find((void *)src) != pagetable.end() ){
                remapped_src = (const void*)pagetable[(void *)src];
                DEBUG_PRINT(RED "Re-direceted %p -> %p\n" RESET, src, remapped_src);
            } 
        }
    }
    cudaError_t err = cudaSuccess;
    CHECK_CUDA(lcudaMemcpy(remapped_dst, remapped_src, size, kind));
    return err;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream){
    void* remapped_dev = devPtr;
    if(pagetable.size() != 0){
        if(pagetable.find(devPtr) != pagetable.end()){
            remapped_dev = pagetable[devPtr];
            DEBUG_PRINT(RED "Re-direceted %p -> %p\n" RESET, devPtr, remapped_dev);
        }
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
        }
        if(pagetable.find((void *)B) != pagetable.end()){
            remapped_B = pagetable[(void *)B];
        }
        if(pagetable.find((void *)C) != pagetable.end()){
            remapped_C = pagetable[(void *)C];
        }
    }

    cublasStatus_t err = lcublasSgemm(handle, transa, transb, m, n, k, alpha, (const float *)remapped_A, lda, (const float *)remapped_B, ldb, (const float *)beta, (float *)remapped_C, ldc);
    
    return err;
}


/* Swap-in handler */
void swapin(int signum){
    double si_s, si_e;
    si_s = what_time_is_it_now();
    DEBUG_PRINT(GREEN "Swap-in (SIGUSR2) handler callback\n" RESET);
    for(auto iter = swap_entry_list.begin(); iter != swap_entry_list.end(); iter++){
        int index = iter->first;
        size_t size = iter->second.size;
        void* new_address;
        void* old_address = iter->second.gpu_address;
        char* hostPtr = (char *)iter->second.cpu_address;

        CHECK_CUDA(cudaMalloc(&new_address, size, index));
        CHECK_CUDA(lcudaMemcpy(new_address, hostPtr, size, cudaMemcpyHostToDevice));
        free(hostPtr);
        /* page table update */
        DEBUG_PRINT(GREEN "Swap in Addr: %p, Size: %d\n" RESET, new_address, size);
        pagetable[old_address] = new_address;
        swap_in_sz_tot += size;
    } 
    swap_entry_list.clear();
    DEBUG_PRINT_SWAP();
    DEBUG_PRINT_ENTRY();
    DEBUG_PRINT_PAGETABLE();
    si_e = what_time_is_it_now();
    swap_in_time += (si_e - si_s);
}

/* Swap out handler */
void swapout(int signum){
    double so_s, so_e;
    so_s = what_time_is_it_now();
    DEBUG_PRINT(GREEN "Swap-out (SIGUSR1) handler callback\n" RESET);

    int ack;
    cudaError_t err;
    
    evict_msg *msg = (evict_msg *)malloc(sizeof(evict_msg));
    CHECK_COMM(read(decision_fd, msg, sizeof(int)*2));
    
    DEBUG_PRINT(GREEN "Swap-out Range [%d, %d]\n" RESET,msg->start_idx, msg->end_idx);
    
    for(int i = msg->start_idx; i <=msg->end_idx; i++){
        int index = i;
        size_t size = gpu_entry_list[index].size;
        void * devPtr = gpu_entry_list[index].address;
        void * hostPtr = (char *)malloc(size);

        CHECK_CUDA(lcudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost));
        add_swap_entry(&swap_entry_list, index, devPtr, hostPtr, size);

        CHECK_CUDA(cudaFree(devPtr));
            
        if(pagetable.size() != 0){
            pagetable.erase(devPtr);
        }
        DEBUG_PRINT(GREEN "Swap out Addr: %p, Size: %d\n" RESET, devPtr, size);
        swap_out_sz_tot += size;
    }
    DEBUG_PRINT_ENTRY();
    DEBUG_PRINT_SWAP();
    DEBUG_PRINT_PAGETABLE();
    so_e = what_time_is_it_now();
    swap_out_time += (so_e - so_s);
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
                CHECK_COMM(write(request_fd, msg, sizeof(int)*REQ_MSG_SIZE));
                DEBUG_PRINT(GREEN "Swap-out Complete\n" RESET);
                SWAP_OUT = true; // swapped flag on
                break;
            case SIGUSR2:
                if(SWAP_OUT) swapin(signum);
                CHECK_COMM(write(request_fd, msg, sizeof(int)*REQ_MSG_SIZE));
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

    CHECK_COMM(write(register_fd, reg, sizeof(int)*2))
    
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
    
    CHECK_COMM(write(request_fd, msg, sizeof(int)*3));
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
    
    CHECK_COMM(write(request_fd, msg, sizeof(int)*3));
    CHECK_COMM(read(decision_fd, &ack, sizeof(int)));
}

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


void Cleanup(){
    DEBUG_PRINT(BLUE "Cleaning up...\n" RESET);
    int ack;
    reg_msg *reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->reg_type = 0;
    reg->pid = getpid();
    CHECK_COMM(write(register_fd, reg, sizeof(int)*2));
    CHECK_COMM(read(decision_fd, &ack, sizeof(int)));
    DEBUG_PRINT(BLUE "==De-registration done==\n" RESET);
    
    //kill(0, SIGTERM);
    //pthread_join(swap_thread_id, NULL);
    DEBUG_PRINT(BLUE "Swap Thread terminated\n" RESET);

    /* LOG */
    // char logname[300];
    // snprintf(logname, 300, "/home/xavier5/BMW/darknet/logs/swap_detail_log_%d.log",prio);
    // FILE *fp = fopen(logname, "a");
    // fprintf(fp, "swap_in_sz_tot: %lld\n", swap_in_sz_tot);
    // fprintf(fp, "swap_out_sz_tot: %lld\n", swap_out_sz_tot);
    // fprintf(fp, "swap_in_time: %f\n", swap_in_time);
    // fprintf(fp, "swap_out_time: %f\n", swap_out_time);
    // fclose(fp);
    DEBUG_PRINT(GREEN "==Termination Sequence Done==\n" RESET);
}



void add_swap_entry(map<int,gswap>* entry_list, int index, void* gpuPtr, void* cpuPtr, size_t size){
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



cudaError_t cudaLaunchKernel( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream){
    cudaError_t err;
    if(pagetable.size() != 0){
        int arg_size = *(int *)args[0];
        
        for(int i = 1; i < arg_size+1; i++){
            
            if(pagetable.find(*(void **)args[i]) != pagetable.end()){
                //fprintf(stderr, " -> %p", pagetable[*(void **)args[i]]);
                args[i] = (void *)&pagetable[*(void **)args[i]];
            }
            //fprintf(stderr,"\n"RESET); 
        }
    }
    err = cudaSuccess;
    CHECK_CUDA(lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
    return err;
}

// static int possible_arg_sizes[] = {1,2,4,8,12};
// static int num_arg_size = 5;
// // bool is_arg_size(int diff){
// //     for(int i = 0; i < num_arg_size; i++){
// //         if(possible_arg_sizes[i] == diff) return true;
// //     }
// //     return false;
// // }
// bool is_arg_size(int diff){
//     if(diff > 100 || diff < -100) return false;
//     return true;
// }


/* cudaLaunchKernel with assuming the number of args (deprecated)*/
// cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream){
//     cudaError_t err;
//     if(pagetable.size() != 0){
//         void * zero_args = (void *)0x100000001;
//         DEBUG_PRINT("cudaLaunchKernel\n");
//         if(args[0] != zero_args){
//             int arg_size = 0;
//             int Done = 0;
//             while(1){
//                 int diff = (char *)args[arg_size] - (char *)args[arg_size+1];
//                 fprintf(stderr, "%d\n",diff);
//                 if( !is_arg_size((char *)args[arg_size] - (char *)args[arg_size+1]) ) Done = 1;
//                 DEBUG_PRINT(RED"args[%d]: %p\n",arg_size, args[arg_size]);
//                 DEBUG_PRINT(RED"args[%d]: %p\n",arg_size+1, args[arg_size+1]);
//                 DEBUG_PRINT(RED"%p", *(void **)args[arg_size]);
//                 if(pagetable.find(*(void **)args[arg_size]) != pagetable.end()){
//                     //fprintf(stderr, " -> %p", pagetable[*(void **)args[arg_size]]);
//                     args[arg_size] = (void *)&pagetable[*(void **)args[arg_size]];
//                 }
//                 //fprintf(stderr,"\n"RESET);
//                 if(Done) break;
//                 arg_size++;
//             }
//         }
//     }
//     err = cudaSuccess;
//     CHECK_CUDA(lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
//     return err;
// }



// cudaError_t cudaLaunchKernel( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream){
//     cudaError_t err;
//     int arg_size = *(int *)args[0];
//     DEBUG_PRINT(RED"arg_size :%d\n"RESET,arg_size);
//     void *new_args[arg_size];

//     for(int i = 1; i < arg_size+1; i++){
//         DEBUG_PRINT(RED"Old args: %p\n"RESET,*(void **)args[i]);
//         if(pagetable.find(*(void **)args[i])!= pagetable.end()){
//             DEBUG_PRINT(RED"New args: %p\n"RESET,pagetable[*(void **)args[i]]);
//             new_args[i-1] = (void *)&pagetable[*(void **)args[i]];
//         } 
//         else new_args[i-1] = args[i];
//     }
    
//     err = cudaSuccess;
//     CHECK_CUDA(lcudaLaunchKernel(func, gridDim, blockDim, new_args, sharedMem, stream));
//     return err;
// }

