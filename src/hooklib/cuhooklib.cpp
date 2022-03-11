#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <signal.h>
#include <map>

#define REGISTRATION "/tmp/registration"
#define string(x) #x

#define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[customHook][%s:%3d:%20s()]: [%d] " fmt, \
__FILE__, __LINE__, __func__,getpid(), ##args)
#else
#define DEBUG_PRINT(fmt, args...)
#endif

using namespace std;

typedef struct _ENTRY{
    const void* address;
    size_t size;
}entry;

static int entry_index = 0;
static map<int,struct _ENTRY> gpu_entry_list;
static map<int,struct _ENTRY> cpu_entry_list;
static int init = 0;
int request_fd = -1;
int decision_fd = -1;
int register_fd = -1;



typedef enum{
    _cudaMalloc_, _cudaMemcpy_, _cudaMemcpyAsync_, _cudaFree_
}cudaAPI;

typedef struct _MSG_PACKET_EVICT{
    int start_idx;
    int end_idx;
}evict_msg;

typedef struct _MSG_PACKET_REGIST{
    int reg_type;
    int pid;
}reg_msg;

typedef struct _MSG_PACKET_REQUEST{
    cudaAPI type;
    cudaMemcpyKind kind;
    int entry_index;
    int size;
}req_msg;


void Init();
int SendRequest(const void* devPtr, cudaAPI type, cudaMemcpyKind kind, size_t size);
char * getcudaAPIString(cudaAPI type);
char * getcudaMemcpyKindString(cudaMemcpyKind kind);
void close_channels();
void close_channel(char * pipe_name);
void Cleanup();

void add_entry(map<int,entry>* entry_list, int index, const void* devPtr, size_t size);
void del_entry(map<int,entry>* entry_list, const void* devPtr);
int find_index_by_ptr(map<int,entry>* entry_list, const void* devPtr);

void sigusr1(int signum);
char * getcudaAPIString(cudaAPI type);
char * getcudaMemcpyKindString(cudaMemcpyKind kind);
void DEBUG_PRINT_ENTRY();


/* CUDA memory hook */
static cudaError_t (*lcudaMalloc)(void **, size_t) = (cudaError_t (*) (void**, size_t))dlsym(RTLD_NEXT,"cudaMalloc");
static cudaError_t (*lcudaMemcpy)(void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void*, const void*, size_t, cudaMemcpyKind))dlsym(RTLD_NEXT, "cudaMemcpy");
static cudaError_t (*lcudaMemcpyAsync)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind, cudaStream_t))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
static cudaError_t (*lcudaFree) (void*) = (cudaError_t (*) (void *))dlsym(RTLD_NEXT,"cudaFree");

cudaError_t cudaMalloc (void **devPtr, size_t size){   
    cudaError_t err;
    if(!init){
        Init();
        init = 1;
    }

    DEBUG_PRINT("cudaMalloc [%d]\n", size);

    SendRequest((const void *)*devPtr, _cudaMalloc_, cudaMemcpyHostToHost, size);
    err = lcudaMalloc(devPtr, size);
    add_entry(&gpu_entry_list, entry_index, (const void *)*devPtr, size);
    entry_index++;
    return err;
}


cudaError_t cudaMemcpy (void* dst, const void* src, size_t count, cudaMemcpyKind kind){
    cudaError_t err;

    DEBUG_PRINT("cudaMemcpy-%s [%d]\n",getcudaMemcpyKindString(kind),count);

    if(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice) {
        SendRequest((const void *)dst, _cudaMemcpy_, kind, count);
        err = lcudaMemcpy(dst, src, count, kind);
        add_entry(&gpu_entry_list,entry_index, (const void *)dst, count);
        entry_index++;
    }
    if(kind == cudaMemcpyDeviceToHost){
       SendRequest((const void *)src, _cudaMemcpy_, kind, count); 
       err = lcudaMemcpy(dst, src, count, kind );
       del_entry(&gpu_entry_list,(const void *)src);
    }
    
    return err;
}

cudaError_t cudaMemcpyAsync (void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str){
    cudaError_t err;

    DEBUG_PRINT("cudaMemcpyAsync-%s [%d]\n",getcudaMemcpyKindString(kind),count);
        
    if(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice) {
        SendRequest((const void *)dst, _cudaMemcpyAsync_, kind, count);
        lcudaMemcpyAsync(dst, src, count, kind, str );
        add_entry(&gpu_entry_list,entry_index,(const void *)dst, count);
        entry_index++;
    }
    if(kind == cudaMemcpyDeviceToHost){
       SendRequest((const void *)src, _cudaMemcpyAsync_, kind, count); 
       lcudaMemcpyAsync(dst, src, count, kind, str );
       del_entry(&gpu_entry_list,(const void *)src);
    }  

    return err;
}

cudaError_t cudaFree(void* devPtr){ /* free */
    
    DEBUG_PRINT("cudaFree\n");
    SendRequest((const void *)devPtr, _cudaFree_, cudaMemcpyHostToHost, 0);
    del_entry(&gpu_entry_list,(const void *)devPtr);

    return lcudaFree(devPtr);
}

void Init(){
    signal(SIGUSR1, sigusr1);
    if((register_fd = open(REGISTRATION, O_WRONLY)) < 0){
        DEBUG_PRINT("\x1b[31m""REGISTRATION CHANNEL OPEN FAIL\n""\x1b[0m");
        exit(-1);
    }

    reg_msg *reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->reg_type = 1;
    reg->pid = getpid();

    if(write(register_fd, reg, sizeof(int)*2) < 0){
        DEBUG_PRINT("\x1b[31m""REGISTRATION FAIL\n""\x1b[0m");
        exit(-1);
    }

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
    
    DEBUG_PRINT_ENTRY();
}

int SendRequest(const void* devPtr, cudaAPI type, cudaMemcpyKind kind, size_t size){
    DEBUG_PRINT_ENTRY();
    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));
    
    msg -> type = type;
    msg -> kind = kind;
    msg -> size = size;
    
    if(type == _cudaMalloc_ || ((type == _cudaMemcpy_ || type == _cudaMemcpyAsync_) && (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice))) \
        msg -> entry_index = entry_index;
    
    if(type == _cudaFree_ || ((type == _cudaMemcpy_ || type == _cudaMemcpyAsync_) && kind == cudaMemcpyDeviceToHost)) \
        msg -> entry_index = find_index_by_ptr(&gpu_entry_list, devPtr);

    if(write(request_fd, msg, sizeof(int)*4) < 0){
        DEBUG_PRINT("\x1b[31m""SendRequest WRITE fail [%s %s %d]\n""\x1b[0m" ,getcudaAPIString(type), getcudaMemcpyKindString(kind), size );
        exit(-1);
    }
    
    int ack;
    if(read(decision_fd, &ack, sizeof(int)) < 0){
        DEBUG_PRINT("\x1b[31m""SendRequest READ fail [%s %s %d]\n""\x1b[0m" , getcudaAPIString(type), getcudaMemcpyKindString(kind), size );
        exit(-1);
    }
    
}

void DEBUG_PRINT_ENTRY(){
    DEBUG_PRINT("Current Entry: ");
    auto iter = gpu_entry_list.begin();
    while(iter != gpu_entry_list.end()){
        fprintf(stderr, "{%d, [%p, %d]} ",iter->first, iter->second.address, iter->second.size);
        ++iter;
    }
    fprintf(stderr,"\n");
}

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

    reg_msg *reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->reg_type = 0;
    reg->pid = getpid();
    if(write(register_fd, reg, sizeof(int)*2) < 0){
        DEBUG_PRINT("\x1b[31m""DE-REGISTRATION FAIL\n""\x1b[0m");
        exit(-1);
    }
    DEBUG_PRINT(" De-registration done\n");

}

void sigusr1(int signum){
    signal(SIGUSR1, sigusr1);
    
    int ack;
    cudaError_t err;
    evict_msg *msg = (evict_msg *)malloc(sizeof(evict_msg));
    
    
    DEBUG_PRINT("\x1b[31m""SIGUSR1 (Swap out) handler callback\n""\x1b[0m");

    if(read(decision_fd, msg, sizeof(int)*2) < 0){
        DEBUG_PRINT("\x1b[31m"" Signal handler read failed\n""\x1b[0m");
        exit(-1);
    }

    auto begin_iter = gpu_entry_list.find(msg->start_idx);
    auto end_iter = gpu_entry_list.find(msg->end_idx);
    end_iter++;

    DEBUG_PRINT("\x1b[31m""Swap out request [%d, %d]\n""\x1b[0m",msg->start_idx, msg->end_idx);
    
    for(auto iter = begin_iter; iter!=end_iter; iter++){
        int index = iter->first;
        const void* ptr = iter->second.address;
        size_t size = iter->second.size;
        char * cpu =(char *)malloc(size);

        err = lcudaMemcpy(cpu, ptr, size, cudaMemcpyDeviceToHost); // error check logic need to add
        add_entry(&cpu_entry_list, index, (const void *)cpu, size);

        DEBUG_PRINT("\x1b[31m""Swap out Address: %p\n""\x1b[0m", ptr);

        lcudaFree((void *)ptr);  
    }
    
    if(write(request_fd, &ack, sizeof(int)) < 0){
        DEBUG_PRINT("\x1b[31m"" Signal handler write failed\n""\x1b[0m");
        exit(-1);
    }
}

char * getcudaAPIString(cudaAPI type){
    switch (type){
        case _cudaMalloc_:
            return string(_cudaMalloc_);
        case _cudaMemcpy_:
            return string(_cudaMemcpy_);
        case _cudaMemcpyAsync_:
            return string(_cudaMemcpyAsync_);
        case _cudaFree_:
            return string(_cudaFree_);
    }
}

char * getcudaMemcpyKindString(cudaMemcpyKind kind){
    switch (kind){
        case cudaMemcpyHostToHost:
            return string(cudaMemcpyHostToHost);
        case cudaMemcpyHostToDevice:
            return string(cudaMemcpyHostToDevice);
        case cudaMemcpyDeviceToHost:
            return string(cudaMemcpyDeviceToHost);
        case cudaMemcpyDeviceToDevice:
            return string(cudaMemcpyDeviceToDevice);
        case cudaMemcpyDefault:
            return string(cudaMemcpyDefault);
    }
}